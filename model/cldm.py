from typing import Mapping, Any
import copy
from collections import OrderedDict

import einops
from einops import rearrange, repeat
import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as F
import math
from torchvision import transforms as tt
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils.common import frozen_module
from .spaced_sampler import SpacedSampler
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from taming.modules.losses.vqperceptual import * 
from basicsr.losses.gan_loss import *
from basicsr.losses.basic_loss import *
from torchvision.ops import roi_align
from .discriminator import *
from utils.common import instantiate_from_config, load_state_dict
        


class TwoStreamControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, learning_rate,
        preprocess_config, withD=False, gamma=200, sd_locked=True, pre_path=None,sync_path=None, synch_control=True,
                 control_mode='canny', ckpt_path_ctr=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        if sync_path is not None:
            self.sync_control_weights_from_base_checkpoint(sync_path)

        self.automatic_optimization = False

        self.control_key = control_key
        self.sd_locked = sd_locked
        self.control_mode = control_mode

        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        self.withD = withD
        self.gamma = gamma

        # instantiate discriminator
        self.init_discriminators()
        
        # instantiate preprocess module (SwinIR)
        self.current_iter = 0
        self.preprocess_model = instantiate_from_config(preprocess_config)
        if pre_path is not None:
            new_state_dict = OrderedDict()
            for k, v in torch.load(pre_path).items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.preprocess_model.load_state_dict(new_state_dict)

        # instantiate whole module
        if ckpt_path_ctr is not None:
            self.load_control_ckpt(ckpt_path_ctr=ckpt_path_ctr)

        frozen_module(self.preprocess_model)

    def init_discriminators(self):
        # Init models
        self.net_d_left_eye = FacialComponentDiscriminator()
        self.net_d_right_eye = FacialComponentDiscriminator()
        self.net_d_mouth = FacialComponentDiscriminator()
        self.net_d_left_eye.train()
        self.net_d_right_eye.train()
        self.net_d_mouth.train()
        self.mouths = None

        # Init losses
        self.gan_loss = GANLoss('vanilla',real_label_val=1.0, fake_label_val=0.0,loss_weight = 0.1)
        self.cri_l1 = L1Loss(loss_weight=1.0, reduction='mean')
        return

    def load_control_ckpt(self, ckpt_path_ctr):
        ckpt = torch.load(ckpt_path_ctr)
        load_state_dict(self,ckpt,strict=False)
        print(['CONTROL WEIGHTS LOADED',ckpt_path_ctr])

    def sync_control_weights_from_base_checkpoint(self, path):
        ckpt_base = torch.load(path)  # load the base model checkpoints
        for key in list(ckpt_base['state_dict'].keys()):
                if "diffusion_model." in key and  'in_layers' not in key and 'op.weight' not in key:
                    if 'control_model.control' + key[15:] in self.state_dict().keys():
                        ckpt_base['state_dict']['control_model.control' + key[15:]] = ckpt_base['state_dict'][key]
 
        res_sync = self.load_state_dict(ckpt_base['state_dict'], strict=False)
        print(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        self.gt = batch['jpg'].permute(0,3,1,2)
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
        # apply preprocess model
        control = self.preprocess_model(control)
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)
        if 'loc_left_eye' in batch:
            self.loc_left_eyes = batch['loc_left_eye']
            self.loc_right_eyes = batch['loc_right_eye']
            self.loc_mouths = batch['loc_mouth']
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control])


    def apply_model(self, x_noisy, t, cond, precomputed_hint=False, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_hint = torch.cat(cond['c_latent'], 1) if self.control_mode != 'multi_control' else cond['c_latent'][0]

        eps = self.control_model(
            x=x_noisy, hint=cond_hint, timesteps=t, context=cond_txt, base_model=diffusion_model, precomputed_hint=precomputed_hint
        )

        return eps

    def apply_condition_encoder(self, control,AFR=False):
        if AFR:
            c_latent_meanvar,encfea = self.first_stage_model.encoder(control * 2 -1,return_fea=True)
        else:
            c_latent_meanvar = self.first_stage_model.encoder(control * 2 -1)
        c_latent_meanvar = self.first_stage_model.quant_conv(c_latent_meanvar)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode()
        c_latent = c_latent * self.scale_factor
        if AFR:
            return c_latent, encfea
        else:
            return c_latent
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def decode_first_stage(self, z, enc_fea=None,predict_cids=False, force_not_quantize=False, hint=None):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        if hint is None:
            if enc_fea is None:
                return self.first_stage_model.decode(z)
            else:
                return self.first_stage_model.decode(z,enc_fea)
        else:
            return self.first_stage_model.decode(z, hint=hint)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["lq"] = c_lq

        samples = self.sample_log(
            # TODO: remove c_concat from cond
            c_cat,steps=sample_steps
        )
        x_samples = samples.clamp(0, 1)
        log["samples"] = x_samples
        ''' 
        # check region accuracy
        if self.mouths is not None:
            log["left_eye"] = (self.left_eyes.detach() +1 )/2
            log["left_eye_gt"] = (self.left_eyes_gt.detach() +1 )/2
            print(log["left_eye"].max(),log["left_eye"].min())
        '''
        return log

    @torch.no_grad()
    def sample_log(self, control, steps):
        sampler = SpacedSampler(self)
        height, width = control.size(-2), control.size(-1)
        n_samples = len(control)
        shape = (n_samples, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=self.device, dtype=torch.float32)
        samples = sampler.sample(
            steps, shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=None,
            color_fix_type='wavelet'
        )
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
   
        self.optimizer_d_left_eye = torch.optim.AdamW(self.net_d_left_eye.parameters(), lr=lr)
        self.optimizer_d_right_eye = torch.optim.AdamW(self.net_d_right_eye.parameters(), lr=lr)
        self.optimizer_d_mouth = torch.optim.AdamW(self.net_d_mouth.parameters(), lr=lr)
  
        return opt,self.optimizer_d_left_eye,self.optimizer_d_right_eye,self.optimizer_d_mouth 
        #return [opt],[]

    def validation_step(self, batch, batch_idx):
        # TODO: 
        pass
    
    def training_step(self, batch, batch_idx):
        '''get data'''
        self.current_iter = batch_idx
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss_dict = self.shared_step(batch)
        
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        
    
    def shared_step(self, batch,**kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        optimizer_g, optimizer_d_left_eye, optimizer_d_right_eye, optimizer_d_mouth = self.optimizers()


        # Diffusion loss
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        model_output,noise = self(x, c,t,img_output=False)
        loss_total, loss_dict = 0,{}

        loss, loss_dict_ = self.p_losses(model_output,t,noise)
        loss_total += loss
        loss_dict.update(loss_dict_)
        
        if self.withD:
        # Generator loss
            t = torch.randint(0, self.gamma, (x.shape[0],), device=self.device).long()
            model_output,noise = self(x, c,t,img_output=True)
            
            loss, loss_dict_ = self.p_losses(model_output,t,noise)
            loss_total += loss
            loss_dict.update(loss_dict_)
    
            loss, loss_dict_ = self.d_losses(optimize_d=False)
            loss_total += loss
            loss_dict.update(loss_dict_)
            
        optimizer_g.zero_grad()
        self.manual_backward(loss_total)
        optimizer_g.step()
        
        if self.withD:
        # Discriminator loss
            loss, loss_dict_ = self.d_losses(optimize_d=True)
            loss_dict.update(loss_dict_)

            optimizer_d_left_eye.zero_grad()
            optimizer_d_right_eye.zero_grad()
            optimizer_d_mouth.zero_grad()
            self.manual_backward(loss)
            optimizer_d_left_eye.step()
            optimizer_d_right_eye.step()
            optimizer_d_mouth.step()
        
        return loss_dict
    

    def forward(self, x, c,t, img_output=False,noise=None,*args, **kwargs):
        ####Generator
        
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        
        
        noise = default(noise,lambda: torch.randn_like(x))
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, c)
        if img_output:
            model_output_ = self.predict_start_from_noise(x_noisy,t,model_output)
            self.output =self.decode_first_stage(model_output_)

        return model_output,noise

     
    
    def d_losses(self, optimize_d = False):
        
        #get location
        self.get_roi_regions()
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if optimize_d:
            fake_d_pred, _ = self.net_d_left_eye(self.left_eyes.detach())
            real_d_pred, _ = self.net_d_left_eye(self.left_eyes_gt)
            l_d_left_eye = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_left_eye'] = l_d_left_eye

            # right eye
            fake_d_pred, _ = self.net_d_right_eye(self.right_eyes.detach())
            real_d_pred, _ = self.net_d_right_eye(self.right_eyes_gt)
            l_d_right_eye = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_right_eye'] = l_d_right_eye

            # mouth
            fake_d_pred, _ = self.net_d_mouth(self.mouths.detach())
            real_d_pred, _ = self.net_d_mouth(self.mouths_gt)
            l_d_mouth = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_mouth'] = l_d_mouth


            return l_d_left_eye + l_d_right_eye + loss_dict['l_d_mouth'], loss_dict
        
        else:
            l_g_total = 0
            fake_left_eye, fake_left_eye_feats = self.net_d_left_eye(self.left_eyes, return_feats=True)
            l_g_gan = self.gan_loss(fake_left_eye, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_left_eye'] = l_g_gan
            # right eye
            fake_right_eye, fake_right_eye_feats = self.net_d_right_eye(self.right_eyes, return_feats=True)
            l_g_gan = self.gan_loss(fake_right_eye, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_right_eye'] = l_g_gan
            # mouth
            fake_mouth, fake_mouth_feats = self.net_d_mouth(self.mouths, return_feats=True)
            l_g_gan = self.gan_loss(fake_mouth, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_mouth'] = l_g_gan

            _, real_left_eye_feats = self.net_d_left_eye(self.left_eyes_gt, return_feats=True)
            _, real_right_eye_feats = self.net_d_right_eye(self.right_eyes_gt, return_feats=True)
            _, real_mouth_feats = self.net_d_mouth(self.mouths_gt, return_feats=True)

            def _comp_style(feat, feat_gt, criterion):
                return criterion(self._gram_mat(feat[0]), self._gram_mat(
                    feat_gt[0].detach())) * 0.5 + criterion(
                        self._gram_mat(feat[1]), self._gram_mat(feat_gt[1].detach()))

            # facial component style loss
            comp_style_loss = 0
            comp_style_loss += _comp_style(fake_left_eye_feats, real_left_eye_feats, self.cri_l1)
            comp_style_loss += _comp_style(fake_right_eye_feats, real_right_eye_feats, self.cri_l1)
            comp_style_loss += _comp_style(fake_mouth_feats, real_mouth_feats, self.cri_l1)
            comp_style_loss = comp_style_loss * 200
            l_g_total += comp_style_loss
            loss_dict['l_g_comp_style_loss'] = comp_style_loss

            return l_g_total, loss_dict
    
    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
    
    def p_losses(self, model_output, t,noise=None):
                 
        loss_dict = {}
        prefix = 'train'
        target = noise
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})      

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    def get_roi_regions(self, eye_out_size=80, mouth_out_size=120):
        face_ratio = int(512 / 512)
        eye_out_size *= face_ratio
        mouth_out_size *= face_ratio

        rois_eyes = []
        rois_mouths = []
        for b in range(self.loc_left_eyes.size(0)):  # loop for batch size
            # left eye and right eye
            img_inds = self.loc_left_eyes.new_full((2, 1), b)
            bbox = torch.stack([self.loc_left_eyes[b, :], self.loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
            rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
            rois_eyes.append(rois)
            # mouse
            img_inds = self.loc_left_eyes.new_full((1, 1), b)
            rois = torch.cat([img_inds, self.loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
            rois_mouths.append(rois)

        rois_eyes = torch.cat(rois_eyes, 0).to(self.device)
        rois_mouths = torch.cat(rois_mouths, 0).to(self.device)

        # real images
        all_eyes = roi_align(self.gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes_gt = all_eyes[0::2, :, :, :]
        self.right_eyes_gt = all_eyes[1::2, :, :, :]
        self.mouths_gt = roi_align(self.gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
        all_eyes = roi_align(self.output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes = all_eyes[0::2, :, :, :]
        self.right_eyes = all_eyes[1::2, :, :, :]
        self.mouths = roi_align(self.output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio



