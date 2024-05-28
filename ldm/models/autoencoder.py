import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from model.mixins import ImageLoggerMixin
from ldm.modules.diffusionmodules.model import Encoder, Decoder, Decoder_Mix
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma
'''
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt
'''
import random

import torchvision.transforms as transforms
from .discriminator import *
from torchvision.ops import roi_align
from basicsr.losses.gan_loss import *
from basicsr.losses.basic_loss import *

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False
                 ):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix=""):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+postfix)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val"+postfix)

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params_list = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(
            self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
            if log_ema or self.use_ema:
                with self.ema_scope():
                    xrec_ema, posterior_ema = self(x)
                    if x.shape[1] > 3:
                        # colorize with random projection
                        assert xrec_ema.shape[1] > 3
                        xrec_ema = self.to_rgb(xrec_ema)
                    log["samples_ema"] = self.decode(torch.randn_like(posterior_ema.sample()))
                    log["reconstructions_ema"] = xrec_ema
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

class AutoencoderKLResi(pl.LightningModule,ImageLoggerMixin):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 fusion_w=1.0,
                 freeze_dec=True,
                 synthesis_data=False,
                 use_usm=False,
                 test_gt=False,
                 learning_rate=5e-5
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder_Mix(**ddconfig)
        self.decoder.fusion_w = fusion_w
        self.loss = instantiate_from_config(lossconfig)
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            missing_list = self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            missing_list = []

        print('>>>>>>>>>>>>>>>>>missing>>>>>>>>>>>>>>>>>>>')
        print(missing_list)
        self.synthesis_data = synthesis_data
        self.use_usm = use_usm
        self.test_gt = test_gt

        if freeze_dec:
            for name, param in self.named_parameters():
                if 'fusion_layer' in name:
                    param.requires_grad = True
                # elif 'encoder' in name:
                #     param.requires_grad = True
                # elif 'quant_conv' in name and 'post_quant_conv' not in name:
                #     param.requires_grad = True
                elif 'loss.discriminator' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        print('>>>>>>>>>>>>>>>>>trainable_list>>>>>>>>>>>>>>>>>>>')
        trainable_list = []
        for name, params in self.named_parameters():
            if params.requires_grad:
                trainable_list.append(name)
        print(trainable_list)

        print('>>>>>>>>>>>>>>>>>Untrainable_list>>>>>>>>>>>>>>>>>>>')
        untrainable_list = []
        for name, params in self.named_parameters():
            if not params.requires_grad:
                untrainable_list.append(name)
        print(untrainable_list)
        # untrainable_list = list(set(trainable_list).difference(set(missing_list)))
        # print('>>>>>>>>>>>>>>>>>untrainable_list>>>>>>>>>>>>>>>>>>>')
        # print(untrainable_list)

    # def init_from_ckpt(self, path, ignore_keys=list()):
    #     sd = torch.load(path, map_location="cpu")["state_dict"]
    #     keys = list(sd.keys())
    #     for k in keys:
    #         for ik in ignore_keys:
    #             if k.startswith(ik):
    #                 print("Deleting key {} from state_dict.".format(k))
    #                 del sd[k]
    #     self.load_state_dict(sd, strict=False)
    #     print(f"Restored from {path}")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if 'first_stage_model' in k:
                sd[k[18:]] = sd[k]
                del sd[k]
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Encoder Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        return missing

    def encode(self, x):
        h, enc_fea = self.encoder(x, return_fea=True)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # posterior = h
        return posterior, enc_fea

    def encode_gt(self, x, new_encoder):
        h = new_encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, moments

    def decode(self, z, enc_fea):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, enc_fea)
        return dec

    def forward(self, input, latent, sample_posterior=True):
        posterior, enc_fea_lq = self.encode(input)
        dec = self.decode(latent, enc_fea_lq)
        return dec, posterior

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        _, c_, h_, w_ = self.latent.size()
        if b == self.configs.data.params.batch_size:
            if not hasattr(self, 'queue_size'):
                self.queue_size = self.configs.data.params.train.params.get('queue_size', b*50)
            if not hasattr(self, 'queue_lr'):
                assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
                self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
                _, c, h, w = self.gt.size()
                self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_sample = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_latent = torch.zeros(self.queue_size, c_, h_, w_).cuda()
                self.queue_ptr = 0
            if self.queue_ptr == self.queue_size:  # the pool is full
                # do dequeue and enqueue
                # shuffle
                idx = torch.randperm(self.queue_size)
                self.queue_lr = self.queue_lr[idx]
                self.queue_gt = self.queue_gt[idx]
                self.queue_sample = self.queue_sample[idx]
                self.queue_latent = self.queue_latent[idx]
                # get first b samples
                lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
                gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
                sample_dequeue = self.queue_sample[0:b, :, :, :].clone()
                latent_dequeue = self.queue_latent[0:b, :, :, :].clone()
                # update the queue
                self.queue_lr[0:b, :, :, :] = self.lq.clone()
                self.queue_gt[0:b, :, :, :] = self.gt.clone()
                self.queue_sample[0:b, :, :, :] = self.sample.clone()
                self.queue_latent[0:b, :, :, :] = self.latent.clone()

                self.lq = lq_dequeue
                self.gt = gt_dequeue
                self.sample = sample_dequeue
                self.latent = latent_dequeue
            else:
                # only do enqueue
                self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
                self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
                self.queue_sample[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.sample.clone()
                self.queue_latent[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.latent.clone()
                self.queue_ptr = self.queue_ptr + b

    def get_input(self, batch):
        input = batch['sr']
        gt = batch['hq']
        latent = batch['latent']
        # sample = batch['sample']

        assert not torch.isnan(latent).any()

        input = input.to(memory_format=torch.contiguous_format).float()
        gt = gt.to(memory_format=torch.contiguous_format).float()
        latent = latent.to(memory_format=torch.contiguous_format).float() / 0.18215

        #gt = gt * 2.0 - 1.0
        #input = input * 2.0 - 1.0
        # sample = sample * 2.0 -1.0

        return input, gt, latent #, sample

    def training_step(self, batch, batch_idx, optimizer_idx):

        inputs, gts, latents = self.get_input(batch)
        reconstructions, posterior = self(inputs, latents)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(gts, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(gts, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs, gts, latents  = self.get_input(batch)

        reconstructions, posterior = self(inputs, latents)
        aeloss, log_dict_ae = self.loss(gts, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(gts, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  # list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        
        x, gts, latents = self.get_input(batch)
        x = x.to(self.device)
        latents = latents.to(self.device)
        # samples = samples.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x, latents)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                gts = self.to_rgb(gts)
                # samples = self.to_rgb(samples)
                xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = (xrec+1)/2
        log["inputs"] = x
        log["gts"] = (gts + 1)/2.
        # log["samples"] = samples
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class AutoencoderKLResiWD(pl.LightningModule,ImageLoggerMixin):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 fusion_w=1.0,
                 freeze_dec=True,
                 synthesis_data=False,
                 use_usm=False,
                 test_gt=False,
                 learning_rate=5e-5,
                 left_eye_ckpt = None,
                 right_eye_ckpt = None,
                 mouth_ckpt= None
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder_Mix(**ddconfig)
        self.decoder.fusion_w = fusion_w
        self.loss = instantiate_from_config(lossconfig)
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate

        '''init_local_discriminator'''
        self.net_d_left_eye = FacialComponentDiscriminator()
        self.net_d_right_eye = FacialComponentDiscriminator()
        self.net_d_mouth = FacialComponentDiscriminator()
        self.net_d_left_eye.train()
        self.net_d_right_eye.train()
        self.net_d_mouth.train()
        self.gan_loss = GANLoss('vanilla',real_label_val=1.0, fake_label_val=0.0,loss_weight = 0.1)
        
        self.cri_l1 = L1Loss(loss_weight=1.0, reduction='mean')
        
        if left_eye_ckpt is not None:
            self.net_d_left_eye.load_state_dict(torch.load(left_eye_ckpt))
            self.net_d_right_eye.load_state_dict(torch.load(right_eye_ckpt))
            self.net_d_mouth_eye.load_state_dict(torch.load(mouth_ckpt))


        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            missing_list = self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            missing_list = []

        print('>>>>>>>>>>>>>>>>>missing>>>>>>>>>>>>>>>>>>>')
        print(missing_list)
        self.synthesis_data = synthesis_data
        self.use_usm = use_usm
        self.test_gt = test_gt

        if freeze_dec:
            for name, param in self.named_parameters():
                if 'fusion_layer' in name:
                    param.requires_grad = True
                # elif 'encoder' in name:
                #     param.requires_grad = True
                # elif 'quant_conv' in name and 'post_quant_conv' not in name:
                #     param.requires_grad = True
                elif 'loss.discriminator' in name:
                    param.requires_grad = True
                elif 'net_d' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        print('>>>>>>>>>>>>>>>>>trainable_list>>>>>>>>>>>>>>>>>>>')
        trainable_list = []
        for name, params in self.named_parameters():
            if params.requires_grad:
                trainable_list.append(name)
        print(trainable_list)

        print('>>>>>>>>>>>>>>>>>Untrainable_list>>>>>>>>>>>>>>>>>>>')
        untrainable_list = []
        for name, params in self.named_parameters():
            if not params.requires_grad:
                untrainable_list.append(name)
        print(untrainable_list)

        self.automatic_optimization = False 
        self.mouths = None
        # untrainable_list = list(set(trainable_list).difference(set(missing_list)))
        # print('>>>>>>>>>>>>>>>>>untrainable_list>>>>>>>>>>>>>>>>>>>')
        # print(untrainable_list)

    # def init_from_ckpt(self, path, ignore_keys=list()):
    #     sd = torch.load(path, map_location="cpu")["state_dict"]
    #     keys = list(sd.keys())
    #     for k in keys:
    #         for ik in ignore_keys:
    #             if k.startswith(ik):
    #                 print("Deleting key {} from state_dict.".format(k))
    #                 del sd[k]
    #     self.load_state_dict(sd, strict=False)
    #     print(f"Restored from {path}")
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

        #print('gt',(self.gt.max(),self.gt.min()),self.gt.shape)
        all_eyes = roi_align(self.gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes_gt = all_eyes[0::2, :, :, :]
        self.right_eyes_gt = all_eyes[1::2, :, :, :]
        self.mouths_gt = roi_align(self.gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
        #print('gt',(self.gt.max(),self.gt.min()),self.gt.shape,self.left_eyes_gt.shape,self.right_eyes_gt.shape,self.mouths_gt.shape)
        # output
        #print('sr',(self.output.max(),self.output.min()),self.output.shape)
        all_eyes = roi_align(self.output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes = all_eyes[0::2, :, :, :]
        self.right_eyes = all_eyes[1::2, :, :, :]
        self.mouths = roi_align(self.output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio


    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if 'first_stage_model' in k:
                sd[k[18:]] = sd[k]
                del sd[k]
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Encoder Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        return missing

    def encode(self, x):
        h, enc_fea = self.encoder(x, return_fea=True)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # posterior = h
        return posterior, enc_fea

    def encode_gt(self, x, new_encoder):
        h = new_encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, moments

    def decode(self, z, enc_fea):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, enc_fea)
        return dec

    def forward(self, input, latent, sample_posterior=True):
        posterior, enc_fea_lq = self.encode(2*input-1)
        dec = self.decode(latent, enc_fea_lq)
        return dec, posterior

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        _, c_, h_, w_ = self.latent.size()
        if b == self.configs.data.params.batch_size:
            if not hasattr(self, 'queue_size'):
                self.queue_size = self.configs.data.params.train.params.get('queue_size', b*50)
            if not hasattr(self, 'queue_lr'):
                assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
                self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
                _, c, h, w = self.gt.size()
                self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_sample = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_latent = torch.zeros(self.queue_size, c_, h_, w_).cuda()
                self.queue_ptr = 0
            if self.queue_ptr == self.queue_size:  # the pool is full
                # do dequeue and enqueue
                # shuffle
                idx = torch.randperm(self.queue_size)
                self.queue_lr = self.queue_lr[idx]
                self.queue_gt = self.queue_gt[idx]
                self.queue_sample = self.queue_sample[idx]
                self.queue_latent = self.queue_latent[idx]
                # get first b samples
                lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
                gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
                sample_dequeue = self.queue_sample[0:b, :, :, :].clone()
                latent_dequeue = self.queue_latent[0:b, :, :, :].clone()
                # update the queue
                self.queue_lr[0:b, :, :, :] = self.lq.clone()
                self.queue_gt[0:b, :, :, :] = self.gt.clone()
                self.queue_sample[0:b, :, :, :] = self.sample.clone()
                self.queue_latent[0:b, :, :, :] = self.latent.clone()

                self.lq = lq_dequeue
                self.gt = gt_dequeue
                self.sample = sample_dequeue
                self.latent = latent_dequeue
            else:
                # only do enqueue
                self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
                self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
                self.queue_sample[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.sample.clone()
                self.queue_latent[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.latent.clone()
                self.queue_ptr = self.queue_ptr + b

    def get_input(self, batch):
        input = batch['sr']
        gt = batch['hq']
        latent = batch['latent']
        # sample = batch['sample']
        self.loc_left_eyes = batch['loc_left_eye']
        self.loc_right_eyes = batch['loc_right_eye']
        self.loc_mouths = batch['loc_mouth']
        self.gt = gt

        assert not torch.isnan(latent).any()

        input = input.to(memory_format=torch.contiguous_format).float()
        gt = gt.to(memory_format=torch.contiguous_format).float()
        latent = latent.to(memory_format=torch.contiguous_format).float() / 0.18215

        #gt = gt * 2.0 - 1.0
        #input = input * 2.0 - 1.0
        # sample = sample * 2.0 -1.0

        return input, gt, latent #, sample

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
    

    def training_step(self, batch, batch_idx ):
        inputs, gts, latents = self.get_input(batch)
        reconstructions, posterior = self(inputs, latents)
        self.output = reconstructions
        self.get_roi_regions()
        optimizer_g, optimizer_d,optimizer_d_left_eye, optimizer_d_right_eye, optimizer_d_mouth = self.optimizers()
      
        # train encoder+decoder+logvar
        loss_total = 0 

        aeloss, log_dict_ae = self.loss(gts, reconstructions, posterior, optimize_d=False, global_step=self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        loss_total += aeloss
        loss, loss_dict_ = self.d_losses(optimize_d=False)
        self.log_dict(loss_dict_, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("gloss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        loss_total += loss
        
        optimizer_g.zero_grad()
        self.manual_backward(loss_total)
        optimizer_g.step()


        # train the discriminator
        loss_total = 0 

        loss, loss_dict_ = self.d_losses(optimize_d=True)
        self.log_dict(loss_dict_, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("dloss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        loss_total += loss

        #optimizer_d.zero_grad()
        optimizer_d_left_eye.zero_grad()
        optimizer_d_right_eye.zero_grad()
        optimizer_d_mouth.zero_grad()
        self.manual_backward(loss_total)
        #optimizer_d.step()
        optimizer_d_left_eye.step()
        optimizer_d_right_eye.step()
        optimizer_d_mouth.step()

        return self.log_dict

        



    def validation_step(self, batch, batch_idx):
        # inputs, gts, latents  = self.get_input(batch)

        # reconstructions, posterior = self(inputs, latents)
        # aeloss, log_dict_ae = self.loss(gts, reconstructions, posterior, 0, self.global_step,
        #                                 last_layer=self.get_last_layer(), split="val")

        # discloss, log_dict_disc = self.loss(gts, reconstructions, posterior, 1, self.global_step,
        #                                     last_layer=self.get_last_layer(), split="val")

        # self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        # self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)
        # return self.log_dict
        pass


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  # list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        
        self.optimizer_d_left_eye = torch.optim.AdamW(self.net_d_left_eye.parameters(), lr=lr)
        self.optimizer_d_right_eye = torch.optim.AdamW(self.net_d_right_eye.parameters(), lr=lr)
        self.optimizer_d_mouth = torch.optim.AdamW(self.net_d_mouth.parameters(), lr=lr)


        return opt_ae, opt_disc, self.optimizer_d_left_eye,self.optimizer_d_right_eye, self.optimizer_d_mouth 

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        
        x, gts, latents = self.get_input(batch)
        x = x.to(self.device)
        latents = latents.to(self.device)
        # samples = samples.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x, latents)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                gts = self.to_rgb(gts)
                # samples = self.to_rgb(samples)
                xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = (xrec+1)/2
        log["inputs"] = x
        log["gts"] = (gts + 1)/2.
        # log["samples"] = samples
        if self.mouths is not None:
            log["left_eye"] = (self.left_eyes.detach() +1 )/2
            log["left_eye_gt"] = (self.left_eyes_gt.detach() +1 )/2
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


