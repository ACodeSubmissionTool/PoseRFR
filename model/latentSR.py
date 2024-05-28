from typing import Mapping, Any
import copy
from collections import OrderedDict

import einops
from einops import rearrange, repeat
import numpy as np
import torch
import torch as th
import torch.nn as nn

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

class LatentSR(LatentDiffusion):
    # class LatentSR contains autoencoder and a latent-space SR predictor
    def __init__(self,
                 learning_rate,
                 sr_config,
                 *args,
                 **kwargs
                 ) -> 'LatentSR':
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.SRmodel = instantiate_from_config(sr_config)

    @torch.no_grad()
    def get_input(self, batch, *args, **kwargs):
        ''' batch['hint'] -> LR, batch['jpg'] -> HR'''
        lr_latent = batch['hint']* 2 - 1
        hr_latent = batch['jpg']

        def tran(x):
            if len(x.shape) == 3:
                x = x[..., None]
            x = rearrange(x, 'b h w c -> b c h w')
            x = x.to(memory_format=torch.contiguous_format).float()
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            return z

        lr_latent = tran(lr_latent)
        hr_latent = tran(hr_latent)
        return dict(lr_latent=lr_latent,hr_latent=hr_latent)
        
    def shared_step(self, batch,**kwargs):
        x = self.get_input(batch)
        loss = self(x)
        return loss
    

    def forward(self, x, *args, **kwargs):
        prefix = 'train'
        loss_dict = {}
        #print(x)
        lr_latent = x['lr_latent']
        hr_latent = x['hr_latent']
        t = torch.zeros((lr_latent.shape[0],), device=self.device).long()
        sr_latent = self.SRmodel(lr_latent,t)
        loss = self.get_loss(sr_latent, hr_latent)
        loss_dict.update({f'{prefix}/loss': loss.mean()})
            
        return loss,loss_dict
    
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

        loss, loss_dict = self.shared_step(batch)
        
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def configure_optimizers(self):
            lr = self.learning_rate
            params = list(self.SRmodel.parameters())
            opt = torch.optim.AdamW(params, lr=lr)
            return opt
    
    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        x = self.get_input(batch)
        
        lr_latent = x['lr_latent']
        hr_latent = x['hr_latent']
        t = torch.zeros((lr_latent.shape[0],), device=self.device).long()
        sr_latent = self.SRmodel(lr_latent,t).detach()

        def _decode(latent):
            decode = (self.decode_first_stage(latent) + 1) / 2
            return decode
        
        log['lr'] = _decode(lr_latent)
        log['sr'] = _decode(sr_latent)
        log['hr'] = _decode(hr_latent)

        return log
        
         
