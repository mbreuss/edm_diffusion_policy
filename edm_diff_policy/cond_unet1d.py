from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from .conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from .utils import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]
            goal: [batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):

    def __init__(self, 
        input_dim: int,
        act_seq_len: int,
        goal_seq_len: int,
        obs_seq_len: int,
        obs_dim: int,
        goal_dim: int,
        device: str,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        goal_conditioned=True,
        goal_drop=0,
        ):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.cond_mask_prob = goal_drop
        self.goal_conditioned = goal_conditioned
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        ).to(self.device)
        cond_dim = dsed + self.obs_seq_len *  2* obs_dim + self.goal_seq_len * goal_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        '''states_encoder = None
        if states_dim is not None:
            _, dim_out = in_out[0]
            dim_in = input_dim * act_seq_len
            states_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])'''

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]).to(self.device))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        ).to(self.device)

        self.diffusion_step_encoder = diffusion_step_encoder
        # self.states_encoder = states_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            states,
            action, 
            goals,
            timestep,
            global_cond=None, 
            uncond=False,
            **kwargs
        ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        states: (B,T,states_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # action = einops.rearrange(action, 'b h t -> b t h')

        if len(goals.shape) == 2:
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        if goals.shape[1] == states.shape[1] and self.goal_seq_len == 1:
            goals = goals[:, 0, :]
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        # 1. time
        timesteps = timestep.log() / 4
        timesteps = einops.rearrange(timesteps, 'b -> b 1')
        emb_t = self.diffusion_step_encoder(timesteps)
        if len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        

        if self.training:
            goals = self.mask_cond(goals)
        # we want to use unconditional sampling during clasisfier free guidance
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)  

        
        global_feature = torch.cat([states, goals, emb_t], dim=-1)
        global_feature = einops.rearrange(global_feature, 'b h t -> b t h ')
        global_feature = einops.rearrange(global_feature, 'b t d -> b (t d)')
        # encode local features
        h_local = list()
        
        x = action
        h = []
        # next we need to change the order
        x = einops.rearrange(x, 'b h t -> b t h ')
        for idx, (resnet, resnet2, downaction) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downaction(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upaction) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upaction(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x

    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            # TODO Check which one is correct
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob) # .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            # mask = torch.bernoulli(torch.ones((bs, t, 1), device=cond.device) * self.cond_mask_prob)
            # mask = einops.repeat(mask, 'b t 1 -> b t (1 d)', d=d)
            return cond * (1. - mask)
        else:
            return cond


class DiffusionUnet1D(nn.Module):

    def __init__(self, 
        input_dim: int,
        obs_dim: int,
        # goal_dim: int,
        device: str,
        diffusion_step_embed_dim=256,
        down_dims=[512, 1024],
        kernel_size=1,
        n_groups=8,
        cond_predict_scale=False,
        goal_conditioned=True,
        goal_drop=0,
        ):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.cond_mask_prob = goal_drop
        self.goal_conditioned = goal_conditioned
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        ).to(self.device)
        cond_dim = dsed + obs_dim  #+ goal_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        '''states_encoder = None
        if states_dim is not None:
            _, dim_out = in_out[0]
            dim_in = input_dim * act_seq_len
            states_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])'''

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]).to(self.device))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        ).to(self.device)

        self.diffusion_step_encoder = diffusion_step_encoder
        # self.states_encoder = states_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
            self, 
            global_cond,
            action, 
            timesteps,
            **kwargs
        ):
        """
        x: (B, input_dim)
        timestep: (B,) or int, diffusion step
        states: (B,states_dim)
        global_cond: (B,global_cond_dim)
        output: (B, input_dim)
        """
        # action = einops.rearrange(action, 'b h t -> b t h')
        if len(global_cond.shape) == 2:
            global_cond = einops.rearrange(global_cond, 'b d -> b 1 d')
        timesteps = einops.rearrange(timesteps, 'b -> b 1')
        emb_t = self.diffusion_step_encoder(timesteps)
        if len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        
        global_feature = torch.cat([global_cond, emb_t], dim=-1)
        global_feature = einops.rearrange(global_feature, 'b h t -> b t h ')
        global_feature = einops.rearrange(global_feature, 'b t d -> b (t d)')
        # encode local features
        h_local = list()
        
        x = action
        h = []
        # next we need to change the order
        if len(x.shape) == 2:
            x = einops.rearrange(x, 'b d -> b 1 d')
        x = einops.rearrange(x, 'b h t -> b t h ')
        for idx, (resnet, resnet2, downaction) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            print("Down: Shape of x:", x.shape)
            h.append(x)
            x = downaction(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        print("After middle module: Shape of x:", x.shape)
        for idx, (resnet, resnet2, upaction) in enumerate(self.up_modules):
            print("Up: Shape of x:", x.shape)
            print("Up: Shape of h[-1]:", h[-1].shape)  # Assuming h[-1] is the tensor to be popped
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == (len(self.up_modules)-1) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upaction(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        x = einops.rearrange(x, 'b 1 t -> b t')
        return x

    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            # TODO Check which one is correct
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob) # .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            # mask = torch.bernoulli(torch.ones((bs, t, 1), device=cond.device) * self.cond_mask_prob)
            # mask = einops.repeat(mask, 'b t 1 -> b t (1 d)', d=d)
            return cond * (1. - mask)
        else:
            return cond