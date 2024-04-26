import logging
from functools import partial
from typing import Optional, Tuple
from collections import deque

import torch
import hydra
import torch.nn as nn
from omegaconf import DictConfig
import einops

from .gc_sampling import *
from .utils import *

logger = logging.getLogger(__name__)


class DiffusionDecoder(nn.Module):
    
    def __init__(
        self,
        out_features: int,
        criterion: str,
        model: nn.Module,
        sampler_type: str,
        rho: float,
        num_sampling_steps: int,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        sigma_sample_density_mean: float,
        sigma_sample_density_std: float,
        action_window_size: int,
        obs_window_size: int, 
        device: str,
        sigma_sample_density_type: str = 'loglogistic',
        noise_scheduler: str = 'exponential',
    ):
        super(DiffusionDecoder, self).__init__()
        self.out_features = out_features
        self.criterion = getattr(nn, criterion)()
        self.device = device
        self.act_window_size = action_window_size
        self.obs_window_size = obs_window_size
        self.window_size = self.act_window_size + self.obs_window_size  -1 
        self.model = model
        self.rho = rho
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_mean = sigma_sample_density_mean
        self.sigma_sample_density_std = sigma_sample_density_std
        self.sigma_sample_density_type = sigma_sample_density_type
        self.obs_context = deque(maxlen=self.obs_window_size)

    
    def forward(  # type: ignore
        self,
        perceptual_emb, # dict or torch.Tensor
        latent_goal, # dict or torch.Tensor
        inference: Optional[bool] = False,
        extra_args={}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        if inference:
            sampling_steps = self.num_sampling_steps
        else:
            sampling_steps = 10
        self.model.eval()
        # batch_size, seq_len = perceptual_emb.shape[0], perceptual_emb.shape[1]
        
        if self.obs_window_size > 1:
            # rearrange from 2d -> sequence
            if isinstance(perceptual_emb, dict):
                pass # !ToDo! FixMe
            else:
                self.obs_context.append(perceptual_emb) # this automatically manages the number of allowed observations
            input_state = torch.concat(tuple(self.obs_context), dim=1)
        else:
            input_state = perceptual_emb

        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)
        latent_goal_length = 0
        if isinstance(latent_goal, dict):
            for latent_goal_key, latent_goal_value in latent_goal.items():
                if len(latent_goal_value.shape) == 2:
                    latent_goal[latent_goal_key] = einops.rearrange(latent_goal_value, 'b d -> 1 b d')
                latent_goal_length += len(latent_goal_value)
        else:
            if len(latent_goal.shape) == 2:
                goal = einops.rearrange(goal, 'b d -> 1 b d')
            latent_goal_length = len(latent_goal)

        
        x = torch.randn((latent_goal_length, self.act_window_size, self.out_features), device=self.device) * self.sigma_max
        actions = self.sample_loop(sigmas, x, input_state, latent_goal, self.sampler_type, extra_args)

        return actions, None

    def evaluate(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        extra_args={}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss and predicted actions given the erceptual embedding, latent goal, and actions.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the predicted loss tensor and predicted actions tensor.
        """
        if isinstance(latent_goal, dict):
            batch_size = next(iter(latent_goal.values())).shape[0]
        else:
            batch_size = latent_goal.shape[0] #, perceptual_emb.shape[1]
        sampling_steps = self.num_sampling_steps
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)
        x = torch.randn((batch_size, self.act_window_size, self.out_features), device=self.device) * self.sigma_max
        pred_actions = self.sample_loop(sigmas, x, perceptual_emb, latent_goal, self.sampler_type, extra_args)
        pred_loss = self.criterion(pred_actions, actions)
        return pred_loss

    def loss(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the loss given the latent plan, perceptual embedding, latent goal, and actions.
        """
        self.model.train()
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        loss, _ = self.model.loss(perceptual_emb, actions, latent_goal, noise, sigmas)
        return loss
    
    def make_sample_density(self):
        """ 
        Generate a sample density function based on the desired type for training the model
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(rand_log_normal, loc=loc, scale=scale)
        
        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_uniform, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'uniform':
            return partial(rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)
        
        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.n_sampling_steps, 'exponential')
            return partial(rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')
    
    def sample_loop(
        self, 
        sigmas, 
        x_t: torch.Tensor,
        state: torch.Tensor, 
        goal: torch.Tensor, 
        sampler_type: str,
        extra_args={}, 
        ):
        """
        Main method to generate samples depending on the chosen sampler type
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x:extra_args[x] for x in keys}
        else:
            reduced_args = {}
        
        if use_scaler:
            scaler = self.scaler
        else:
            scaler=None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min, disable=True)
        # ODE deterministic 
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True) 
        # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas), disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'debugging':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            # x_0 = sample_euler_visualization(self.model, state, x_t, goal, sigmas, self.scaler, self.working_dir, disable=True, extra_args={'keep_last_actions': True})
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0    
    
    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')
    
    def get_inner_model(self):
        return self.model.get_inner_model()

    def set_inner_model(self, model):
        self.model.set_inner_model(model)
