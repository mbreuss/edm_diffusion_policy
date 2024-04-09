import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
# code adapted from 
# https://github.com/GuyTevet/motion-diffusion-model/blob/55cd84e14ba8e0e765700913023259b244f5e4ed/model/cfg_sampler.py
class ClassifierFreeSampleModel(nn.Module):
    """
    A wrapper model that adds conditional sampling capabilities to an existing model.

    Args:
        model (nn.Module): The underlying model to run.
        cond_lambda (float): Optional. The conditional lambda value. Defaults to 2.

    Attributes:
        model (nn.Module): The underlying model.
        cond_lambda (float): The conditional lambda value.
        cond (bool): Indicates whether conditional sampling is enabled based on the cond_lambda value.
    """
    def __init__(self, model, cond_lambda: float=2):
        super().__init__()
        self.model = model  # model is the actual model to run
        # pointers to inner model
        self.cond_lambda = cond_lambda
        if self.cond_lambda == 1:
            self.cond = True
        else:
            self.cond = False

    def forward(self, state, action, goal, sigma, **extra_args):
        if self.cond:
            return self.model(state, action, goal, sigma)
        elif self.cond_lambda == 0:
            uncond_dict = {'uncond': True}
            out_uncond = self.model(state, action, goal, sigma, **uncond_dict)
            return out_uncond
        else:
            action = deepcopy(action)
            
            out = self.model(state, action, goal, sigma, **extra_args)
            uncond_dict = {'uncond': True}
            out_uncond = self.model(state, action, goal, sigma, **uncond_dict)
            
            return out_uncond + self.cond_lambda * (out - out_uncond)
    
    def get_params(self):
        return self.model.get_params()
    

class CompositionalBeso(nn.Module):

    def __init__(self, model_1, model_2, cond_lambda: float=0.75) -> None:
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.cond_lambda = cond_lambda 

    def forward(self, state, action, goal, sigma, **extra_args):

        action = deepcopy(action)
            
        pred_1 = self.model_1(state, action, goal, sigma, **extra_args)

        pred_2 = self.model_2(state, action, goal, sigma, **extra_args)

        return pred_1 + self.cond_lambda * (pred_2 - pred_1)