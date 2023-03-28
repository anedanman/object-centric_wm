import os
import pickle
import torch
import numpy as np 
import moviepy.editor as mpy

import matplotlib.pyplot as plt 
from typing import Iterable
from torch.nn import Module


def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
          output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):

        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


def compute_return(rewards, values, discounts, td_lam, last_value):

    next_values = torch.cat([values[1:], last_value.unsqueeze(0)],0)  
    targets = rewards + discounts * next_values * (1-td_lam)
    rets =[]
    last_rew = last_value

    for t in range(rewards.shape[0]-1, -1, -1):
        last_rew = targets[t] + discounts[t] * td_lam *(last_rew)
        rets.append(last_rew)

    returns = torch.flip(torch.stack(rets), [0])
    return returns


def spatial_broadcast(x, resolution):
    # x = x.reshape(-1, x.shape[-1], 1, 1)
    x = torch.unsqueeze(x, dim=-1)
    x = torch.unsqueeze(x, dim=-1)
    x = x.expand(*x.shape[:-2], *resolution)
    return x


def spatial_flatten(x):
    x = torch.swapaxes(x, -3, -1)
    return torch.flatten(x, -3, -2)


def build_grid(resolution):
    """
    :param resolution: tuple of 2 numbers
    :return: grid for positional embeddings built on input resolution
    """
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)
