# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Linear
from torch.nn.modules.utils import _pair
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from scipy import ndimage

from einops import rearrange, reduce, repeat, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

import models.configs as configs

from abc import *


class Kernel(ABC, nn.Module):
    def __init__(self, config, img_size):
        super(Kernel, self).__init__()
        
        patch_size = _pair(config.patches['size'])
        n_patches = (img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1])
        self.dists = self.get_pairwise_distances(n_patches, to_float=config['fp16'])

    def get_pairwise_distances(self, n_patches, to_float=False, to_cuda=True):
        """
        Input
        - n_patches:
        Ouput
        - dists: (n,n)
        """
        n_rows = n_cols = int(np.sqrt(n_patches))
        rows, cols = np.indices((n_rows, n_cols))
        token_idxs = np.array(list(zip(rows.flatten(), cols.flatten())))
        dists = torch.Tensor(np.expand_dims(token_idxs,1) - np.expand_dims(token_idxs,0)) # (n,n,2)
        dists = F.pad(dists, (0,0, 1,0, 0,0), mode='constant', value=0) # cls token row > the cls token sees every other token
        dists = F.pad(dists, (0,0, 0,0, 1,0), mode='constant', value=0) # cls token col > every token sees the cls token.
        if to_float:
            dists = dists.to(torch.float16)
        if to_cuda:
            dists = dists.to('cuda')
        return dists

    @abstractmethod
    def forward(self, query, temperature):
        """
        Input
        - query: (b,h,n,hd)
        Outputs
        - mask: (b,h,n,n)
        - sigmas: (b,h,n,1)
        """
        pass


class MultivariateGaussianKernel(Kernel):
    def __init__(self, config, img_size):
        super(MultivariateGaussianKernel, self).__init__(config, img_size)

        self.sigma_layer_1 = Linear(config.hidden_size//config.transformer['num_heads'], 64)
        self.gelu = F.gelu
        self.relu = F.relu
        self.sigma_layer_2 = Linear(64, 3)  # sigma_y, sigma_x, sigma_xy

        self.identity_mat = torch.tensor([[0,1],[1,0]])
        if config['fp16']:
            self.identity_mat = self.identity_mat.to(torch.float16)
        self.identity_mat = self.identity_mat.to('cuda')

    def forward(self, query, temperature):
        #for name, params in self.sigma_layer_1.named_parameters():
        #    print(name)
        #    print('GRAD', params.grad)

        b, h, q, d = query.size()

        sigmas = self.sigma_layer_1(query)
        sigmas = self.gelu(sigmas)
        sigmas = self.sigma_layer_2(sigmas)
        sigmas_yx  = torch.exp(sigmas[..., [0,1]]) + 1.0 
        sigmas_cov = torch.tanh(sigmas[..., [2]]) * 0.99  # 0.99 to prevent precision issue
        
        sigmas_yx_mat = torch.einsum('bhqy,bhqx->bhqyx', sigmas_yx, sigmas_yx) # [Sx, Sy]^T * [Sx, Sy] -> [[SxSx, SxSy],[SySx, SySy]]
        sigmas_cov_mat = repeat(self.identity_mat, 'y x -> b h q y x', b=b, h=h, q=q) * sigmas_cov[...,None]
        sigmas_cov_mat[..., [0,1], [0,1]] = 1 # make diagonal = 1
        sigmas_mat = sigmas_yx_mat * sigmas_cov_mat    # (b,h,q,2,2)
        #print(sigmas_mat[-1,-1,-1])
  
        sigmas_inv = torch.linalg.inv(sigmas_mat.to(torch.float32)) # Invert the last two dimensions
        sigmas_inv = repeat(sigmas_inv, 'b h q y x -> b h q k y x', k=q).to(torch.float16)

        dists = repeat(self.dists, 'q k dist -> b h q k dist 1', b=b, h=h)
        weights = -1 / 2 * torch.matmul(torch.matmul(dists.permute([0, 1, 2, 3, 5, 4]), sigmas_inv), dists)
        weights = rearrange(weights, 'b h q k 1 1 -> b h q k')
        kernel  = torch.exp(weights)

        probs   = kernel / torch.amax(kernel, dim=-1, keepdim=True)
        probs   = torch.clamp(probs, max = 1)                             # to make sure that probs are constrnained to 1. 
        sampler = RelaxedBernoulli(temperature, probs=probs)
        mask    = sampler.rsample()

        return mask, sigmas_mat


class GaussianKernel(Kernel):
    def __init__(self, config, img_size):
        super(GaussianKernel, self).__init__(config, img_size)

        self.sigma_layer_1 = Linear(config.hidden_size//config.transformer['num_heads'], 64)
        self.gelu = F.gelu
        self.sigma_layer_2 = Linear(64, 1)
        self.relu = F.relu

    def forward(self, query, temperature):
        #for name, params in self.sigma_layer_1.named_parameters():
        #    print('GRAD', params.grad)

        b, h, q, d = query.size()

        sigmas = self.sigma_layer_1(query)
        sigmas = self.gelu(sigmas)
        sigmas = self.sigma_layer_2(sigmas)
        sigmas = self.relu(sigmas) + 2 # (b,h,q,1)
        sigmas = torch.clamp(sigmas, min=2)
        
        dists = torch.sum(self.dists**2, dim=-1)
        dists = repeat(dists, 'q k -> b h q k', b=b, h=h)
        norm = 2*sigmas**2
        #print("dists : ", dists.shape, "norm : ", norm.shape)
        kernel = torch.exp(-1 * dists / norm)
        
        probs   = kernel / torch.amax(kernel, dim=[-1], keepdim=True)  
        probs   = torch.clamp(probs, max = 1, min = 0)  
        #print(probs)
        sampler = RelaxedBernoulli(temperature, probs=probs)
        mask = sampler.rsample()
        
        return mask, sigmas


class ParabolicKernel(Kernel):
    """
    k(u) = 3/4 * (1-x**2)
    support : |x| <= 1 (if |x|>1, weight becomes 0)

    ### sqrt 안씌우는 대신에 weights에 square 안함 ###
    """
    def __init__(self, config, img_size):
        super(ParabolicKernel, self).__init__(config, img_size)
        self.sigma_layer_1 = Linear(self.attention_head_size, 64)
        self.gelu = F.gelu
        self.sigma_layer_2 = Linear(64, 1)

    def forward(self, query):
        b, q, h, d = query.size()

        sigmas = self.sigma_layer_1(query)
        sigmas = self.gelu(sigmas)
        sigmas = self.sigma_layer_2(sigmas)
        sigmas = torch.relu(sigmas) + 1.0

        """
        dists = torch.sum(self.dists**2, dim=-1)
        dists = repeat(dists, 'y x -> b h y x', b=b, h=h)
        norm = 2*sigmas**2
        kernel = torch.exp(-1 * dists / norm)
        """

        dists = self.dists.repeat([query.size()[0], self.num_attention_heads, 1, 1]) # (b,h,n,n)
        dists = dists / (sigmas**2)
        dists = torch.where(abs(dists)<=1, dists, torch.tensor(1).to(torch.float16).to('cuda'))
        
        kernel = 0.75 * (1 - dists)
        probs = kernel / torch.amax(kernel, dim=[-1], keepdim=True)

        sampler = RelaxedBernoulli(self.temperature, probs=probs)
        mask = sampler.sample()

        return mask. sigmas


class UniformKernel(Kernel):
    """
    k(u) = 1/2
    support : |u| <= 1 (if |u|>1, weight becomes 0)
    """
    def __init__(self, config, img_size):
        super(UniformKernel, self).__init__(config, img_size, norm=2)
        self.sigma_layer = Linear(self.attention_head_size, 1)
        self.temperature = torch.tensor([config.temperature]).to(torch.float16).to('cuda')

    def forward(self, query):
        sigmas = self.sigma_layer(query) # (b,h,n,hd) -> (b,h,n,1)

        dists = self.dists.repeat([query.size()[0], self.num_attention_heads, 1, 1]) # (b,h,n,n)
        dists = torch.sqrt(dists) / (sigmas**2)
        dists = torch.where(abs(dists)<=1, torch.tensor(0.5).to(torch.float16).to('cuda'), torch.tensor(0).to(torch.float16).to('cuda'))
        
        probs = dists / torch.amax(dists, dim=[-1], keepdim=True)

        sampler = RelaxedBernoulli(self.temperature, probs=probs)
        mask = sampler.sample()

        return mask, sigmas


class TriangularKernel(Kernel):
    """
    k(u) = (1-|u|)
    support : |u| <= 1 (if |u|>1, weight becomes 0)
    """
    def __init__(self, config, img_size):
        super(TriangularKernel, self).__init__(config, img_size, norm=2)
        self.sigma_layer = Linear(self.attention_head_size, 1)
        self.temperature = torch.tensor([config.temperature]).to(torch.float16).to('cuda')

    def forward(self, query):
        sigmas = self.sigma_layer(query) # (b,h,n,hd) -> (b,h,n,1)

        dists = self.dists.repeat([query.size()[0], self.num_attention_heads, 1, 1]) # (b,h,n,n)
        dists = torch.sqrt(dists) / (sigmas**2)
        dists = torch.where(abs(dists)<=1, dists, torch.tensor(1).to(torch.float16).to('cuda'))
        
        kernel = 1 - abs(dists)
        probs = kernel / torch.amax(kernel, dim=[-1], keepdim=True)
        probs = torch.nan_to_num(probs, nan=0.0)

        sampler = RelaxedBernoulli(self.temperature, probs=probs)
        mask = sampler.sample() > 0.5

        return mask, sigmas


class QuarticKernel(Kernel):
    """
    k(u) = 15/16 * (1-u**2)**2
    support : |u| <= 1 (if |u|>1, weight becomes 0)
    """
    def __init__(self, config, img_size):
        super(QuarticKernel, self).__init__(config, img_size, norm=2)
        self.sigma_layer = Linear(self.attention_head_size, 1)
        #nn.init.constant_(self.sigma_layer.weight, 0.5)
        self.temperature = torch.tensor([config.temperature]).to(torch.float16).to('cuda')

    def forward(self, query):
        sigmas = self.sigma_layer(query) # (b,h,n,hd) -> (b,h,n,1)

        dists = self.dists.repeat([query.size()[0], self.num_attention_heads, 1, 1]) # (b,h,n,n)
        dists = dists / (sigmas**2)
        dists = torch.where(abs(dists)<=1, dists, torch.tensor(1).to(torch.float16).to('cuda'))
        
        kernel = 15/16 * ((1 - dists)**2)
        probs = kernel / torch.amax(kernel, dim=[-1], keepdim=True)
        probs = torch.nan_to_num(probs, nan=0.0)

        sampler = RelaxedBernoulli(self.temperature, probs=probs)
        mask = sampler.sample() > 0.5

        return mask, sigmas