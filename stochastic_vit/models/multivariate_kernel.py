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

import models.configs as configs

from abc import *


class Kernel(ABC, nn.Module):
    def __init__(self, config, img_size, norm):
        super(Kernel, self).__init__()
        self.num_attention_heads = config.transformer['num_heads']
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        patch_size = _pair(config.patches['size'])
        n_patches = (img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1])
        self.dists = self.get_pairwise_distances(n_patches, norm)

    def get_pairwise_distances(self, n_patches, norm, absolute=True, to_cuda=True):
        """
        Input
        - n_patches:
        Ouput
        - dists: (n,n)
        """
        n_rows = n_cols = int(np.sqrt(n_patches))
        rows, cols = np.indices((n_rows, n_cols))
        token_idxs = np.array(list(zip(rows.flatten(), cols.flatten())))
        dists = torch.Tensor(np.expand_dims(token_idxs,1) - np.expand_dims(token_idxs,0))# (n,n,2)
        dists = F.pad(dists, (0,0, 1,0, 1,0), mode='constant', value=0) # for cls token, dist alayws set to 0

        if to_cuda:
            dists = dists.to('cuda')
        return dists

    @abstractmethod
    def forward(self, query):
        """
        Input
        - query: (b,h,n,hd)
        Outputs
        - mask: (b,h,n,n)
        - sigmas: (b,h,n,1)
        """
        pass


class GaussianKernel(Kernel):
    def __init__(self, config, img_size):
        super(GaussianKernel, self).__init__(config, img_size, norm=2)
        self.sigma_layer_1 = Linear(self.attention_head_size, 64)
        self.gelu = F.gelu
        self.relu = F.relu
        self.sigma_layer_2 = Linear(64, 3)  # sigma_x, sigma_y, sigma_xy
        # if config.bias_init:
        #    nn.init.constant_(self.sigma_layer_1.bias, 1)
        # nn.init.constant_(self.sigma_layer_1.weight, 0)

        self.temperature = torch.tensor([config.temperature]).to(torch.float16).to('cuda')

    def forward(self, query):
        #for name, params in self.sigma_layer_1.named_parameters():
        #    print(name)
        #    print('GRAD', params.grad)
        batch_size = query.size()[0]
        n_patches = query.size()[1]

        sigmas = self.sigma_layer_1(query)
        sigmas = self.gelu(sigmas)  # self.gelu(sigmas)
        sigmas = self.sigma_layer_2(sigmas)
        #print("var shape : ", sigmas.shape)
        #print("distances shape : ", self.dists.shape)
        # sigmas = torch.exp(sigmas) + 1  # (b,h,n,3) sigma_yy, sigma_xx, sigma_xy

        repeats = [*sigmas.shape[:-1], 1, 1]  # (b,h,n,1,1)

        #print("exped : ", torch.exp(sigmas[..., [0,1]]).shape)

        sigma_x_and_y = self.relu(sigmas[..., [0,1]]) + 1.0                                                          #(b,h,n,2)
        covariances   = 0.99*torch.tensor([[0,1],[1,0]]).repeat(repeats).to('cuda') * torch.tanh(sigmas[...,2,None,None])    #(b,h,n,2,2)
        sig_0_matrix  = torch.einsum('bhnd,bhnc->bhndc', sigma_x_and_y, sigma_x_and_y)                                  #(b,h,n,2,2)
        # [Sx, Sy]^T * [Sx, Sy] -> [[SxSx, SxSy],[SySx, SySy]]
        #print(sigma_x_and_y.shape, covariances.shape, sig_0_matrix.shape)
        covariances[..., [0, 1], [0, 1]] = 1                                             # Make diagonal = 1
        sigma_matrix   = sig_0_matrix * covariances
        sigmas_inverse = torch.linalg.inv(sigma_matrix.to(torch.float32))                # Invert the last two dimensions
        sigmas_inverse = sigmas_inverse.unsqueeze(3).repeat([1, 1, 1, n_patches, 1, 1])  # (b,h,n,n,2,2)

        dists = self.dists.repeat([batch_size, self.num_attention_heads, 1, 1, 1])
        dists = torch.unsqueeze(dists, -1)                                               # (b,h,n,n,2,1)
        weights = -1 / 2 * torch.matmul(torch.matmul(dists.permute([0, 1, 2, 3, 5, 4]), sigmas_inverse), dists)
        weights = weights.view(weights.shape[:-2]).to(torch.float16) 
        kernel  = torch.exp(weights)                                                     # (b,h,n,n)

        probs   = kernel / torch.amax(kernel, dim=-1, keepdim=True)
        #print(probs)
        sampler = RelaxedBernoulli(self.temperature, probs=probs)
        mask    = sampler.rsample()

        return mask, sigma_matrix