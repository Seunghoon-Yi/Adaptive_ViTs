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

# import models.configs as configs

from abc import *


class GetDists(nn.Module):
    def __init__(self):
        super().__init__()

    def get_distances(self, h, w, has_cls=False):
        rows, cols = np.indices((w, w))
        token_idxs = np.array(list(zip(rows.flatten(), cols.flatten())))

        dists = torch.Tensor(np.expand_dims(token_idxs, 1) - np.expand_dims(token_idxs, 0))  # (n,n,2)
        #print("before pad : ", dists.shape)
        if has_cls:
            dists = F.pad(dists, (0, 0, 1, 0, 0, 0), mode='constant',value=0)
            dists = F.pad(dists, (0, 0, 0, 0, 1, 0), mode='constant',value=0)
        dists = dists.to(torch.float32)
        dists = dists.to('cuda')

        return dists


class GaussianKernel(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.head_size = head_size

        self.sigma_layer_1 = Linear(head_size, 64)
        self.gelu = F.gelu
        self.sigma_layer_2 = Linear(64, 1)
        self.relu = F.relu

        self.dist_cls = GetDists()

    def forward(self, query, h, w, temperature, with_cls_token):
        # for name, params in self.sigma_layer_1.named_parameters():
        #    print('GRAD', params.grad)
        #print(h, w)
        b, h, q, d = query.size()
        #print(b, h, q, d)

        sigmas = self.sigma_layer_1(query)
        sigmas = self.gelu(sigmas)
        sigmas = self.sigma_layer_2(sigmas)
        sigmas = self.relu(sigmas) + 1.75  # (b,h,q,1)

        dists = self.dist_cls.get_distances(h, w, with_cls_token)
        dists = torch.sum(dists ** 2, dim=-1)
        #print(dists.shape, dists)

        dists = repeat(dists, 'q k -> b h q k', b=b, h=h)
        norm = 2 * sigmas ** 2
        #print(dists.shape, norm.shape)
        kernel = torch.exp(-1 * dists / norm)

        probs = kernel / torch.amax(kernel, dim=[-1], keepdim=True)
        probs = torch.clamp(probs, max=1, min=0)
        # print(probs)
        sampler = RelaxedBernoulli(temperature, probs=probs)
        mask = sampler.rsample()

        return mask, sigmas