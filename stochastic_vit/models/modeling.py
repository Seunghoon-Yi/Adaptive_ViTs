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

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2
from .kernels import *

from functools import partial # added for Kernel

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)    
        self.value = Linear(config.hidden_size, self.all_head_size)  

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        self.config  = config

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, temperature):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)        # (b,n,d)
        value_layer = self.transpose_for_scores(mixed_value_layer)    # (b,n,d)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # (b,n,d) * (b,d,n) > (b,n,n)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        if self.config.kernel == 'gaussian_multivariate':
            #print("attn probs shape : ", attention_probs.shape)
            sigmas = torch.zeros_like(attention_probs[..., -1]).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 2,2)
            #print("sigmas shape : ", sigmas.shape)
        else:
            sigmas = torch.zeros_like(attention_probs[..., -1]).unsqueeze(-1)
            #print("sigmas shape : ", sigmas.shape)
        return attention_output, weights, sigmas


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size)) # (1, n_patches+1, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class StochasticAttention(nn.Module):
    """
    """
    
    def __init__(self, config, img_size, vis, mask_mode='bernoulli', kernel='gaussian', **kwargs):
        """
        """
        super(StochasticAttention, self).__init__()
                
        self.vis = vis
        self.num_attention_heads = config.transformer['num_heads']
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer['attention_dropout_rate'])
        self.proj_dropout = Dropout(config.transformer['attention_dropout_rate'])
        
        self.softmax = Softmax(dim=-1)

        img_size = _pair(img_size) 
        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.n_patches = n_patches

        if config.kernel == 'gaussian':
            self.kernel = GaussianKernel(config, img_size)
        if config.kernel == 'parabolic':
            self.kernel = ParabolicKernel(config, img_size)
        if config.kernel == 'triangular':
            self.kernel = TriangularKernel(config, img_size)
        if config.kernel == 'uniform':
            self.kernel = UniformKernel(config, img_size)
        if config.kernel == 'quartic':
            self.kernel = QuarticKernel(config, img_size)
        if config.kernel == 'gaussian_multivariate':
            self.kernel = MultivariateGaussianKernel(config, img_size)

    def forward(self, hidden_states, temperature):
        """
        """
        mixed_query_layer = self.query(hidden_states)              # (b,n,d)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer) # (b,n,d) -> (b,n,h,head_dim)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) 
        #print("Q shape : ", query_layer.shape)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # O(n^2)
        #print("Attn shape : ", attention_scores.shape)
        #print("Mask shape : ", self.mask.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # do I need this for local attention?

        mask, sigmas = self.kernel(query_layer, temperature)
        sparse_scores = mask * attention_scores

        attention_probs = self.softmax(sparse_scores) #torch.sparse.softmax(sparse_scores, dim=3)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs) # sparse tensor version? (b,c,n,n)
        
        context_layer = torch.matmul(attention_probs, value_layer) # torch.sparse.mm(attention_probs, value_layer) 
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights, sigmas


    def transpose_for_scores(self, x, permute=True):
        """
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        if permute:
            return x.permute(0,2,1,3)
        else:
            return x

        

class Block(nn.Module):
    def __init__(self, config, vis, img_size, idx):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)

        if idx not in config.transformer["vanila"]:
            self.attn = StochasticAttention(config, img_size, vis) # use config later
        else:
            self.attn = Attention(config, vis)

    def forward(self, x, temperature):
        h = x
        x = self.attention_norm(x)
        x, weights, sigmas = self.attn(x, temperature)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights, sigmas


class Encoder(nn.Module):
    def __init__(self, config, vis,img_size):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for idx in range(config.transformer["num_layers"]):
            layer = Block(config, vis, img_size, idx)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, temperature):
        attn_weights = []
        sigmas_list = []
        for layer_block in self.layer:
            hidden_states, weights, sigmas = layer_block(hidden_states, temperature)
            if self.vis:
                attn_weights.append(weights.detach().cpu().numpy())
                sigmas_list.append(sigmas.detach().cpu().numpy())
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights, sigmas_list


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, img_size=img_size)

    def forward(self, input_ids, temperature):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights, sigmas_list = self.encoder(embedding_output, temperature)
        return encoded, attn_weights, sigmas_list


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=True):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None, temperature=1.0):
        x, attn_weights, sigmas_list = self.transformer(x, temperature)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights, sigmas_list



CONFIGS = {
    'ViT-M_16': configs.get_m16_config(),
    'ViT-S_16': configs.get_s16_config(),
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
