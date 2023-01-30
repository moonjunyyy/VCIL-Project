from typing import TypeVar, Iterable
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs

from models.vit import _create_vision_transformer

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

T = TypeVar('T', bound = 'nn.Module')

default_cfgs['vit_base_patch16_224_l2p'] = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        num_classes=21843)

# Register the backbone model to timm
@register_model
def vit_base_patch16_224_l2p(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_l2p', pretrained=pretrained, **model_kwargs)
    return model

class Ours(nn.Module):
    def __init__(self,
                 pos_g_prompt   : Iterable[int] = (0, 1),
                 len_g_prompt   : int   = 5,
                 pos_e_prompt   : Iterable[int] = (2, 3, 4),
                 len_e_prompt   : int   = 20,
                 prompt_func    : str   = None,
                 task_num       : int   = 10,
                 class_num      : int   = 100,
                 lambd          : float = 1.0,
                 backbone_name  : str   = None,
                 **kwargs):

        super().__init__()
        
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        self.lambd       = lambd
        self.class_num   = class_num

        # model_kwargs = dict(
        # patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
        
        # self.add_module('backbone', timm.models.create_model(backbone_name, pretrained=True, num_classes=class_num))
        self.add_module('backbone', timm.models.create_model(backbone_name, pretrained=True, num_classes=class_num,
                                                             drop_rate=0.,drop_path_rate=0.,drop_block_rate=None))
        for name, param in self.backbone.named_parameters():
                param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        # self.fc = self.backbone.fc
        
        # self.key     = nn.Parameter(torch.randn(pool_size, self.backbone.embed_dim))
        # self.prompts = nn.Parameter(torch.randn(pool_size, self.prompt_len, self.backbone.embed_dim))
        # self.mask    = nn.Parameter((torch.zeros(pool_size, self.class_num)))

        self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype=torch.int64))
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
        self.register_buffer('similarity', torch.zeros(1))
        self.register_buffer('mask', torch.zeros(class_num))
        
        self.len_g_prompt = len_g_prompt
        self.len_e_prompt = len_e_prompt
        self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0
        g_pool = 1
        e_pool = task_num

        self.key     = nn.Parameter(torch.randn(e_pool, self.backbone.embed_dim))
        self.mask    = nn.Parameter((torch.zeros(e_pool, self.class_num)))

        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_prompts = nn.Parameter(torch.randn(g_pool, 1 * self.g_length * self.len_g_prompt, self.backbone.embed_dim))
            self.e_prompts = nn.Parameter(torch.randn(e_pool, 1 * self.e_length * self.len_e_prompt, self.backbone.embed_dim))

        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.g_prompts = nn.Parameter(torch.randn(g_pool, 2 * self.g_length * self.len_g_prompt, self.backbone.embed_dim))
            self.e_prompts = nn.Parameter(torch.randn(e_pool, 2 * self.e_length * self.len_e_prompt, self.backbone.embed_dim))
        # self.register_buffer('mask', torch.zeros(pool_size, self.class_num, requires_grad=True))

        self.register_buffer('simmilarity', torch.zeros(1), persistent=False)
        # self.register_buffer('unsimmilarity', torch.zeros(1), persistent=False)
    
    
    def prompt_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, self.e_length, self.len_e_prompt, C)
        g_prompt = g_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.e_length, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                x = torch.cat((x, g_prompt[:, pos_g].unsqueeze(0).expand(B,-1,-1)), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e].unsqueeze(0).expand(B,-1,-1)), dim = 1)
            x = block(x)
        return x
    
    def prefix_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, 2 * self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, 2 * self.e_length, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):

            xq = block.norm1(x)
            xk = xq.clone()
            xv = xq.clone()

            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                xk = torch.cat((xk, g_prompt[:, pos_g * 2 + 0]), dim = 1)
                xv = torch.cat((xv, g_prompt[:, pos_g * 2 + 1]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                xk = torch.cat((xk, e_prompt[:, pos_e * 2 + 0]), dim = 1)
                xv = torch.cat((xv, e_prompt[:, pos_e * 2 + 1]), dim = 1)
            
            attn   = block.attn
            weight = attn.qkv.weight
            bias   = attn.qkv.bias
            
            B, N, C = xq.shape
            xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attention = (xq @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)

            attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
            attention = attn.proj(attention)
            attention = attn.proj_drop(attention)

            x = x + block.drop_path1(block.ls1(attention))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        return x

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:

        x = self.backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        with torch.no_grad():
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0].clone()

        simmilarity = F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        topk = simmilarity.topk(self.selection_size, dim=1)[1]
        simmilarity = simmilarity.gather(1, topk)
        e_prompts = self.e_prompts[topk]
        g_prompts = self.g_prompts[0]
        mask = self.mask[topk].squeeze()
        self.simmilarity = simmilarity.mean()

        x = self.prompt_func(x, g_prompts, e_prompts)
        x = self.backbone.norm(x)
        x = x[:, 1:self.selection_size * self.prompt_len + 1].clone()
        x = x.mean(dim=1).squeeze()
        x = self.backbone.fc_norm(x)
        x = self.backbone.fc(x)

        # if self.training:
        # mask =  F.softmax(mask * 1, dim=1)
        mask = torch.sigmoid(mask)
        # mask = mask / mask.max(dim=1, keepdim=True)[0]
        # x_min = x.min(dim=1, keepdim=True)[0]
        # x_max = x.max(dim=1, keepdim=True)[0]i/
        # x = (x - x_min) * mask + x_min
        # if self.training:
        x = x * mask
        return x
    
    def loss_fn(self, output, target):
        # B, C = output.size()
        return F.cross_entropy(output, target) - self.lambd * self.simmilarity

    def convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task] = 0
        return

    def get_count(self):
        return self.prompt.update()

    def train(self: T, mode : bool = True, **kwargs):
        ten = super().train()
        self.backbone.eval()
        return ten
    
    def eval(self: T, mode : bool = True, **kwargs):
        ten = super().eval()
        self.backbone.eval()
        return ten
    
