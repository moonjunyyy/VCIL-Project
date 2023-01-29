from typing import TypeVar
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
                 pool_size      : int   = 10,
                 selection_size : int   = 1,
                 prompt_len     : int   = 5,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 lambd          : float = 0.5,
                 _batchwise_selection  : bool = False,
                 _diversed_selection   : bool = True,
                 **kwargs):

        super().__init__()
        
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        if pool_size < selection_size:
            raise ValueError('pool_size must be larger than selection_size')

        self.prompt_len     = prompt_len
        self.selection_size = selection_size
        self.lambd          = lambd
        self._batchwise_selection = _batchwise_selection
        self.class_num            = class_num

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
        
        self.key     = nn.Parameter(torch.randn(pool_size, self.backbone.embed_dim))
        self.prompts = nn.Parameter(torch.randn(pool_size, self.prompt_len, self.backbone.embed_dim))
        self.mask    = nn.Parameter(torch.zeros(pool_size, self.class_num))

        self.register_buffer('simmilarity', torch.zeros(1), persistent=False)
        self.register_buffer('unsimmilarity', torch.zeros(1), persistent=False)
    
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
        prompts = self.prompts[topk]
        mask = self.mask[topk].squeeze()

        self.simmilarity = simmilarity.mean()
        self.unsimmilarity = (1 - simmilarity).mean()

        prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
        prompts = prompts + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
        
        x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
        x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x = x[:, 1:self.selection_size * self.prompt_len + 1].clone()
        x = x.mean(dim=1).squeeze()
        x = self.backbone.fc_norm(x)
        x = self.backbone.fc(x)
        x = x * F.softmax(mask, dim=1)
        return x
    
    def loss_fn(self, output, target):
        # B, C = output.size()
        return F.cross_entropy(output, target) + self.lambd * self.simmilarity

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
    
