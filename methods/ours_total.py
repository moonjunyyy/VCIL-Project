from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from torchvision import transforms
from randaugment.randaugment import RandAugment

from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset
from utils.augment import Cutout, Invert, Solarize, select_autoaugment

import logging
import copy
import time
import datetime

import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from methods._trainer import _Trainer

from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler

import timm
from timm.models import create_model
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs
from models.vit import _create_vision_transformer


logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

T = TypeVar('T', bound = 'nn.Module')

default_cfgs['vit_base_patch16_224'] = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        num_classes=21843)

# Register the backbone model to timm
@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

class Ours_total(_Trainer):
    def __init__(self, *args, **kwargs):
        super(Ours_total, self).__init__(*args, **kwargs)
        self.use_mask    = kwargs.get("use_mask")
        self.use_dyna_exp    = kwargs.get("use_dyna_exp")
        self.use_contrastiv  = kwargs.get("use_contrastiv")
        self.use_last_layer  = kwargs.get("use_last_layer")
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999
        
        self.class_mask = None
        self.class_mask_dict={}
        self.sample_criterion = nn.CrossEntropyLoss(reduction='none')
    
    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        del(images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter
    
    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        
        ref_fc = copy.deepcopy(self.model.backbone.fc)
        ref_fc.eval()
        x, y = data
        # for j in range(len(y)):
        #     y[j] = self.exposed_classes.index(y[j].item())

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)
        
        self.optimizer.zero_grad()
        logit, loss = self.model_forward(x,y,ref_fc=ref_fc)
        _, preds = logit.topk(self.topk, 1, True, True)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y,ref_fc):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # logit,feat = self.model(x)
            feat,mask,mass,similarity,topk = self.model.forward_features(x)
            ign_score,sample_g,batch_g,total_batch_g = self.get_score(ref_head=ref_fc,feat=feat,y=y,mask=mask)
            # _get_loss(self,str_score,feat,y,total_batch_g)
            loss,logit = self._get_loss(ign_score,feat,y,total_batch_g,mask,mass,similarity,topk)
        return logit, loss

    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit,_ = self.model(x)
                logit = logit + self.mask
                loss = self.sample_criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.mean().item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()
            
    def online_before_task(self,train_loader):
        # self.model_without_ddp.convert_train_task(self.exposed_classes)
        pass

    def online_after_task(self, cur_iter):
        pass

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, True)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
        
    
    
    def _compute_grads_uncert(self,ref_head,feat,y,mask):
        sample_criterion = nn.CrossEntropyLoss(reduction='none')
        batch_criterion = nn.CrossEntropyLoss()
        sample_g = []
        ref_head.zero_grad()
        tmp_logit = ref_head(feat)
        if self.use_mask:
            tmp_logit = tmp_logit*mask

        p = torch.log_softmax(tmp_logit[:,:len(self.exposed_classes)],dim=1)
        idx = torch.arange(len(y))
        uncert = 1. - p[idx,y[idx]].clone().detach() #B
        
        sample_loss = sample_criterion(tmp_logit,y)
        
        for idx in range(len(y)):
            sample_loss[idx].backward(retain_graph=True)
            _g = ref_head.weight.grad[y[idx]].clone()
            sample_g.append(_g)
            ref_head.zero_grad()
        sample_g = torch.stack(sample_g)    #B,dim
        
        ref_head.zero_grad()
        batch_loss = batch_criterion(tmp_logit,y)
        batch_loss.backward(retain_graph=True)
        total_batch_g = ref_head.weight.grad.clone()  # C,dim
        idx = torch.arange(len(y))
        batch_g=total_batch_g[y[idx]]    #B,dim
        ref_head.zero_grad()
        
        return uncert, sample_g, batch_g, total_batch_g
    
    def min_max(self,x):
        min = x.min()
        max = x.max()
        denom = max-min
        minmax= []
        for i in range(len(x)):
            minmax.append( (x[i] -min) / denom)
        minmax = torch.tensor(minmax,device=self.device)+1.
        if True in torch.isnan(minmax):
            minmax=None
        return minmax
    
    def _get_strength(self,ref_head,feat,y,mask):
        uncert, sample_g, batch_g,total_batch_g = self._compute_grads_uncert(ref_head,feat,y,mask)
        str_score = torch.max(1. - torch.cosine_similarity(sample_g,batch_g,dim=1),torch.zeros(1,device=self.device)) #B
        str_score = self.min_max(str_score)
        return str_score,sample_g,batch_g,total_batch_g
    

    def _get_drift(self,y,sample_g):
        idx = torch.arange(len(y))
        pre_wts = self.model.backbone.fc.weight[y[idx]].clone().detach()
        post_wts = pre_wts-sample_g
        drift = 1 - torch.cosine_similarity(pre_wts,post_wts,dim=1) # B
        return drift

    def get_score(self,ref_head,feat,y,mask):
        ign_feat = self.model.backbone.fc_norm(feat[:,0].clone().detach())
        
        ignore_score,sample_g,batch_g,total_batch_g = self._get_strength(ref_head,ign_feat,y,mask)
        
        return ignore_score,sample_g,batch_g,total_batch_g
    
    #* for ignore problem
    def str_loss(self,logit,y,str_score):
        log_p = F.log_softmax(logit,dim=1)
        ce = F.nll_loss(log_p,y,reduction='none')
        loss = (str_score**self.gamma)*ce
        
        return loss
        
    #* for drift problem
    def cp_loss(self,feat,y,total_batch_g,mask):
        # sample_b = self.model.backbone.fc.bias[y]
        peeking_w = self.model.backbone.fc.weight - self.lr*self.beta*total_batch_g    #* B,dim
        
        peeking_feat = self.model.backbone.fc_norm(feat.clone().detach()[:,0])
        
        #todo bias 까지 고려하는것도 해보기!
        peeking_logit = F.linear(peeking_feat,weight=peeking_w,bias=None)
        if self.use_mask:
            peeking_logit = peeking_logit*mask
        cp_loss = self.sample_criterion(peeking_logit+self.mask,y)
        
        return cp_loss

    def _get_loss(self,str_score,feat,y,total_batch_g,mask,mass,similarity,topk):
        #*#########################################################################
        #* CE_loss (masking / main)
        ce_logit = self.model.forward_head(feat,mask,mass,similarity,topk)
        ce_logit = ce_logit + self.mask
        ce_loss = self.criterion(ce_logit, y.to(torch.int64))
        #*#########################################################################
        #* strength_loss
        if str_score != None and self.alpha !=0.:
            ign_feat = self.model.backbone.fc_norm(feat[:,0])
            ign_logit = self.model.backbone.fc(ign_feat)
            if self.use_mask:
                ign_logit = ign_logit*mask
            str_loss = self.str_loss(ign_logit+self.mask, y, str_score)
            str_loss = str_loss.mean()
        elif str_score == None and self.alpha != 0.:   #* non ignore
            ign_feat = self.model.backbone.fc_norm(feat[:,0])
            ign_logit = self.model.backbone.fc(ign_feat) + self.mask
            str_loss = self.sample_criterion(ign_logit, y).mean()
        else:   #* for the baseline
            str_loss = torch.zeros(1,device=self.device)
        
        #*########################################################################
        #* compensation_loss
        if self.charlie != 0.:
            cp_loss = self.cp_loss(feat,y,total_batch_g,mask)
            cp_loss = cp_loss.mean()
        else:
            cp_loss = torch.zeros(1,device=self.device)
        #*########################################################################
        
        # if self.alpha == 0. and self.charlie == 0.: #* masking or baseline
        #     loss = ce_loss + self.alpha*str_loss + self.charlie * cp_loss 
        # elif self.alpha>0:
        #     loss =str_loss
        if self.alpha >0. or self.charlie>0.:
            loss = ce_loss + self.alpha*str_loss + self.charlie * cp_loss 
        else:
            loss = ce_loss
        return loss, ce_logit
    
    def setup_distributed_model(self):
        super().setup_distributed_model()
        self.model_without_ddp.use_mask = self.use_mask
        self.model_without_ddp.use_contrastiv = self.use_contrastiv
        self.model_without_ddp.use_dyna_exp = self.use_dyna_exp
        self.model_without_ddp.use_last_layer = self.use_last_layer