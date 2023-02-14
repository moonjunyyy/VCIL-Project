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

class Ours(_Trainer):
    def __init__(self, *args, **kwargs):
        super(Ours, self).__init__(*args, **kwargs)
        
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999
        
        self.class_mask = None
        self.class_mask_dict={}
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
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
        x, y = data
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

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
            feat = self.model(x,only_feat=True)
            # print("x",x.shape)
            # print("feat:",feat.shape)
            ign_score,sample_g = self.get_score(ref_head=ref_fc,feat=feat,y=y)
            loss,logit = self._get_loss(ign_score,feat,y,sample_g)
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

                logit = self.model(x)
                logit = logit + self.mask
                loss = self.criterion(logit, y)
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
        # Task-Free
        self.model_without_ddp.convert_train_task(self.exposed_classes)

    def online_after_task(self, cur_iter):
        # self.model_without_ddp.convert_train_task(self.exposed_classes)
        pass

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, True)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
        
    
    
    def _compute_grads_uncert(self,ref_head,feat,y):
        sample_criterion = nn.CrossEntropyLoss(reduction='none')
        batch_criterion = nn.CrossEntropyLoss()
        sample_g = []
        ref_head.zero_grad()
        tmp_logit = ref_head(feat)[:,:len(self.exposed_classes)]
    #     print('feats')
    #     print(feat[:3]); print()
    #     print('tmp_logit')
    #     print(tmp_logit); print()

        p = torch.softmax(tmp_logit,dim=1)
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
        batch_g = ref_head.weight.grad[:len(self.exposed_classes)].clone()  # C,dim
        idx = torch.arange(len(y))
        batch_g=batch_g[y[idx]]    #B,dim
        ref_head.zero_grad()
    #     del ref_head
    #     print('uncertain')
    #     print(uncert); print()
    #     print('sample_g')
    #     print(sample_g); print()
    #     print('batch_g')
    #     print(batch_g); print()
        
        
        return uncert, sample_g, batch_g
    
    
    
    def _get_ignore(self,ref_head,feat,y):
        uncert, sample_g, batch_g = self._compute_grads_uncert(ref_head,feat,y)
        sample_l2norm = torch.norm(sample_g,p=2,dim=1)  # B
        if sample_l2norm.sum().item() == 0: #all loss is zero case
            return None,sample_g
        batch_l2norm = torch.norm(batch_g,p=2,dim=1)    # B
        uncert_soft = torch.softmax(uncert,dim=0)
        ignore_score = torch.max((sample_l2norm-batch_l2norm),torch.zeros(1,device=self.device))
        # ignore_score = 1. + torch.max((sample_l2norm/batch_l2norm),torch.zeros(1,device=self.device))
        return ignore_score,sample_g

    def _get_drift(self,y,sample_g):
        idx = torch.arange(len(y))
        pre_wts = self.model.backbone.fc.weight[y[idx]].clone().detach()
    #     with torch.no_grad():
        post_wts = pre_wts-sample_g
        drift = 1 - torch.cosine_similarity(pre_wts,post_wts,dim=1) # B
        return drift

    def get_score(self,ref_head,feat,y):
        ign_feat = self.model.backbone.fc_norm(feat[:,0].clone().detach())
        
        ignore_score,sample_g = self._get_ignore(ref_head,ign_feat,y)
        # drift_score = self._get_drift(y,sample_g)
        
        return ignore_score,sample_g
    
    #* for ignore problem
    def str_loss(self,logit,y,str_score):
        # targets = F.one_hot(y,len(self.exposed_classes))
        gamma=1.
        log_p = F.log_softmax(logit,dim=1)
        ce = F.nll_loss(log_p,y,reduction='none')
        
        loss = (str_score**gamma)*ce
        
        return loss
        
    #* for drift problem
    def cp_loss(self,y,sample_g):
        # idx = torch.arange(len(y))
        sample_w = self.model.backbone.fc.weight[y]
        next_w = sample_w.clone().detach() - self.lr*sample_g
        
        cp_loss = 1- torch.cosine_similarity(sample_w,next_w,dim=1)
        
        return cp_loss


    def _get_loss(self,ign_score,feat,y,sample_g):
        # for p in ref_head.parameters():
        #     p.requires_grad = False
        #*#########################################################################
        #* strength_loss
        if ign_score != None:
            ign_feat = self.model.backbone.fc_norm(feat[:,0])
            #* 원래는 Head는 학습X !!
            #* ign_logit = ref_head(ign_feat)[:,:len(self.exposed_classes)]
            ign_logit = self.model.backbone.fc(ign_feat)[:,:len(self.exposed_classes)]
            # ign_logit = ign_logit*(1/ign_score[:,None])
            str_loss = self.str_loss(ign_logit, y)
            str_loss = str_loss.mean()
        else:
            str_loss = torch.zeros(1,device=self.device)
        
    #     print('ignore_score')
    #     print(ign_score)
    #     print('ignore_loss:',ignore_loss)
        
        #*########################################################################
        #* compensation_loss
        # cur_wts = 
        # drift_loss = self.criterion(ori_logit, y) * drift_score
        cp_loss = self.cp_loss(y,sample_g)
        cp_loss = cp_loss.mean()
        #*########################################################################
        #* CE_loss
        ce_logit = self.model.backbone.forward_head(feat)[:,:len(self.exposed_classes)]
        ce_loss = self.criterion(ce_logit, y)
        ce_loss = ce_loss.mean() + self.model._get_sim()
        loss = ce_loss + str_loss + cp_loss 
        # print(f"ignore: {alpha*ignore_loss.item():.4f} drift:{(1-alpha)*drift_loss.item():.4f} loss: {loss.item():.4f}")
        return loss, ce_logit