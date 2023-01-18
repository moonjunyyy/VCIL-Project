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

class L2P(_Trainer):
    def __init__(self, *args, **kwargs):
        super(L2P, self).__init__(*args, **kwargs)
        
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999
        
        self.class_mask = None
        self.class_mask_dict={}
    
    def setup_distributed_model(self):
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)
        # self.n_classes = n_classes
    
        print("[L2P] Building model...")
        self.model = select_model(self.model_name, self.dataset, n_classes)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.writer = SummaryWriter(f"{self.log_path}/tensorboard/{self.dataset}/{self.note}/seed_{self.rnd_seed}")
        
        self.model.to(self.device)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model._set_static_graph()
            self.model_without_ddp = self.model.module
        self.criterion = self.model_without_ddp.loss_fn if hasattr(self.model_without_ddp, "loss_fn") else nn.CrossEntropyLoss(reduction="mean")
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        print(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        print(f"Learnable Parameters :\t{n_params}")
        print("")

    def online_step(self, sample, samples_cnt):
        image, label = sample
        self.add_new_class(label)
        self.num_updates += self.online_iter * self.batchsize * self.world_size
        train_loss, train_acc = self.online_train([image, label], iterations=int(self.num_updates))
        self.num_updates -= int(self.num_updates)
        return train_loss, train_acc


    def add_new_class(self, class_name):# For DDP, normally go into this function
        len_class = len(self.exposed_classes)
        exposed_classes = []
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
        if self.distributed:
            exposed_classes = torch.cat(self.all_gather(torch.tensor(self.exposed_classes, device=self.device))).cpu().tolist()
            self.exposed_classes = []
            for cls in exposed_classes:
                if cls not in self.exposed_classes:
                    self.exposed_classes.append(cls)
        # print('exposed_classes:',self.exposed_classes)
        #* Class mask
        #*===========================================================================
        if self.class_mask is not None:
            self.class_mask = self.class_mask.cpu().tolist()
        else:
            self.class_mask=[]

        new=[]
        for label in class_name:
            if self.exposed_classes.index(label.item()) not in self.class_mask and \
            self.exposed_classes.index(label.item()) not in new:
                new.append(self.exposed_classes.index(label.item()))
        self.class_mask += new
        
        self.class_mask = torch.tensor(self.class_mask).to(self.device)
        #*===========================================================================
        
        
        
        
        
        self.num_learned_class = len(self.exposed_classes)
        # prev_weight = copy.deepcopy(self.model.backbone.fc.weight.data)
        # prev_bias   = copy.deepcopy(self.model.backbone.fc.bias.data)
        # self.model.backbone.reset_classifier(self.num_learned_class)

        # self.model.backbone.fc.to(self.device)
        # with torch.no_grad():
        #     if self.num_learned_class > 1:
        #         self.model.backbone.fc.weight[:len_class] = prev_weight
        #         self.model.backbone.fc.bias[:len_class]   = prev_bias
        # for param in self.optimizer.param_groups[1]['params']:
        #     if param in self.optimizer.state.keys():
        #         del self.optimizer.state[param]
        # del self.optimizer.param_groups[1]
        # self.optimizer.add_param_group({'params': self.model.backbone.fc.parameters()})
        # self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
        # # self.memory.add_new_class(cls_list=self.exposed_classes)
        # if 'reset' in self.sched_name:
        #     self.update_schedule(reset=True)

    def online_train(self, data, iterations):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        image, label = data
        # print('label-before:',label)
        for j in range(len(label)):
            label[j] = self.exposed_classes.index(label[j].item())
        # print('label-after:',label)
        for i in range(iterations):
            x = image.detach().clone()
            y = label.detach().clone()
            # if len(self.memory) > 0 and self.memory_batchsize > 0:
            #     memory_batchsize = min(self.memory_batchsize, len(self.memory))
            #     memory_images, memory_labels = self.memory.get_batch(memory_batchsize)
            #     x = torch.cat([x, memory_images], dim=0)
            #     y = torch.cat([y, memory_labels], dim=0)
            
            x = torch.cat([self.train_transform(transforms.ToPILImage()(_x)).unsqueeze(0) for _x in x])
            # x = torch.cat([self.train_transform(_x).unsqueeze(0) for _x in x])

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logit, loss = self.model_forward(x,y)
            _, preds = logit.topk(self.topk, 1, True, True)
           
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.update_schedule()

            total_loss += loss.item()
            total_correct += torch.sum(preds == y.unsqueeze(1)).item()
            total_num_data += y.size(0)

        return total_loss / iterations, total_correct / total_num_data
    
    def model_forward(self, x, y):

        # print('class_mask:',self.class_mask)
        # print('classifier:',self.model_without_ddp.backbone.fc.out_features)
        
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits,sim = self.model(x)
                logits_mask = torch.ones_like(logits, device=self.device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, self.class_mask, 0.0)
                logits = logits + logits_mask
                
                
                loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
                loss += 0.5*sim
        else:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits,sim = self.model(x)
                logits_mask = torch.ones_like(logits, device=self.device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, self.class_mask, 0.0)
                logits = logits + logits_mask
                loss = self.criterion(logits, y)
                loss += 0.5*sim
        return logits, loss
    #?==========================================================================
    # mask = class_mask[task_id]
    # not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
    # not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
    # logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
    #?==========================================================================
    
    # def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        
    #     total_loss, correct, num_data = 0.0, 0.0, 0.0

    #     # if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
    #     #     memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
    #     for i in range(iterations):
    #         self.model.train()
    #         x, y = sample
    #         x = torch.cat([self.train_transform(transforms.ToPILImage()(img)).unsqueeze(0) for img in x])
    #         y = torch.cat([torch.tensor([self.exposed_classes.index(label)]) for label in y])
    #         # if len(self.memory) > 0:
    #         #     memory_data = self.memory.get_batch(memory_batch_size)
    #         #     x = torch.cat([x, memory_data['image']])
    #         #     y = torch.cat([y, memory_data['label']])
    #         x = x.to(self.device)
    #         y = y.to(self.device)

    #         self.optimizer.zero_grad()

    #         logit, loss = self.model_forward(x,y)

    #         _, preds = logit.topk(self.topk, 1, True, True)

    #         if self.use_amp:
    #             self.scaler.scale(loss).backward()
    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()
    #         else:
    #             loss.backward()
    #             self.optimizer.step()

    #         self.update_schedule()

    #         total_loss += loss.item()
    #         correct += torch.sum(preds == y.unsqueeze(1)).item()
    #         num_data += y.size(0)

    #     return total_loss / iterations, correct / num_data

    

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def online_evaluate(self, test_loader):
        eval_dict = self.evaluation(test_loader)
        # self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"])
        return eval_dict

    def online_before_task(self,train_loader):
        # Task-Free
        # self.class_mask =None
        pass

    def online_after_task(self, cur_iter):
        self.class_mask_dict[cur_iter]=self.class_mask
        pass


    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, True)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def evaluation(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        # self.num_learned_class
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logits,_ = self.model(x)
                #! test때는 이전에 보았던 클래스들만 고려하여 Classification
                #todo 결과보고 수정하던지 하기 그냥 전체에 대해서 하는 것으로?!
                logits = logits[:,:self.num_learned_class]
                
                # logits_mask = torch.ones_like(logits, device=self.device) * float('-inf')
                # logits_mask = logits_mask.index_fill(1, self.class_mask, 0.0)
                # logits = logits + logits_mask
                
                # loss = self.criterion(logits, y)
                
                loss = self.criterion(logits, y)
                #!==============================================================
                pred = torch.argmax(logits, dim=-1)
                _, preds = logits.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()
                #!==============================================================

                total_loss += loss.item()
                label += y.tolist()
        # print(f'[EVAL] total_correct:{total_correct} // total_num_data:{total_num_data}')
        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        # print(f'[EVAL] avg_loss:{avg_loss} avg_acc:{avg_acc}')
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict