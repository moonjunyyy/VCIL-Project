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

from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler,select_optimizer_with_extern_params

import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs
from models.vit import _create_vision_transformer

from models.prompt_kearney import Prompt


logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

T = TypeVar('T', bound = 'nn.Module')

default_cfgs['vit_base_patch16_224_l2p'] = _cfg(
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

from methods._trainer import _Trainer

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class L2P_new(_Trainer):
    def __init__(self,*args,**kwargs):
        super(L2P_new,self).__init__(*args,**kwargs)
        # self.num_learned_class = 0
        # self.num_learning_class = 1
        # self.n_classes = n_classes
        # self.exposed_classes = []
        # self.seen = 0
        # self.topk = kwargs["topk"]

        # self.device = device
        # self.dataset = kwargs["dataset"]
        # self.model_name = kwargs["model_name"]
        # self.opt_name = kwargs["opt_name"]
        # self.sched_name = kwargs["sched_name"]
        # if self.sched_name == "default":
        #     self.sched_name = 'exp_reset'
        # self.lr = kwargs["lr"]

        # self.train_transform = train_transform
        # self.cutmix = "cutmix" in kwargs["transforms"]
        # self.test_transform = test_transform

        # self.memory_size = kwargs["memory_size"]
        # self.data_dir = kwargs["data_dir"]

        # self.online_iter = kwargs["online_iter"]
        # self.batch_size = kwargs["batchsize"]
        # self.temp_batchsize = kwargs["temp_batchsize"]
        # if self.temp_batchsize is None:
        #     self.temp_batchsize = self.batch_size//2
        # if self.temp_batchsize > self.batch_size:
        #     self.temp_batchsize = self.batch_size
        # self.memory_size -= self.temp_batchsize

        # self.gpu_transform = kwargs["gpu_transform"]
        # self.use_amp = kwargs["use_amp"]
        # if self.use_amp:
        #     self.scaler = torch.cuda.amp.GradScaler()

        # self.model = select_model(self.model_name, self.dataset, 1).to(self.device)
        # for n,p in self.model.named_parameters():
        #     if "fc" not in n:
        #         p.requires_grad=False
        # #?=========================================================
        # #?      Prompt Configuration section [edit everything relate to prompt here!]
        # self.poolsize = 10
        # self.prompt_len = 5
        # self.selection_size = 5
        # self.dimension=self.model.fc.in_features
        # self.prompt = Prompt(pool_size=self.poolsize,
        #                      selection_size=self.selection_size,
        #                      prompt_len=self.prompt_len,
        #                      dimension=self.dimension).to(self.device)
        # #?=========================================================
        # self.optimizer = select_optimizer_with_extern_params(self.opt_name, self.lr, self.model,is_vit=True,extern_param=self.prompt)
        # if 'imagenet' in self.dataset:
        #     self.lr_gamma = 0.99995
        # else:
        #     self.lr_gamma = 0.9999
        # self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        # self.criterion = criterion.to(self.device)
        # self.memory = MemoryDataset(self.train_transform, cls_list=self.exposed_classes,
        #                             test_transform=self.test_transform)
        # self.temp_batch = []
        # self.num_updates = 0
        # self.train_count = 0
        # self.batch_size = kwargs["batchsize"]

        # self.start_time = time.time()
        # num_samples = {'cifar10': 50000, 'cifar100': 50000, 'tinyimagenet': 100000, 'imagenet': 1281167}
        # self.total_samples = num_samples[self.dataset]
    def setup_distributed_model(self):
    
        print("Building model...")
        self.model = select_model(self.model_name, self.dataset, 1)
        for name,param in self.model.named_parameters():
            print(name)
            if 'fc' not in name:
                param.requires_grad=False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.writer = SummaryWriter(f"{self.log_path}/tensorboard/{self.dataset}/{self.note}/seed_{self.rnd_seed}")
        #?=========================================================
        #?      Prompt Configuration section [edit everything relate to prompt here!]
        self.poolsize = 10
        self.prompt_len = 5
        self.selection_size = 5
        self.dimension=self.model.fc.in_features
        self.model.prompt = Prompt(pool_size=self.poolsize,
                             selection_size=self.selection_size,
                             prompt_len=self.prompt_len,
                             dimension=self.dimension).to(self.device)
        #?=========================================================
        self.model.to(self.device)
        
        for name,param in self.model.named_parameters():
            print(name)
            # if 'fc' not in name:
            #     param.requires_grad=False
        
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model._set_static_graph()
            self.model_without_ddp = self.model.module
        self.criterion = self.model_without_ddp.loss_fn if hasattr(self.model_without_ddp, "loss_fn") else nn.CrossEntropyLoss(reduction="mean")
        # self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.optimizer = select_optimizer_with_extern_params(self.opt_name, self.lr, self.model,is_vit=True,extern_param=self.model_without_ddp.prompt)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        print(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        print(f"Learnable Parameters :\t{n_params}")
        print("")
    
    def add_new_class(self, class_name):
        len_class = len(self.exposed_classes)
        exposed_classes = []
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
        if self.distributed:
            exposed_classes = torch.cat(self.all_gather(torch.tensor(self.exposed_classes, device=self.device))).cpu().tolist()
            self.exposed_classes=[]
            for cls in exposed_classes:
                if cls not in self.exposed_classes:
                    self.exposed_classes.append(cls)
        # self.memory.add_new_class(cls_list=self.exposed_classes)

        prev_weight = copy.deepcopy(self.model_without_ddp.backbone.fc.weight.data)
        self.num_learned_class = len(self.exposed_classes)
        self.model_without_ddp.backbone.fc = nn.Linear(self.model_without_ddp.backbone.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model_without_ddp.backbone.fc.weight[:len_class] = prev_weight
        
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        params = [param for name, param in self.model_without_ddp.backbone.named_parameters() if 'fc' in name] + [param for param in self.model_without_ddp.prompt.parameters()]
        # params = [param for name, param in self.model.prompt.named_parameters() if 'fc' in name]
        self.optimizer.add_param_group({'params': params})
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_train(self, sample, iterations=1):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        image, label = sample
        for j in range(len(label)):
            label[j] = self.exposed_classes.index(label[j].item())
        for i in range(iterations):
            x = image.detach().clone()
            y = label.detach().clone()
            
            x = torch.cat([self.train_transform(transforms.ToPILImage()(_x)).unsqueeze(0) for _x in x])
            # y = torch.cat([torch.tensor([self.exposed_classes.index(label)]) for label in y])
            x = x.to(self.device)
            y = y.to(self.device)
            
            # if i==0:
            #     print(f"[Train] Label:{y}")

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
            
        # self.update_schedule()
        # print(f'[Train] exposed_Class:{len(self.exposed_classes)} // total_num_data:{total_num_data}')
        return total_loss / iterations, total_correct / total_num_data

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # logit = self.model(x)
                # loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                x = self.model_without_ddp.patch_embed(x)
                B,N,D = x.size()
                cls_tkn = self.model_without_ddp.cls_token.expand(B,-1,-1)
                x_cat = torch.cat([cls_tkn,x],dim=1)
                with torch.no_grad():
                    x = self.model_without_ddp.pos_drop(x_cat + self.model.pos_embed)
                    query = self.model_without_ddp.blocks(x)
                    # query = self.model.norm(query)[:, 0].clone()
                    query = self.model_without_ddp.norm(query)[:, 0]
                
                similarity,prompts = self.model_without_ddp.prompt(query)
                
                prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
                prompts = prompts + self.model_without_ddp.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
                
                x = self.model_without_ddp.pos_drop(x_cat + self.model_without_ddp.pos_embed)
                x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)

                x = self.model_without_ddp.blocks(x)
                x = self.model_without_ddp.norm(x)

                x = x[:, 1:self.selection_size * self.prompt_len + 1].mean(dim=1)
                x = self.model_without_ddp.fc_norm(x)
                logit = self.model_without_ddp.fc(x)
                #todo ---------------------------------------------------------------------------
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b) +0.5*similarity
        else:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # logit = self.model(x)
                # loss = self.criterion(logit, y)
                x = self.model_without_ddp.patch_embed(x)
                B,N,D = x.size()
                cls_tkn = self.model_without_ddp.cls_token.expand(B,-1,-1)
                x_cat = torch.cat([cls_tkn,x],dim=1)
                with torch.no_grad():
                    x = self.model_without_ddp.pos_drop(x_cat + self.model.pos_embed)
                    query = self.model_without_ddp.blocks(x)
                    # query = self.model.norm(query)[:, 0].clone()
                    query = self.model_without_ddp.norm(query)[:, 0]
                
                similarity,prompts = self.model_without_ddp.prompt(query)
                
                prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
                prompts = prompts + self.model_without_ddp.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
                
                x = self.model_without_ddp.pos_drop(x_cat + self.model_without_ddp.pos_embed)
                x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)

                x = self.model_without_ddp.blocks(x)
                x = self.model_without_ddp.norm(x)

                x = x[:, 1:self.selection_size * self.prompt_len + 1].mean(dim=1)
                x = self.model_without_ddp.fc_norm(x)
                logit = self.model_without_ddp.fc(x)
                #todo ---------------------------------------------------------------------------
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b) +0.5*similarity
        return logit, loss

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

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
                # if i==0:
                #     print(f"[Eval] Label:{y}")
                # logit = self.model(x)
                # loss = self.criterion(logit, y)
                logit, loss = self.model_forward(x,y)
                #!==============================================================
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()
                #!==============================================================

                total_loss += loss.item()
                label += y.tolist()
        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def online_before_task(self,task_id):
        pass

    def online_after_task(self, cur_iter):
        pass











#!=========================================================================================================================================================
    # def online_step(self, sample, sample_num, n_worker):
    #     image, label = sample
    #     for l in label:
    #         if l.item() not in self.exposed_classes:
    #             self.add_new_class(l.item())

    #     self.num_updates += self.online_iter * self.batch_size
    #     train_loss, train_acc = self.online_train([image, label], self.batch_size * 2, n_worker,
    #                                                 iterations=int(self.num_updates), stream_batch_size=self.batch_size)
    #     self.report_training(sample_num, train_loss, train_acc)
    #     for stored_sample, stored_label in zip(image, label):
    #         self.update_memory((stored_sample, stored_label))
    #     self.temp_batch = []
    #     self.num_updates -= int(self.num_updates)

    # def add_new_class(self, class_name):
    #     len_class = len(self.exposed_classes)
    #     exposed_classes = []
    #     for label in class_name:
    #         if label.item() not in self.exposed_classes:
    #             self.exposed_classes.append(label.item())
    #     if self.distributed:
    #         exposed_classes = torch.cat(self.all_gather(torch.tensor(self.exposed_classes, device=self.device))).cpu().tolist()
    #         self.exposed_classes=[]
    #         for cls in exposed_classes:
    #             if cls not in self.exposed_classes:
    #                 self.exposed_classes.append(cls)
    #     # self.memory.add_new_class(cls_list=self.exposed_classes)

    #     prev_weight = copy.deepcopy(self.model_without_ddp.backbone.fc.weight.data)
    #     self.num_learned_class = len(self.exposed_classes)
    #     self.model_without_ddp.backbone.fc = nn.Linear(self.model_without_ddp.backbone.fc.in_features, self.num_learned_class).to(self.device)
    #     with torch.no_grad():
    #         if self.num_learned_class > 1:
    #             self.model_without_ddp.backbone.fc.weight[:len_class] = prev_weight
        
    #     for param in self.optimizer.param_groups[1]['params']:
    #         if param in self.optimizer.state.keys():
    #             del self.optimizer.state[param]
    #     del self.optimizer.param_groups[1]
    #     params = [param for name, param in self.model_without_ddp.backbone.named_parameters() if 'fc' in name] + [param for param in self.model_without_ddp.prompt.parameters()]
    #     # params = [param for name, param in self.model.prompt.named_parameters() if 'fc' in name]
    #     self.optimizer.add_param_group({'params': params})
    #     if 'reset' in self.sched_name:
    #         self.update_schedule(reset=True)

    # def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        
    #     total_loss, correct, num_data = 0.0, 0.0, 0.0

    #     if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
    #         memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
    #     for i in range(iterations):
    #         self.model.train()
    #         x, y = sample
    #         x = torch.cat([self.train_transform(transforms.ToPILImage()(img)).unsqueeze(0) for img in x])
    #         y = torch.cat([torch.tensor([self.exposed_classes.index(label)]) for label in y])
    #         if len(self.memory) > 0:
    #             memory_data = self.memory.get_batch(memory_batch_size)
    #             x = torch.cat([x, memory_data['image']])
    #             y = torch.cat([y, memory_data['label']])
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

    # def model_forward(self, x, y):
    #     do_cutmix = self.cutmix and np.random.rand(1) < 0.5
    #     if do_cutmix:
    #         x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
    #         if self.use_amp:
    #             with torch.cuda.amp.autocast():
    #                 #todo ---------------------------------------------------------------------------
    #                 x = self.model.patch_embed(transforms.Resize((224,224))(x))
    #                 B,N,D = x.size()
    #                 cls_tkn = self.model.cls_token.expand(B,-1,-1)
    #                 x_cat = torch.cat([cls_tkn,x],dim=1)
    #                 with torch.no_grad():
    #                     x = self.model.pos_drop(x_cat + self.model.pos_embed)
    #                     query = self.model.blocks(x)
    #                     # query = self.model.norm(query)[:, 0].clone()
    #                     query = self.model.norm(query)[:, 0]
                    
    #                 similarity,prompts = self.prompt(query)
                    
    #                 prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
    #                 prompts = prompts + self.model.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
                    
    #                 x = self.model.pos_drop(x_cat + self.model.pos_embed)
    #                 x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)

    #                 x = self.model.blocks(x)
    #                 x = self.model.norm(x)

    #                 x = x[:, 1:self.selection_size * self.prompt_len + 1].mean(dim=1)
    #                 # x = x.mean(dim=1)
    #                 x = self.model.fc_norm(x)
    #                 logit = self.model.fc(x)
    #                 #todo ---------------------------------------------------------------------------
    #                 # logit = self.model(transforms.Resize((224,224))(x))['logits']
    #                 loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
    #         else:
    #             x = self.model.patch_embed(transforms.Resize((224,224))(x))
    #             B,N,D = x.size()
    #             cls_tkn = self.model.cls_token.expand(B,-1,-1)
    #             x_cat = torch.cat([cls_tkn,x],dim=1)
    #             with torch.no_grad():
    #                 x = self.model.pos_drop(x_cat + self.model.pos_embed)
    #                 query = self.model.blocks(x)
    #                 # query = self.model.norm(query)[:, 0].clone()
    #                 query = self.model.norm(query)[:, 0]
                
    #             similarity,prompts = self.prompt(query)
                
    #             prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
    #             prompts = prompts + self.model.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
                
    #             x = self.model.pos_drop(x_cat + self.model.pos_embed)
    #             x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)

    #             x = self.model.blocks(x)
    #             x = self.model.norm(x)

    #             x = x[:, 1:self.selection_size * self.prompt_len + 1].mean(dim=1)
    #             # x = x.mean(dim=1)
    #             x = self.model.fc_norm(x)
    #             logit = self.model.fc(x)
    #             # logit = self.model(transforms.Resize((224,224))(x))['logits']
    #             loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
    #     else:
    #         if self.use_amp:
    #             with torch.cuda.amp.autocast():
    #                 x = self.model.patch_embed(transforms.Resize((224,224))(x))
    #                 B,N,D = x.size()
    #                 cls_tkn = self.model.cls_token.expand(B,-1,-1)
    #                 x_cat = torch.cat([cls_tkn,x],dim=1)
    #                 with torch.no_grad():
    #                     x = self.model.pos_drop(x_cat + self.model.pos_embed)
    #                     query = self.model.blocks(x)
    #                     # query = self.model.norm(query)[:, 0].clone()
    #                     query = self.model.norm(query)[:, 0]
                    
    #                 similarity,prompts = self.prompt(query)
                    
    #                 prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
    #                 prompts = prompts + self.model.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
                    
    #                 x = self.model.pos_drop(x_cat + self.model.pos_embed)
    #                 x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)

    #                 x = self.model.blocks(x)
    #                 x = self.model.norm(x)

    #                 x = x[:, 1:self.selection_size * self.prompt_len + 1].mean(dim=1)
    #                 # x = x.mean(dim=1)
    #                 x = self.model.fc_norm(x)
    #                 logit = self.model.fc(x)
    #                 # logit = self.model(transforms.Resize((224,224))(x))['logits']
    #                 loss = self.criterion(logit, y)
    #         else:
    #             x = self.model.patch_embed(transforms.Resize((224,224))(x))
    #             B,N,D = x.size()
    #             cls_tkn = self.model.cls_token.expand(B,-1,-1)
    #             x_cat = torch.cat([cls_tkn,x],dim=1)
    #             with torch.no_grad():
    #                 x = self.model.pos_drop(x_cat + self.model.pos_embed)
    #                 query = self.model.blocks(x)
    #                 # query = self.model.norm(query)[:, 0].clone()
    #                 query = self.model.norm(query)[:, 0]
                
    #             similarity,prompts = self.prompt(query)
                
    #             prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
    #             prompts = prompts + self.model.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
                
    #             x = self.model.pos_drop(x_cat + self.model.pos_embed)
    #             x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)

    #             x = self.model.blocks(x)
    #             x = self.model.norm(x)

    #             x = x[:, 1:self.selection_size * self.prompt_len + 1].mean(dim=1)
    #             # x = x.mean(dim=1)
    #             x = self.model.fc_norm(x)
    #             logit = self.model.fc(x)
    #             # logit = self.model(transforms.Resize((224,224))(x))['logits']
    #             loss = self.criterion(logit, y) -0.5* similarity
    #             # loss = self.criterion(logit, y)
    #     return logit, loss

    # def report_training(self, sample_num, train_loss, train_acc):
    #     writer.add_scalar(f"train/loss", train_loss, sample_num)
    #     writer.add_scalar(f"train/acc", train_acc, sample_num)
    #     logger.info(
    #         f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
    #         f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
    #         f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
    #         f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
    #     )

    # def report_test(self, sample_num, avg_loss, avg_acc):
    #     writer.add_scalar(f"test/loss", avg_loss, sample_num)
    #     writer.add_scalar(f"test/acc", avg_acc, sample_num)
    #     logger.info(
    #         f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | "
    #     )

    # def update_memory(self, sample):
    #     self.reservoir_memory(sample)

    # def update_schedule(self, reset=False):
    #     if reset:
    #         self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
    #         for param_group in self.optimizer.param_groups:
    #             param_group["lr"] = self.lr
    #     else:
    #         self.scheduler.step()

    # def online_evaluate(self, test_loader, sample_num):
    #     eval_dict = self.evaluation(test_loader, self.criterion)
    #     self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"])
    #     return eval_dict

    # def online_before_task(self, cur_iter):
    #     # Task-Free
    #     pass

    # def online_after_task(self, cur_iter):
    #     # Task-Free
    #     pass

    # def reservoir_memory(self, sample):
    #     self.seen += 1
    #     if len(self.memory.images) >= self.memory_size:
    #         j = np.random.randint(0, self.seen)
    #         if j < self.memory_size:
    #             self.memory.replace_sample(sample, j)
    #     else:
    #         self.memory.replace_sample(sample)

    # def reset_opt(self):
    #     self.optimizer = select_optimizer_with_extern_params(self.opt_name, self.lr, self.model,self.prompt)
    #     self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    # def evaluation(self, test_loader, criterion):
    #     total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
    #     correct_l = torch.zeros(self.n_classes)
    #     num_data_l = torch.zeros(self.n_classes)
    #     label = []

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, data in enumerate(test_loader):
    #             x, y = data
    #             for j in range(len(y)):
    #                 y[j] = self.exposed_classes.index(y[j].item())
    #             x = x.to(self.device)
    #             y = y.to(self.device)
                
    #             x = self.model.patch_embed(transforms.Resize((224,224))(x))
    #             B,N,D = x.size()
    #             cls_tkn = self.model.cls_token.expand(B,-1,-1)
    #             x_cat = torch.cat([cls_tkn,x],dim=1)
                
    #             x = self.model.pos_drop(x_cat+self.model.pos_embed)
    #             query = self.model.blocks(x)
    #             query = self.model.norm(query)[:,0]
                
    #             _, prompts = self.prompt(query)
                
    #             prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
    #             prompts = prompts + self.model.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
                
    #             x = self.model.pos_drop(x_cat+self.model.pos_embed)
    #             x = torch.cat([x[:,0].unsqueeze(1),prompts,x[:,1:]],dim=1)
                
    #             x = self.model.blocks(x)
    #             x = self.model.norm(x)
                
    #             x= x[:,1:self.selection_size*self.prompt_len+1].mean(dim=1)
    #             x = self.model.fc_norm(x)
    #             logit = self.model.fc(x)
                
    #             # logit = self.model(transforms.Resize((224,224))(x))['logits']
                

    #             loss = criterion(logit, y)
    #             pred = torch.argmax(logit, dim=-1)
    #             _, preds = logit.topk(self.topk, 1, True, True)

    #             total_correct += torch.sum(preds == y.unsqueeze(1)).item()
    #             total_num_data += y.size(0)

    #             xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
    #             correct_l += correct_xlabel_cnt.detach().cpu()
    #             num_data_l += xlabel_cnt.detach().cpu()

    #             total_loss += loss.item()
    #             label += y.tolist()

    #     avg_acc = total_correct / total_num_data
    #     avg_loss = total_loss / len(test_loader)
    #     cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
    #     ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

    #     return ret

    # def _interpret_pred(self, y, pred):
    #     # xlable is batch
    #     ret_num_data = torch.zeros(self.n_classes)
    #     ret_corrects = torch.zeros(self.n_classes)

    #     xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
    #     for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
    #         ret_num_data[cls_idx] = cnt

    #     correct_xlabel = y.masked_select(y == pred)
    #     correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
    #     for cls_idx, cnt in zip(correct_cls, correct_cnt):
    #         ret_corrects[cls_idx] = cnt

    #     return ret_num_data, ret_corrects