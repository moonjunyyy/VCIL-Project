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

class Prompt(nn.Module):
    def __init__(self,
                 pool_size            : int,
                 selection_size       : int,
                 prompt_len           : int,
                 dimention            : int,
                 _diversed_selection  : bool = True,
                 _batchwise_selection : bool = True,
                 **kwargs):
        super().__init__()

        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimention      = dimention
        self._diversed_selection  = _diversed_selection
        self._batchwise_selection = _batchwise_selection

        self.key     = nn.Parameter(torch.randn(pool_size, dimention, requires_grad= True))
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimention, requires_grad= True))
        
        torch.nn.init.uniform_(self.key,     -1, 1)
        torch.nn.init.uniform_(self.prompts, -1, 1)

        self.register_buffer('frequency', torch.ones (pool_size))
        self.register_buffer('counter',   torch.zeros(pool_size))
    
    def forward(self, query : torch.Tensor, **kwargs):

        B, D = query.shape
        assert D == self.dimention, f'Query dimention {D} does not match prompt dimention {self.dimention}'
        # Select prompts
        match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        if self.training and self._diversed_selection:
            topk = match * F.normalize(self.frequency, p=1, dim=-1)
        else:
            topk = match
        _ ,topk = topk.topk(self.selection_size, dim=-1, largest=False, sorted=True)
        # Batch-wise prompt selection
        if self._batchwise_selection:
            idx, counts = topk.unique(sorted=True, return_counts=True)
            _,  mosts  = counts.topk(self.selection_size, largest=True, sorted=True)
            topk = idx[mosts].clone().expand(B, -1)
        # Frequency counter
        self.counter += torch.bincount(topk.reshape(-1).clone(), minlength = self.pool_size)
        # selected prompts
        selection = self.prompts.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len, self.dimention).clone())
        simmilarity = match.gather(1, topk)
        # get unsimilar prompts also 
        return simmilarity, selection

    def update(self):
        if self.training:
            self.frequency += self.counter
        counter = self.counter.clone()
        self.counter *= 0
        if self.training:
            return self.frequency - 1
        else:
            return counter

class L2P_Model(nn.Module):
    def __init__(self,
                 pool_size      : int   = 10,
                 selection_size : int   = 5,
                 prompt_len     : int   = 5,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 lambd          : float = 0.1,
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
        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num,
                                                             drop_rate=0.,drop_path_rate=0.,drop_block_rate=None))
        for name, param in self.backbone.named_parameters():
            
            if 'fc.' not in name:
                param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        # self.head = self.backbone.fc
        
        self.prompt = Prompt(
            pool_size,
            selection_size,
            prompt_len,
            self.backbone.num_features,
            _diversed_selection  = _diversed_selection,
            _batchwise_selection = _batchwise_selection)

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
        simmilarity, prompts = self.prompt(query)
        self.simmilarity = simmilarity.mean()
        prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
        prompts = prompts + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
        x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
        x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x = x[:, 1:self.selection_size * self.prompt_len + 1].clone()
        x = x.mean(dim=1)
        x = self.backbone.fc_norm(x)
        x = self.backbone.fc(x)
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size()
        return F.cross_entropy(output, target) + self.lambd * self.simmilarity

    def convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task] = 0
        return

    def get_count(self):
        return self.prompt.update()

    def train(self: T, mode: bool = True, **kwargs):
        ten = super().train(mode)
        self.backbone.eval()
        self.prompt.train()
        # #todo-----------------
        # self.prompt.train()
        # #todo-----------------
        return ten

    # def eval(self: T, mode: bool = True, **kwargs):
    #     ten = super().eval(mode)
    #     self.backbone.eval()
    #     self.prompt.eval()
    #     # #todo-----------------
    #     # self.prompt.train()
    #     # #todo-----------------
    #     return ten
    
    # def get_params(self):
    #     params = [param for name, param in self.backbone.named_parameters() if 'fc' not in name]
    #     net_params = params + [param for param in self.prompt.parameters()]
        
    #     fc_params = [param for name, param in self.backbone.named_parameters() if 'fc' in name]
    #     return net_params, fc_params


class L2P(_Trainer):
    def __init__(self,*args,**kwargs):
        super(L2P,self).__init__(*args,**kwargs)
    def setup_distributed_model(self):
        print("Building model...")
        # self.model = select_model(self.model_name, self.dataset, 1).to(self.device)
        self.model = L2P_Model(backbone_name='vit_base_patch16_224', class_num=1)
        # for name,param in self.model.named_parameters():
        #     # print(name)
        #     if 'fc.' in name or 'prompt.' in name:
        #         param.requires_grad=True
        #     else:
        #         param.requires_grad=False
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.writer = SummaryWriter(f"{self.log_path}/tensorboard/{self.dataset}/{self.note}/seed_{self.rnd_seed}")
        
        self.model.to(self.device)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model._set_static_graph()
            self.model_without_ddp = self.model.module
        self.criterion = self.model_without_ddp.loss_fn if hasattr(self.model_without_ddp, "loss_fn") else nn.CrossEntropyLoss(reduction="mean")
        # self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        #todo------------------------------
        # params = [param for name, param in self.model_without_ddp.backbone.named_parameters() if 'fc' not in name]
        # net_params = params + [param for param in self.model_without_ddp.prompt.parameters()]
        # freeze_params = [param for name, param in self.model_without_ddp.backbone.named_parameters() if 'backbone.fc' not in name]
        
        
        # train_params = [param for name, param in self.model_without_ddp.backbone.named_parameters() if 'backbone.fc' in name] + [param for param in self.model_without_ddp.prompt.parameters()]
        # if self.opt_name=="adam":
        #     params = [param for name, param in self.model.named_parameters() if 'fc.' not in name]
        #     self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=0)
        # elif self.opt_name=="sgd":
        #     params = [param for name, param in self.model.named_parameters() if 'fc.' not in name]
        #     self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        # else:
        #     raise NotImplementedError("Please select the opt_name [adam, sgd]")
        # fc_params = [param for name, param in self.model.named_parameters() if 'fc.' in name]
        # self.optimizer.add_param_group({'params':fc_params})
        #todo------------------------------
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        print(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        print(f"Learnable Parameters :\t{n_params}")
        print("")
        # self.model = L2P_Model(backbone_name='vit_base_patch16_224_l2p', class_num=1)
        #?---------------------------------------------
        # for n,p in self.model.named_parameters():
        #     print(n)
        # for n,p in self.model.prompt.named_parameters():
        #     print(n)
        #?---------------------------------------------
        
    def online_step(self, sample, sample_num):
        image, label = sample
        # for l in label:
        #     if l.item() not in self.exposed_classes:
        #         self.add_new_class(l.item())
        self.add_new_class(label)

        self.num_updates += self.online_iter * self.batchsize
        # print(self.optimizer)
        train_loss, train_acc = self.online_train([image.clone(), label.clone()], iterations=int(self.num_updates))
        # self.update_schedule()
        # self.report_training(sample_num, train_loss, train_acc)
        self.temp_batch = []
        self.num_updates -= int(self.num_updates)
        return train_loss, train_acc

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
        # params = [param for name, param in self.model_without_ddp.backbone.named_parameters() if 'fc' in name] + [param for param in self.model_without_ddp.prompt.parameters()]
        params = [param for name, param in self.model.named_parameters() if 'fc.' in name]
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
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(x)
                loss = self.criterion(logit, y)
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
        # print(f'[EVAL] total_correct:{total_correct} // total_num_data:{total_num_data}')
        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        # print(f'[EVAL] avg_loss:{avg_loss} avg_acc:{avg_acc}')
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def online_before_task(self,task_id):
        # if task_id == 0:
        #     del self.optimizer
        #     # def get_params(self):
        #     params = [param for name, param in self.model_without_ddp.backbone.named_parameters() if 'fc' not in name]
        #     net_params = params + [param for param in self.model_without_ddp.prompt.parameters()]
            
        #     fc_params = [param for name, param in self.model_without_ddp.backbone.named_parameters() if 'fc' in name]
        #     # return net_params, fc_params
            
            
        #     # net_params, fc_params = self.model_without_ddp.get_params()
        #     if self.opt_name=="adam":
        #         self.optimizer = optim.Adam(net_params, lr=self.lr, weight_decay=0)
        #     elif self.opt_name=="sgd":
        #         self.optimizer = optim.SGD(net_params, lr=self.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        #     else:
        #         raise NotImplementedError("Please select the opt_name [adam, sgd]")
        #     self.optimizer.add_param_group({'params':fc_params})
        # else:
        #     pass
        pass

    def online_after_task(self, cur_iter):
        # Task-Free
        pass

    # def reset_opt(self):
    #     self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, True)
    #     self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    # def online_evaluate(self, test_loader):
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
    #             logit = self.model(x)

    #             loss = self.criterion(logit, y)
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
    

    # def train_data_config(self,n_task, train_dataset,train_sampler):
    #     from torch.utils.data import DataLoader
    #     for t_i in range(n_task):
    #         train_sampler.set_task(t_i)
    #         train_dataloader= DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=4)
    #         data_info = {}
    #         for i, data in enumerate(train_dataloader):
    #             _,label = data
    #             label = label.to(self.device)
                
    #             for b in range(len(label)):
    #                 if 'Class_'+str(label[b].item()) in data_info.keys():
    #                     data_info['Class_'+str(label[b].item())] +=1
    #                 else:
    #                     data_info['Class_'+str(label[b].item())] =1
    #         print(f"[Train] Task {t_i} Data Info")
    #         convert_data_info = self.convert_class_from_int_to_str(data_info)
    #         print(convert_data_info)
    #         print()
    
    # def test_data_config(self, test_dataloader,task_id):
    #     from torch.utils.data import DataLoader
    #     data_info = {}
    #     for i, data in enumerate(test_dataloader):
    #         _,label = data
    #         label = label.to(self.device)
            
    #         for b in range(len(label)):
    #             if 'Class_'+str(label[b].item()) in data_info.keys():
    #                 data_info['Class_'+str(label[b].item())] +=1
    #             else:
    #                 data_info['Class_'+str(label[b].item())] =1
                    
    #     print('<<Exposed Class>>')
    #     print(self.exposed_classes)
        
    #     print(f"[Test] Task {task_id} Data Info")
    #     print(data_info)
    #     print("<<Convert>>")
    #     convert_data_info = self.convert_class_from_int_to_str(data_info)
    #     print(convert_data_info)
    #     print()
        
        
    # def convert_class_from_int_to_str(self,data_info):
        
    #     self.convert_li
    #     for key in list(data_info.keys()):
    #         old_key = int(key[6:])
    #         data_info[self.convert_li[old_key]] = data_info.pop(key)
        
    #     return data_info