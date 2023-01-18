import logging
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import ttest_ind

import torch.distributed as dist

from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset, StreamDataset, MemoryDataset
from utils.memory import MemoryBatchSampler, MemoryOrderedSampler

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class CLIB(ER):
    def __init__(self, *args, **kwargs) -> None:
        super(CLIB, self).__init__(*args, **kwargs)
        self.loss = torch.empty((0,))
        self.dropped_idx = []
        self.memory_dropped_idx = []
        self.imp_update_counter = 0
        self.imp_update_period = kwargs['imp_update_period']
        if kwargs["sched_name"] == 'default':
            self.sched_name = 'adaptive_lr'

        # Adaptive LR variables
        self.lr_step = kwargs["lr_step"]
        self.lr_length = kwargs["lr_length"]
        self.lr_period = kwargs["lr_period"]
        self.prev_loss = None
        self.lr_is_high = True
        self.high_lr = self.lr
        self.low_lr = self.lr_step * self.lr
        self.high_lr_loss = []
        self.low_lr_loss = []
        self.current_lr = self.lr

    def setup_distributed_dataset(self):
        super(CLIB, self).setup_distributed_dataset()
        self.loss_update_dataset = self.datasets[self.dataset](root=self.data_dir, train=True, download=True,
                                     transform=transforms.Compose([transforms.Resize((self.inp_size,self.inp_size)),transforms.ToTensor()]))

    def online_step(self, images, labels, idx):
        self.add_new_class(labels[0])
        self.update_memory(idx, labels[0])
        self.memory_sampler = MemoryBatchSampler(self.memory, self.memory_batchsize, self.temp_batchsize * self.online_iter * self.world_size)
        self.memory_dataloader = iter(DataLoader(self.train_dataset, batch_size=self.memory_batchsize, sampler=self.memory_sampler, num_workers=self.n_worker, pin_memory=True))
       # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for image, label in zip(images, labels):
            loss, acc = self.online_train([image.clone(), label.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        self.num_updates -= int(self.num_updates)
        return _loss / _iter, _acc / _iter
    
    # def add_new_class(self, class_name):
    #     len_class = len(self.exposed_classes)
    #     exposed_classes = []
    #     for label in class_name:
    #         if label.item() not in self.exposed_classes:
    #             self.exposed_classes.append(label.item())
    #     if self.distributed:
    #         exposed_classes = torch.cat(self.all_gather(torch.tensor(self.exposed_classes, device=self.device))).cpu().tolist()
    #         self.exposed_classes = []
    #         for cls in exposed_classes:
    #             if cls not in self.exposed_classes:
    #                 self.exposed_classes.append(cls)
    #     self.num_learned_class = len(self.exposed_classes)
        
    #     with torch.no_grad():
    #         prev_weight = copy.deepcopy(self.model.fc.weight.data)
    #         self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)

    #         if self.num_learned_class > 1:
    #             self.model.fc.weight[:len_class] = prev_weight
    #         sdict = copy.deepcopy(self.optimizer.state_dict())
    #         fc_params = sdict['param_groups'][1]['params']
    #         if len(sdict['state']) > 0:
    #             fc_weight_state = sdict['state'][fc_params[0]]
    #             fc_bias_state = sdict['state'][fc_params[1]]
    #         for param in self.optimizer.param_groups[1]['params']:
    #             if param in self.optimizer.state.keys():
    #                 del self.optimizer.state[param]
    #         del self.optimizer.param_groups[1]
    #         self.optimizer.add_param_group({'params': self.model.fc.parameters()})
    #         if len(sdict['state']) > 0:
    #             if 'adam' in self.opt_name:
    #                 fc_weight = self.optimizer.param_groups[1]['params'][0]
    #                 fc_bias = self.optimizer.param_groups[1]['params'][1]
    #                 self.optimizer.state[fc_weight]['step'] = fc_weight_state['step']
    #                 self.optimizer.state[fc_weight]['exp_avg'] = torch.cat([fc_weight_state['exp_avg'],
    #                                                                         torch.zeros([1, fc_weight_state['exp_avg'].size(
    #                                                                             dim=1)]).to(self.device)], dim=0)
    #                 self.optimizer.state[fc_weight]['exp_avg_sq'] = torch.cat([fc_weight_state['exp_avg_sq'],
    #                                                                         torch.zeros([1, fc_weight_state[
    #                                                                             'exp_avg_sq'].size(dim=1)]).to(
    #                                                                             self.device)], dim=0)
    #                 self.optimizer.state[fc_bias]['step'] = fc_bias_state['step']
    #                 self.optimizer.state[fc_bias]['exp_avg'] = torch.cat([fc_bias_state['exp_avg'],
    #                                                                     torch.tensor([0]).to(
    #                                                                         self.device)], dim=0)
    #                 self.optimizer.state[fc_bias]['exp_avg_sq'] = torch.cat([fc_bias_state['exp_avg_sq'],
    #                                                                         torch.tensor([0]).to(
    #                                                                             self.device)], dim=0)
    #     self.memory.add_new_class(cls_list=self.exposed_classes)
    #     if 'reset' in self.sched_name:
    #         self.update_schedule(reset=True)

    def update_memory(self, sample, label):
        # Update memory
        if self.distributed:
            sample = torch.cat(self.all_gather(sample.to(self.device)))
            label  = torch.cat(self.all_gather(label.to(self.device)))
            sample = sample.cpu()
            label  = label.cpu()
        
        for x, y in zip(sample, label):
            if len(self.memory) >= self.memory_size:
                label_frequency = copy.deepcopy(self.memory.cls_count)
                label_frequency[self.exposed_classes.index(y.item())] += 1
                cls_to_replace = torch.argmax(label_frequency)
                cand_idx = (self.memory.labels == self.memory.cls_list[cls_to_replace]).nonzero().squeeze()
                score = self.memory.others_loss_decrease[cand_idx]
                idx_to_replace = cand_idx[torch.argmin(score)]
                self.memory.replace_data([x, y], idx_to_replace)
                self.dropped_idx.append(idx_to_replace)
                self.memory_dropped_idx.append(idx_to_replace)
            else:
                self.memory.replace_data([x, y])
                self.dropped_idx.append(len(self.memory) - 1)
                self.memory_dropped_idx.append(len(self.memory) - 1)

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, task_id):
        pass

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data
        if len(self.memory) > 0 and self.memory_batchsize > 0:
            # memory_batchsize = min(self.memory_batchsize, len(self.memory))
            # memory_images, memory_labels = self.memory.get_batch(memory_batchsize)
            memory_images, memory_labels = next(self.memory_dataloader)
            x = torch.cat([x, memory_images], dim=0)
            y = torch.cat([y, memory_labels], dim=0)
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())
        # x = torch.cat([self.train_transform(transforms.ToPILImage()(_x)).unsqueeze(0) for _x in x])
        # x = torch.cat([self.train_transform(_x).unsqueeze(0) for _x in x])

        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        logit, loss = self.model_forward(x,y)
        _, preds = logit.topk(self.topk, 1, True, True)
        
        self.samplewise_loss_update()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)


        return total_loss, total_correct / total_num_data

    def update_schedule(self, reset=False):
        if self.sched_name == 'adaptive_lr':
            self.adaptive_lr(period=self.lr_period, min_iter=self.lr_length)
            self.model.train()
        else:
            super().update_schedule(reset)

    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        if self.imp_update_counter % self.imp_update_period == 0:
            self.train_count += 1
            if len(self.loss) == 0: return
            mask = torch.ones(len(self.loss), dtype=bool)
            mask[torch.tensor(self.dropped_idx, dtype=torch.int64).squeeze()] = False
            if self.train_count % period == 0:
                if self.lr_is_high:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.high_lr_loss.append(torch.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]).cpu())
                        if len(self.high_lr_loss) > min_iter:
                            del self.high_lr_loss[0]
                    self.prev_loss = self.loss
                    self.lr_is_high = False
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.low_lr
                        param_group["initial_lr"] = self.low_lr
                else:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.low_lr_loss.append(torch.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]).cpu())
                        if len(self.low_lr_loss) > min_iter:
                            del self.low_lr_loss[0]
                    self.prev_loss = self.loss
                    self.lr_is_high = True
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.high_lr
                        param_group["initial_lr"] = self.high_lr
                self.dropped_idx = []
                if len(self.high_lr_loss) == len(self.low_lr_loss) and len(self.high_lr_loss) >= min_iter:
                    stat, pvalue = ttest_ind(self.low_lr_loss, self.high_lr_loss, equal_var=False, alternative='greater')
                    # print(pvalue)
                    if pvalue < significance:
                        self.high_lr = self.low_lr
                        self.low_lr *= self.lr_step
                        self.high_lr_loss = []
                        self.low_lr_loss = []
                        if self.lr_is_high:
                            self.lr_is_high = False
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.low_lr
                                param_group["initial_lr"] = self.low_lr
                        else:
                            self.lr_is_high = True
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.high_lr
                                param_group["initial_lr"] = self.high_lr
                    elif pvalue > 1 - significance:
                        self.low_lr = self.high_lr
                        self.high_lr /= self.lr_step
                        self.high_lr_loss = []
                        self.low_lr_loss = []
                        if self.lr_is_high:
                            self.lr_is_high = False
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.low_lr
                                param_group["initial_lr"] = self.low_lr
                        else:
                            self.lr_is_high = True
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.high_lr
                                param_group["initial_lr"] = self.high_lr

    def samplewise_loss_update(self, ema_ratio=0.90, batchsize=512):
        self.imp_update_counter += 1
        if self.imp_update_counter % self.imp_update_period == 0:
            if len(self.memory) > 0:
                self.memory_sampler = MemoryOrderedSampler(self.memory, self.batchsize)
                self.memory_dataloader = DataLoader(self.loss_update_dataset, batch_size=batchsize, sampler=self.memory_sampler, num_workers=4, pin_memory=True)
                self.model.eval()
                with torch.no_grad():
                    logit = [self.model(x.to(self.device), y.to(self.device)) + self.mask for (x, y) in self.memory_dataloader]
                    loss = F.cross_entropy(logit, self.memory.labels.to(self.device), reduction='none')
                    if self.distributed:
                        loss = torch.cat(self.all_gather(loss), dim=-1).flatten()
                self.memory.update_loss_history(loss, self.loss, ema_ratio=ema_ratio, dropped_idx=self.memory_dropped_idx)
                self.memory_dropped_idx = []
                self.loss = loss
