# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import time
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler

import torchvision.transforms as transforms
from methods._trainer import _Trainer


logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class LwF(_Trainer):
    def __init__(self, *args, **kwargs):
        super(LwF, self).__init__(*args, **kwargs)
        self.prev_fc=None
        self.kd_hp =10
        self.task_id=None
    
    def online_step(self, images, labels, idx):
        self.add_new_class(labels[0])
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for image, label in zip(images, labels):
            loss, acc = self.online_train([image.clone(), label.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        return _loss / _iter, _acc / _iter

    def add_new_class(self, class_name):
        # For DDP, normally go into this function
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
        # self.memory.add_new_class(cls_list=self.exposed_classes)
        self.mask[:len(self.exposed_classes)] = 0
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
    
    
    def online_before_task(self, task_id):
        self.task_id = task_id
        if task_id != 0:
            # self.prev_fc = copy.deepcopy(self.model_without_ddp.fc)
            self.prev_num_class = len(self.exposed_classes)
            self.prev_fc = nn.Linear(self.model.fc.in_features, self.prev_num_class)
            # self.prev_fc.weight = copy.deepcopy(self.model.fc.weight[:self.prev_num_class])
            # self.prev_fc.bias = copy.deepcopy(self.model.fc.bias[:self.prev_num_class])
            self.prev_fc.weight.data = self.model.fc.weight[:self.prev_num_class].clone().detach()
            self.prev_fc.bias.data = self.model.fc.bias[:self.prev_num_class].clone().detach()
            
            for p in self.prev_fc.parameters():
                p.requires_grad=False
                
        else:
            pass
    
    def online_after_task(self,task_id):
        pass
    
    def lwf_forward(self,x,y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        self.prev_fc.eval()
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                feats = self.model.forward_features(x)
                cur_logits = self.model.forward_head(feats)
                cur_logits = cur_logits + self.mask
                with torch.no_grad():
                    feats=feats[:,0]
                    feat_norm  = self.model.fc_norm(feats)
                    prev_logits = self.prev_fc(feat_norm)
                    # prev_logits = prev_logits + self.mask
                # cur_loss = self.criterion(cur_logits,y)
                cur_loss = lam * self.criterion(cur_logits, labels_a) + (1 - lam) * self.criterion(cur_logits, labels_b)
                kd_loss = self._KD_loss(cur_logits[:,:self.prev_num_class],prev_logits,2.)
                # logit = self.model(x)
                # logit += self.mask
                # loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            # loss_kd = torch.dist(feature, feature_old, 2)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                feats = self.model.forward_features(x)
                cur_logits = self.model.forward_head(feats)
                cur_logits = cur_logits + self.mask
                with torch.no_grad():
                    feats=feats[:,0]
                    feat_norm  = self.model.fc_norm(feats)
                    prev_logits = self.prev_fc(feat_norm)
                    # prev_logits = prev_logits + self.mask
                cur_loss = self.criterion(cur_logits,y)
                # kd_loss = self._KD_loss(cur_logits[:,:len(self.exposed_classes)],prev_logits[:,:len(self.exposed_classes)],2.)
                kd_loss = self._KD_loss(cur_logits[:,:self.prev_num_class],prev_logits,2.)
        # print('cur_loss:',cur_loss)
        # print('kd_loss:',kd_loss)
        return cur_logits, cur_loss +self.kd_hp*kd_loss
        
        
    def _KD_loss(self,pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        # print("cur_shape",pred.shape)
        # print("cur",pred)
        # print()
        # print("pred_shape",soft.shape)
        # print("prev",soft)
        # print()
        # print("kd:",-1 * torch.mul(soft, pred).sum())
        # print('div:',pred.shape[0])
        # if pred.isnan().sum()!=0:
        #     print("cur has a nan")
        # if soft.isnan().sum()!=0:
        #     print("prev has a nan")
            
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
    
    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        
        if self.task_id==0:
            logit, loss = self.model_forward(x,y)
        else:
            logit, loss = self.lwf_forward(x,y)
            
        _, preds = logit.topk(self.topk, 1, True, True)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(x)
                logit += self.mask
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(x)
                logit += self.mask
                loss = self.criterion(logit, y)
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

                total_loss += loss.item()
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