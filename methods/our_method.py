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

class Ours(_Trainer):
    def __init__(self, *args, **kwargs):
        super(Ours, self).__init__(*args, **kwargs)
        self.task_id = None
        self.seq_criterion = nn.CrossEntropyLoss(reduction='none')
    
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

    def online_before_task(self, task_id):
        self.task_id = task_id
        if self.task_id != 0:
            #* classifier weight 저장!
            self.prev_wts = self.model.fc.weight.data.clone().detach()
            # print("[before task]prev_wts",self.prev_wts.shape)
            
        pass
    
    def online_after_task(self,task_id):
        # self.test_data_config(test_dataloader,task_id)
        pass
    
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
            logit, loss = self.model_seq_forward(x,y)
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
    
    def model_seq_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(x)
                logit += self.mask
                
                uncertain = self._compute_uncertainty(logit,y)
                cos_dist = self._compute_cosine_dist(y)
                score = cos_dist - uncertain
                neg_idx= score<0
                #* score negative => cos_dist ==0 -> update X 
                #* Naive Learning
                score[neg_idx] =1.
                    
                #* score normalization using l2_norm
                score_norm = torch.nn.functional.normalize(score,dim=0)
                # logit = logit*score_norm
                loss_a = score_norm * self.seq_criterion(logit, labels_a)
                loss_b = score_norm * self.seq_criterion(logit, labels_b)
                loss = lam * loss_a.mean() + (1 - lam) * loss_b.mean()
        else:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(x)
                logit += self.mask
                
                uncertain = self._compute_uncertainty(logit,y)
                cos_dist = self._compute_cosine_dist(y)
                score = cos_dist - uncertain
                neg_idx= score<0
                #* score negative => cos_dist ==0 -> update X 
                #* Naive Learning
                score[neg_idx] =1.
                    
                #* score normalization using l2_norm
                score_norm = torch.nn.functional.normalize(score,dim=0)
                # logit = logit*score_norm
                
                loss = score_norm * self.seq_criterion(logit, y)
                loss = loss.mean()
        return logit, loss
    
    def _compute_uncertainty(self,logit,labels):
        logit = logit[:,:len(self.exposed_classes)]
        s_logit = torch.softmax(logit,dim=1)
        # unc = 1 - s_logit
        uncertain=[]
        for idx, label in enumerate(labels):
            uncertain.append(1.-s_logit[idx,label])
            
        return torch.tensor(uncertain).to(self.device)
    
    def _compute_cosine_dist(self,label):
        # self.prev_wts = self.model.fc.weight.data.clone().detach()
        #? Task 0 학습이후 Task 1 첫 Iter에서는 Parameter 동일!
        #? Cosine distance 동일한 경우 -> Parameter 변화 X
        #todo cosine_dist ==0 -> consider only uncertainty
        #todo Unseen Class의 경우: 동일하게 처리
        
        # prev_wts = self.prev_wts[:len(self.exposed_classes)]
        
        # cur_wts = self.model.fc.weight.data[:len(self.exposed_classes)].clone().detach()
        prev_wts = self.prev_wts[label]
        
        cur_wts = self.model.fc.weight.data[label].clone().detach()
        
        # print("prev_wts",prev_wts.shape)
        # print("cur_wts",cur_wts.shape)
        cos_dist = 1- torch.cosine_similarity(prev_wts,cur_wts,dim=1)
        
        return cos_dist
        
        
    
        
        
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