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

import torch.distributed as dist

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class ER(_Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super(ER, self).__init__(*args, **kwargs)

    def online_step(self, sample, samples_cnt):
        image, label = sample
        self.add_new_class(label)
        self.num_updates += self.online_iter * self.batchsize
        train_loss, train_acc = self.online_train([image.clone(), label.clone()], iterations=int(self.num_updates))
        self.update_memory(sample)
        self.num_updates -= int(self.num_updates)
        return train_loss, train_acc
    
    def update_memory(self, sample):
        image, label = sample
        if self.distributed:
            image = torch.cat(self.all_gather(image.to(self.device)))
            label = torch.cat(self.all_gather(label.to(self.device)))
        idx = []
        if self.is_main_process():
            for lbl in label:
                self.seen += 1
                if len(self.memory) < self.memory_size:
                    idx.append(-1)
                else:
                    j = torch.randint(0, self.seen, (1,), generator=self.generator).item()
                    if j < self.memory_size:
                        idx.append(j)
                    else:
                        idx.append(self.memory_size)
        if self.distributed:
            idx = torch.tensor(idx).to(self.device)
            size = torch.tensor([idx.size(0)]).to(self.device)
            dist.broadcast(size, 0)
            if dist.get_rank() != 0:
                idx = torch.zeros(size.item(), dtype=torch.long).to(self.device)
            dist.barrier() # wait for all processes to reach this point
            dist.broadcast(idx, 0)
            idx = idx.cpu().tolist()
        # idx = torch.cat(self.all_gather(torch.tensor(idx).to(self.device))).cpu().tolist()
        for i, index in enumerate(idx):
            if len(self.memory) >= self.memory_size:
                if index < self.memory_size:
                    self.memory.replace_data([image[i], label[i]], index)
            else:
                self.memory.replace_data([image[i], label[i]])        

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, task_id):
        pass
    
    def online_train(self, data, iterations):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        image, label = data
        for j in range(len(label)):
            label[j] = self.exposed_classes.index(label[j].item())
        for i in range(iterations):
            x = image.detach().clone()
            y = label.detach().clone()
            if len(self.memory) > 0 and self.memory_batchsize > 0:
                memory_batchsize = min(self.memory_batchsize, len(self.memory))
                memory_images, memory_labels = self.memory.get_batch(memory_batchsize)
                x = torch.cat([x, memory_images], dim=0)
                y = torch.cat([y, memory_labels], dim=0)
            
            x = torch.cat([self.train_transform(transforms.ToPILImage()(_x)).unsqueeze(0) for _x in x])

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