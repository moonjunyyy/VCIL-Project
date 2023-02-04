from methods._trainer import _Trainer
from methods.er_baseline import ER
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.memory import MemoryBatchSampler
from utils.data_loader import cutmix_data
from utils.train_utils import select_optimizer, select_scheduler
import numpy as np
import torch.distributed as dist

class Ours(_Trainer):

    def __init__(self, **kwargs):
        super(Ours, self).__init__(**kwargs)
    
    def online_step(self, images, labels, idx):
        # image, label = sample
        self.add_new_class(labels[0])
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for image, label in zip(images, labels):
            loss, acc = self.online_train([image.clone(), label.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        # mask = torch.sigmoid(self.model_without_ddp.mask).tolist()
        # mask = self.model_without_ddp.mask.tolist()
        # for m in mask:
        #     print(m)
        # m = F.softmax(self.model_without_ddp.mask)
        # print(m)
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
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.mask[:len(self.exposed_classes)] = 0
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
        self.model_without_ddp.set_exposed_classes(self.exposed_classes)
        # with torch.no_grad():
        #     for name, param in self.model_without_ddp.named_parameters():
        #         if name == 'mask':
        #             param.data[len_class:len(self.exposed_classes)] = 0
        #             param.data[len(self.exposed_classes):] = -torch.inf

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, task_id):
        pass
    
    def online_train(self, data):
        self.model.train()
        self.model.train(True)
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())
        # x = torch.cat([self.train_transform(transforms.ToPILImage()(_x)).unsqueeze(0) for _x in x])
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

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        # do_cutmix = False
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(x)
                logit = logit + self.mask
                loss = lam * self.criterion(logit, labels_a.to(torch.int64)) + (1 - lam) * self.criterion(logit, labels_b.to(torch.int64))
        else:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(x)
                logit = logit + self.mask
                loss = self.criterion(logit, y.to(torch.int64))
        return logit, loss

    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        self.model.train(False)
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