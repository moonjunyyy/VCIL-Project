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


#! 1. New Class들어올때마다 Prompt 1개씩 추가
#! --> Prompt를 가능한 작게 prompt_len:1
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
        return _loss / _iter, _acc / _iter
    
    def add_new_class(self, class_name):
        #* 새로운 Class 들어오면 Prompt한개씩 확장
        #* 이전에 존재하는 Class의 경우 Temp Prompt만들어서 Distillation할수있도록 셋팅하자!
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
        
    def online_before_task(self, task_id):
        
        self.model_without_ddp.set_device(self.device)
        self.model_without_ddp.set_info(task_id)
        self.model.main_cnt=0.
        self.model.sub_cnt=0.
        self.sample_cnt = 0.
        
        # self.model_without_ddp.expand_prompt(task_id)
        
        self.reset_opt()
        
        print("main_prompt:",self.model_without_ddp.main_prompts.shape)
        print("main_key:",self.model_without_ddp.main_key.shape)
        print("sub_prompt:",self.model_without_ddp.sub_prompt.shape)
        print("sub_key:",self.model_without_ddp.sub_key.shape)
    
    

    def online_after_task(self, task_id):
        print("main_cnt:",int(self.model.main_cnt))
        print("sub_cnt:",int(self.model.sub_cnt))
        print("total_samples:", self.sample_cnt); print()
        # print("data Iter:",self.temp_batchsize * self.online_iter * self.world_size)
        pass
    
    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data
        # print('y',torch.unique(y))
        self.sample_cnt += y.size(0)
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

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
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit,_,m_idx,s_idx = self.model(x)     #* logit, feats, main_idx, sub_idx
                m_logit,s_logit = logit['main'],logit['sub']
                m_logit = m_logit + self.mask
                m_loss = lam * self.criterion(m_logit, labels_a.to(torch.int64)[m_idx]) + (1 - lam) * self.criterion(m_logit, labels_b.to(torch.int64)[m_idx])
                if s_logit is not None:
                    if s_logit.dim() == 1:
                        s_logit = s_logit.unsqueeze(0) #* sub sample1개일 경우 dim=1 (100) -> (1,100)
                    s_logit = s_logit + self.mask
                    s_loss = lam * self.criterion(s_logit, labels_a.to(torch.int64)[s_idx]) + (1 - lam) * self.criterion(s_logit, labels_b.to(torch.int64)[s_idx])
                    loss = 0.5*(m_loss+s_loss)
                else:
                    loss = m_loss
                    # loss = m_loss+s_loss
        else:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit,_,m_idx,s_idx = self.model(x)     #* logit, feats, main_idx, sub_idx
                m_logit,s_logit = logit['main'],logit['sub']
                m_logit = m_logit + self.mask
                m_loss = self.criterion(m_logit, y.to(torch.int64)[m_idx])
                if s_logit is not None:
                    if s_logit.dim() == 1:
                        s_logit = s_logit.unsqueeze(0) #* sub sample1개일 경우 dim=1 (100) -> (1,100)
                    s_logit = s_logit + self.mask
                    s_loss = self.criterion(s_logit, y.to(torch.int64)[s_idx])
                    loss = 0.5*(m_loss+s_loss)
                else:
                    loss = m_loss
                # loss = m_loss+s_loss
        if s_logit is not None:
            logit = torch.cat([m_logit,s_logit],dim=0)
            logit[m_idx] = m_logit
            logit[s_idx] = s_logit
        else:
            logit = torch.zeros_like(m_logit).to(self.device)
            logit[m_idx]=m_logit
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

                logit,_,m_idx,s_idx = self.model(x,test=True)     #* logit, feats, main_idx, sub_idx
                m_logit,s_logit = logit['main'], logit['sub']
                if s_logit is not None:
                    # print("[Test] m_logit:",m_logit.shape)
                    # print("[Test] s_logit:",s_logit.shape)
                    if s_logit.dim() == 1:
                        s_logit = s_logit.unsqueeze(0) #* sub sample1개일 경우 dim=1 (100) -> (1,100)
                    logit = torch.cat([m_logit,s_logit],dim=0)
                    logit[m_idx] = m_logit
                    logit[s_idx] = s_logit
                else:
                    logit = torch.zeros_like(m_logit).to(self.device)
                    logit[m_idx]=m_logit
                
                # print("logit:",logit.shape)
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
    
    