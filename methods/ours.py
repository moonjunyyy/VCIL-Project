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

import copy
#! 1. New Class들어올때마다 Prompt 1개씩 추가
#! --> Prompt를 가능한 작게 prompt_len:1
class Ours(_Trainer):

    def __init__(self, **kwargs):
        super(Ours, self).__init__(**kwargs)
        # self.prev_query = None
        self.old_fc = None
        self.old_mask = None
        self.old_query = None
        
    
    def prev_trace(self):
        self.old_fc = copy.deepcopy(self.model.backbone.fc)
        self.old_mask = self.mask.clone().detach()
        for p in self.old_fc.parameters():
            p.requires_grad=False
        self.old_fc.eval()
        self.old_prompt = self.model.main_prompts.clone().detach()
        self.old_key = self.model.main_key.clone().detach()
        # self.old_query = query.clone().detach()
        
    
    def online_step(self, images, labels, idx):
        # image, label = sample
        #* check task shift
        
        
        if self.old_fc is not None:
            flag = self.is_task_changed(images[0])
            # if flag:
            #     print("Detect New Task incomes!!")
            #     self.model.expand_prompt(self.device)
            #     self.reset_opt()
        self.add_new_class(labels[0])
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        self.prev_trace()
        for Bidx,(image, label) in enumerate(zip(images, labels)):
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
        # self.memory.add_new_class(cls_list=self.exposed_classes)
        self.mask[:len(self.exposed_classes)] = 0
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
        
    def online_before_task(self, task_id):
        
        # self.model_without_ddp.set_device(self.device)
        # self.model_without_ddp.set_info(task_id)
        self.model.main_cnt=0.
        self.model.sub_cnt=0.
        self.sample_cnt = 0.
        
        print("main_prompt:",self.model_without_ddp.main_prompts.shape)
        print("main_key:",self.model_without_ddp.main_key.shape)
        # print("sub_prompt:",self.model_without_ddp.sub_prompt.shape)
        # print("sub_key:",self.model_without_ddp.sub_key.shape)
    
    

    def online_after_task(self, task_id):
        print("main_cnt:",int(self.model.main_cnt))
        print("sub_cnt:",int(self.model.sub_cnt))
        print("total_samples:", self.sample_cnt); print()
        # print("data Iter:",self.temp_batchsize * self.online_iter * self.world_size)
        # pass
    
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
        logit, loss, main_idx, query = self.model_forward(x,y)
        _, preds = logit.topk(self.topk, 1, True, True)
        
        
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()
        
        # self.prev_trace()
        self.old_query = query[main_idx]

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)
        
        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                ori_logit,feats,query,main_idx,sub_idx= self.model(x)     #* logit, feats, main_idx, sub_idx
                #* logit scale
                if len(sub_idx) !=0:
                    ori_logit[sub_idx] = (len(sub_idx)/(len(sub_idx)+len(main_idx)))*ori_logit[sub_idx]
                    # ori_logit[sub_idx] = (len(sub_idx)/len(main_idx))*ori_logit[sub_idx]
                #     ori_logit[main_idx] = ((len(sub_idx)+len(main_idx))/(len(main_idx)))*ori_logit[main_idx]
                
                logit = ori_logit + self.mask
                a_loss, a_sim = self.criterion(logit, labels_a)
                b_loss, b_sim = self.criterion(logit, labels_b)
                #* loss scale
                # if len(sub_idx) !=0:
                #     a_loss[sub_idx] = (len(main_idx)/len(sub_idx))*a_loss[sub_idx]
                #     b_loss[sub_idx] = (len(main_idx)/len(sub_idx))*b_loss[sub_idx]
                a_loss = a_loss.mean()
                b_loss = b_loss.mean()
                # loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                loss = lam * (a_loss + a_sim) + (1 - lam) * (b_loss + b_sim)
                if self.old_fc is not None:
                    old_feats = self.model.old_forward(x,self.old_prompt,self.old_key)
                    old_logit = self.old_fc(old_feats) + self.old_mask
                    kd_loss = self._KD_loss(ori_logit[:,:len(self.old_mask)],old_logit[:,:len(self.old_mask)],T=2.)
                    
                    loss += 0.02 * kd_loss
        else:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                ori_logit,feats,query,main_idx,sub_idx= self.model(x)     #* logit, feats, main_idx, sub_idx
                # if len(sub_idx) !=0:
                #     ori_logit[sub_idx] = (len(sub_idx)/(len(sub_idx)+len(main_idx)))*ori_logit[sub_idx]
                #     ori_logit[main_idx] = ((len(sub_idx)+len(main_idx))/(len(main_idx)))*ori_logit[main_idx]
                    
                logit = ori_logit + self.mask
                loss,sim = self.criterion(logit, y)
                if len(sub_idx) !=0:
                    ori_logit[sub_idx] = (len(sub_idx)/(len(sub_idx)+len(main_idx)))*ori_logit[sub_idx]
                #     loss[sub_idx] = (len(main_idx)/len(sub_idx))*loss[sub_idx]
                
                loss = loss.mean() + sim
                if self.old_fc is not None:
                    old_feats = self.model.old_forward(x,self.old_prompt,self.old_key)
                    old_logit = self.old_fc(old_feats) + self.old_mask
                    kd_loss = self._KD_loss(ori_logit[:,:len(self.old_mask)],old_logit[:,:len(self.old_mask)],T=2.)
                    
                    loss += 0.02 * kd_loss
            

        return logit, loss, main_idx, query

    def _KD_loss(self,pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

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

                logit = self.model(x,test=True)     #* logit, feats, main_idx, sub_idx
                logit = logit + self.mask
                
                loss,sim = self.criterion(logit, y)
                loss = loss.mean()+sim
                
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
    
    
    def is_task_changed(self,x):
        self.model.eval()
        # self.old_fc.eval()
        # self.old_fc.eval()
        with torch.no_grad():
            x =x.to(self.device)
            query,main_idx,_ = self.model.task_forward(x)
            # query = self.model.backbone.fc_norm(query)
            
            # new_S_logit = self.model.backbone.fc(query[main_idx]) + self.mask
            # new_S_ent = self._get_entropy(new_S_logit)
            new_S_logit = self.model.backbone.fc(query[main_idx]) + self.mask
            new_S_ent = self._get_entropy(new_S_logit)
            
            #*      Plan B
            old_S_logit = self.model.backbone.fc(self.old_query) + self.mask
            old_S_ent = self._get_entropy(old_S_logit)
            # old_S_logit = self.model.backbone.fc(query[main_idx]) + self.mask
            # old_S_ent = self._get_entropy(old_S_logit)
            
        result = new_S_ent - old_S_ent > 1.1
        print("old fc Entropy:",old_S_ent)
        print("new fc Entropy:",new_S_ent)
        print('result:',new_S_ent - old_S_ent)
        print()
        if result:
            # cur_q = query[main_idx].mean(dim=0).unsqueeze(0)
            # prev_q = self.old_query.mean(dim=0).unsqueeze(0)
            # size = min(query[main_idx].shape[0],self.old_query.shape[0])
            print("task change!")
            # print("KL_DIV_mean:", F.kl_div(cur_q.softmax(dim=1).log(),prev_q.softmax(dim=1),reduction='batchmean'))
            # print("KL_DIV_each:", F.kl_div(query[main_idx][:size].softmax(dim=1).log(),
            print("old fc Entropy:",old_S_ent)
            print("new fc Entropy:",new_S_ent)
            print()
        
        return result
            
    def _get_entropy(self,p, mean=True):
        p = F.softmax(p,dim=1)
        en = -torch.sum(p * torch.log(p+1e-5), 1)
        if mean:
            return torch.mean(en)
        else:
            return en

            
            
            
        
        # self.prev_query = (prev_main_query + cur_main_query)/2
        # self.query.
        
    
