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

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class CLIB(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.memory_size = kwargs["memory_size"]

        # Samplewise importance variables
        self.loss = np.array([])
        self.dropped_idx = []
        self.memory_dropped_idx = []
        self.imp_update_counter = 0
        self.memory = MemoryDataset(self.train_transform, cls_list=self.exposed_classes,
                                    test_transform=self.test_transform, save_test=True, keep_history=True)
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

    def online_step(self, sample, sample_num, n_worker):
        image, label = sample
        for l in label:
            if l.item() not in self.exposed_classes:
                self.add_new_class(l.item())
        for stored_sample, stored_label in zip(image, label):
            self.update_memory((stored_sample, stored_label))
        self.num_updates += self.online_iter * self.batch_size
        # if len(self.temp_batch) == self.temp_batchsize:
        train_loss, train_acc = self.online_train([], self.batch_size, n_worker,
                                                    iterations=int(self.num_updates), stream_batch_size=0)
        self.report_training(sample_num, train_loss, train_acc)
        self.temp_batch = []
        self.num_updates -= int(self.num_updates)
    def update_memory(self, sample):
        self.samplewise_importance_memory(sample)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=0):
        # print("This is an online_train process in clib.py")
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        for i in range(iterations):
            self.model.train()
            if len(sample) > 0:
                x, y = sample
                x = torch.cat([self.train_transform(transforms.ToPILImage()(img)).unsqueeze(0) for img in x])
                y = torch.cat([torch.tensor([self.exposed_classes.index(label)]) for label in y])
            if len(self.memory) > 0:
                if len(sample) > 0:
                    memory_data = self.memory.get_batch(memory_batch_size)
                    x = torch.cat([x, memory_data['image']])
                    y = torch.cat([y, memory_data['label']])
                else:
                    memory_data = self.memory.get_batch(batch_size)
                    x = memory_data['image']
                    y = memory_data['label']
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logit, loss = self.model_forward(x, y)
            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.samplewise_loss_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)

        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        sdict = copy.deepcopy(self.optimizer.state_dict())
        fc_params = sdict['param_groups'][1]['params']
        if len(sdict['state']) > 0:
            fc_weight_state = sdict['state'][fc_params[0]]
            fc_bias_state = sdict['state'][fc_params[1]]
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})
        if len(sdict['state']) > 0:
            if 'adam' in self.opt_name:
                fc_weight = self.optimizer.param_groups[1]['params'][0]
                fc_bias = self.optimizer.param_groups[1]['params'][1]
                self.optimizer.state[fc_weight]['step'] = fc_weight_state['step']
                self.optimizer.state[fc_weight]['exp_avg'] = torch.cat([fc_weight_state['exp_avg'],
                                                                        torch.zeros([1, fc_weight_state['exp_avg'].size(
                                                                            dim=1)]).to(self.device)], dim=0)
                self.optimizer.state[fc_weight]['exp_avg_sq'] = torch.cat([fc_weight_state['exp_avg_sq'],
                                                                           torch.zeros([1, fc_weight_state[
                                                                               'exp_avg_sq'].size(dim=1)]).to(
                                                                               self.device)], dim=0)
                self.optimizer.state[fc_bias]['step'] = fc_bias_state['step']
                self.optimizer.state[fc_bias]['exp_avg'] = torch.cat([fc_bias_state['exp_avg'],
                                                                      torch.tensor([0]).to(
                                                                          self.device)], dim=0)
                self.optimizer.state[fc_bias]['exp_avg_sq'] = torch.cat([fc_bias_state['exp_avg_sq'],
                                                                         torch.tensor([0]).to(
                                                                             self.device)], dim=0)
        self.memory.add_new_class(cls_list=self.exposed_classes)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def update_schedule(self, reset=False):
        if self.sched_name == 'adaptive_lr':
            self.adaptive_lr(period=self.lr_period, min_iter=self.lr_length)
            self.model.train()
        else:
            super().update_schedule(reset)

    def samplewise_loss_update(self, ema_ratio=0.90, batchsize=512):
        self.imp_update_counter += 1
        if self.imp_update_counter % self.imp_update_period == 0:
            if len(self.memory) > 0:
                self.model.eval()
                with torch.no_grad():
                    x = self.memory.device_img
                    y = torch.LongTensor(self.memory.labels)
                    y = y.to(self.device)
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            logit = torch.cat(
                                [self.model(torch.cat(x[i * batchsize:min((i + 1) * batchsize, len(x))]).to(self.device))
                                for i in range(-(-len(x) // batchsize))], dim=0)

                    else:
                        logit = torch.cat(
                            [self.model(torch.cat(x[i * batchsize:min((i + 1) * batchsize, len(x))]).to(self.device))
                             for i in range(-(-len(x) // batchsize))], dim=0)

                    loss = F.cross_entropy(logit, y, reduction='none').cpu().numpy()
                self.memory.update_loss_history(loss, self.loss, ema_ratio=ema_ratio, dropped_idx=self.memory_dropped_idx)
                self.memory_dropped_idx = []
                self.loss = loss

    def samplewise_importance_memory(self, sample):
        x, y = sample
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(y.item())] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            cand_idx = self.memory.cls_idx[cls_to_replace]
            score = self.memory.others_loss_decrease[cand_idx]
            idx_to_replace = cand_idx[np.argmin(score)]
            self.memory.replace_sample(sample, idx_to_replace)
            self.dropped_idx.append(idx_to_replace)
            self.memory_dropped_idx.append(idx_to_replace)
        else:
            self.memory.replace_sample(sample)
            self.dropped_idx.append(len(self.memory) - 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)

    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        if self.imp_update_counter % self.imp_update_period == 0:
            self.train_count += 1
            mask = np.ones(len(self.loss), bool)
            mask[self.dropped_idx] = False
            if self.train_count % period == 0:
                if self.lr_is_high:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.high_lr_loss.append(np.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]))
                        if len(self.high_lr_loss) > min_iter:
                            del self.high_lr_loss[0]
                    self.prev_loss = self.loss
                    self.lr_is_high = False
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.low_lr
                        param_group["initial_lr"] = self.low_lr
                else:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.low_lr_loss.append(np.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]))
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
                    print(pvalue)
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

    def online_step(self, sample, samples_cnt):
        image, label = sample
        self.add_new_class(label)
        self.update_memory(sample)
        self.num_updates += self.online_iter * self.batchsize
        train_loss, train_acc = self.online_train([torch.empty((0,)), torch.empty((0,))], iterations=int(self.num_updates))
        self.num_updates -= int(self.num_updates)
        return train_loss, train_acc
    
    def update_memory(self, sample):
        image, label = sample
        image = torch.cat(self.all_gather(image.to(self.device)))
        label = torch.cat(self.all_gather(label.to(self.device)))
        x, y = sample
        
        for x, y in zip(image, label):
            if len(self.memory) >= self.memory_size:
                label_frequency = copy.deepcopy(self.memory.cls_count)
                label_frequency[self.exposed_classes.index(y.item())] += 1
                cls_to_replace = torch.argmax(label_frequency)
                cand_idx = self.memory.labels == self.memory.cls_list[cls_to_replace]
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

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a.to(torch.long)) + (1 - lam) * self.criterion(logit, labels_b.to(torch.long))
        else:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(x)
                loss = self.criterion(logit, y.to(torch.long))
        return logit, loss

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
           
            self.samplewise_loss_update()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.update_schedule()

            total_loss += loss.item()
            total_correct += torch.sum(preds == y.unsqueeze(1)).item()
            total_num_data += y.size(0)


        return total_loss / iterations, total_correct / total_num_data

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
        if self.sched_name == 'adaptive_lr':
            self.adaptive_lr(period=self.lr_period, min_iter=self.lr_length)
            self.model.train()
        else:
            super().update_schedule(reset)

    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        if self.imp_update_counter % self.imp_update_period == 0:
            self.train_count += 1
            mask = torch.ones(len(self.loss), dtype=bool)
            mask[self.dropped_idx] = False
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
                    print(pvalue)
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
                self.model.eval()
                with torch.no_grad():
                    x = self.memory.images.clone()
                    y = self.memory.labels.clone()
                    if self.distributed:
                        if dist.get_rank() == 0:
                            xlist = [x[i::dist.get_world_size()] for i in range(dist.get_world_size())]
                            ylist = [y[i::dist.get_world_size()] for i in range(dist.get_world_size())]
                            x = x[::dist.get_world_size()].contiguous()
                            y = y[::dist.get_world_size()].contiguous()
                            dist.scatter(x, xlist, src=0)
                            dist.scatter(y, ylist, src=0)
                        else:
                            x = x[dist.get_rank()::dist.get_world_size()].contiguous()
                            y = y[dist.get_rank()::dist.get_world_size()].contiguous()
                            dist.scatter(x, src=0)
                            dist.scatter(y, src=0)
                    for j in range(len(y)):
                        y[j] = self.exposed_classes.index(y[j].item())
                    x = x.to(self.device)
                    y = y.to(self.device)
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        logit = torch.cat(
                            [self.model(transforms.Resize((self.inp_size,self.inp_size))(x[i * batchsize:min((i + 1) * batchsize, len(x))].to(self.device)))
                            for i in range(-(-len(x) // batchsize))], dim=0)
                    loss = F.cross_entropy(logit, y.to(torch.int64), reduction='none')
                    if self.distributed:
                        loss = torch.cat(self.all_gather(loss), dim=-1).flatten()
                self.memory.update_loss_history(loss, self.loss, ema_ratio=ema_ratio, dropped_idx=self.memory_dropped_idx)
                self.memory_dropped_idx = []
                self.loss = loss