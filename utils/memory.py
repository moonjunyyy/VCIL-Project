import torch
import torch.distributed as dist
import numpy as np
from typing import Optional, Sized

class Memory:
    def __init__(self, data_source=None) -> None:
        
        self.data_source = data_source
        if self.data_source is not None:
            self.images = []

        self.memory = torch.empty(0)
        self.labels = torch.empty(0)
        self.cls_list = torch.empty(0)
        self.cls_count = torch.empty(0)
        self.cls_train_cnt = torch.empty(0)
        self.previous_idx = torch.empty(0)
        self.others_loss_decrease = torch.empty(0)

    def add_new_class(self, cls_list):
        self.cls_list = torch.tensor(cls_list)
        self.cls_count = torch.cat([self.cls_count, torch.zeros(len(self.cls_list) - len(self.cls_count))])
        self.cls_train_cnt = torch.cat([self.cls_train_cnt, torch.zeros(len(self.cls_list) - len(self.cls_train_cnt))])

    def replace_data(self, data, idx=None):
        index, label = data
        if self.data_source is not None:
            image, label = self.data_source.__getitem__(index)
        if idx is None:
            if self.data_source is not None:
                self.images.append(image.unsqueeze(0))
            self.memory = torch.cat([self.memory, torch.tensor([index])])
            self.labels = torch.cat([self.labels, torch.tensor([label])])
            self.cls_count[(self.cls_list == label).nonzero().squeeze()] += 1
            if self.cls_count[(self.cls_list == label).nonzero().squeeze()] == 1:
                self.others_loss_decrease = torch.cat([self.others_loss_decrease, torch.tensor([0])])
            else:
                indice = (self.labels == label).nonzero().squeeze()
                self.others_loss_decrease = torch.cat([self.others_loss_decrease, torch.mean(self.others_loss_decrease[indice[:-1]]).unsqueeze(0)])
        else:
            if self.data_source is not None:
                self.images[idx] = image.unsqueeze(0)
            _label = self.labels[idx]
            self.cls_count[(self.cls_list == _label).nonzero().squeeze()] -= 1
            self.memory[idx] = index
            self.labels[idx] = label
            self.cls_count[(self.cls_list == label).nonzero().squeeze()] += 1
            if self.cls_count[(self.cls_list == label).nonzero().squeeze()] == 1:
                self.others_loss_decrease[idx] = torch.mean(self.others_loss_decrease)
            else:
                indice = (self.labels == label).nonzero().squeeze()
                self.others_loss_decrease[idx] = torch.mean(self.others_loss_decrease[indice[indice != idx]])

    def update_loss_history(self, loss, prev_loss, ema_ratio=0.90, dropped_idx=None):
        if dropped_idx is None:
            loss_diff = torch.mean(loss - prev_loss)
        elif len(prev_loss) > 0:
            mask = torch.ones(len(loss), dtype=bool)
            mask[torch.tensor(dropped_idx, dtype=torch.int64).squeeze()] = False
            loss_diff = torch.mean((loss[:len(prev_loss)] - prev_loss)[mask[:len(prev_loss)]])
        else:
            loss_diff = 0
        difference = loss_diff - torch.mean(self.others_loss_decrease[self.previous_idx.to(torch.int64)]) / len(self.previous_idx)
        self.others_loss_decrease[self.previous_idx.to(torch.int64)] -= (1 - ema_ratio) * difference
        self.previous_idx = torch.empty(0)
    
    def get_weight(self):
        weight = np.zeros(len(self.images))
        weight = torch.zeros(self.images.size(0))
        for cls in self.cls_list:
            weight[(self.labels == cls).nonzero().squeeze()] = 1 / (self.labels == cls).nonzero().numel()
        return weight

    def update_gss_score(self, score, idx=None):
        if idx is None:
            self.score.append(score)
        else:
            self.score[idx] = score

    def __len__(self):
        return len(self.labels)

    def sample(self, memory_batchsize):
        assert self.data_source is not None
        idx = torch.randperm(len(self.images), dtype=torch.int64)[:memory_batchsize]
        images = []
        labels = []
        for i in idx:
            images.append(self.images[i])
            labels.append(self.labels[i])
        return torch.cat(images), torch.tensor(labels)

class MemoryBatchSampler(torch.utils.data.Sampler):
    def __init__(self, memory: Memory, batch_size: int, iterations: int = 1) -> None:
        self.memory = memory
        self.batch_size = batch_size
        self.iterations = int(iterations)
        self.indices = torch.cat([torch.randperm(len(self.memory), dtype=torch.int64)[:min(self.batch_size, len(self.memory))] for _ in range(self.iterations)]).tolist()
        for i, idx in enumerate(self.indices):
            self.indices[i] = int(self.memory.memory[idx])
    
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class MemoryOrderedSampler(torch.utils.data.Sampler):
    def __init__(self, memory: Memory, batch_size: int, iterations: int = 1) -> None:
        self.memory = memory
        self.batch_size = batch_size
        self.iterations = int(iterations)
        self.indices = torch.cat([torch.arange(len(self.memory), dtype=torch.int64) for _ in range(self.iterations)]).tolist()
        for i, idx in enumerate(self.indices):
            self.indices[i] =  int(self.memory.memory[idx])
    
    def __iter__(self):
        if dist.is_initialized():
            return iter(self.indices[dist.get_rank()::dist.get_world_size()])
        else:
            return iter(self.indices)
    def __len__(self):
        if dist.is_initialized():
            return len(self.indices[dist.get_rank()::dist.get_world_size()])
        else:
            return len(self.indices)