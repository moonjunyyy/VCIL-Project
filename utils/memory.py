import torch
import torch.distributed as dist
import numpy as np
from typing import Optional, Sized

class Memory:
    def __init__(self, data_source: Optional[Sized]) -> None:
        self.data_source = data_source
        self.memory = []
        self.images = []
        self.labels = []
        self.cls_idx = []
        self.cls_dict = {}
        self.cls_train_cnt = np.array([], dtype=int)
    
    def add_new_class(self, cls_list):
        self.cls_list = cls_list
        self.cls_idx.append([])
        self.cls_dict = {self.cls_list[i]:i for i in range(len(self.cls_list))}
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)

    def replace_data(self, data, idx=None, distributed=False):
        image, label = data
        if idx is None:
            self.images.append(image)
            self.labels.append(label)
            self.cls_idx[self.cls_dict[label]].append(len(self.memory)-1)
        else:
            self.cls_idx[self.cls_dict[label[idx]]].remove(idx)
            self.images[idx] = image
            self.labels[idx] = label
            self.cls_idx[self.cls_dict[label]].append(idx)
        
        if distributed:
            memory = torch.tensor(self.memory, dtype=torch.int64)
            dist.broadcast(memory, dist.get_rank(), async_op=False)
            self.memory = memory.tolist()
            for i in range(len(self.cls_idx)):
                cls_idx = torch.tensor(self.cls_idx[i], dtype=torch.int64)
                dist.broadcast(cls_idx, dist.get_rank(), async_op=False)
                self.cls_idx[i] = cls_idx.tolist()

    def update_gss_score(self, score, idx=None):
        if idx is None:
            self.score.append(score)
        else:
            self.score[idx] = score

    @torch.no_grad()
    def get_batch(self, batch_size, use_weight=False):
        if use_weight:
            weight = self.get_weight()
            indices = np.random.choice(range(len(self.memory)), size=batch_size, p=weight/np.sum(weight), replace=False)
        else:
            indices = np.random.choice(range(len(self.memory)), size=batch_size, replace=False)
        images = []
        labels = []
        for i in indices:
            images.append(self.images[i])
            labels.append(self.cls_dict[self.labels[i]])
            self.cls_train_cnt[self.cls_dict[self.labels[i]]] += 1
        # self.previous_idx = np.append(self.previous_idx, indices)
        return torch.stack(images), torch.LongTensor(labels)
    
    def update_loss_history(self, loss, prev_loss, ema_ratio=0.90, dropped_idx=None):
        if dropped_idx is None:
            loss_diff = np.mean(loss - prev_loss)
        elif len(prev_loss) > 0:
            mask = np.ones(len(loss), bool)
            mask[dropped_idx] = False
            loss_diff = np.mean((loss[:len(prev_loss)] - prev_loss)[mask[:len(prev_loss)]])
        else:
            loss_diff = 0
        difference = loss_diff - np.mean(self.others_loss_decrease[self.previous_idx]) / len(self.previous_idx)
        self.others_loss_decrease[self.previous_idx] -= (1 - ema_ratio) * difference
        self.previous_idx = np.array([], dtype=int)
    
    def get_weight(self):
        weight = np.zeros(len(self.images))
        for i, indices in enumerate(self.cls_idx):
            weight[indices] = 1/self.cls_count[i]
        return weight

    def __len__(self):
        return len(self.memory)