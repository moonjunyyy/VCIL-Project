import torch
import torch.distributed as dist
import numpy as np
from typing import Optional, Sized

class Memory:
    def __init__(self, device) -> None:
        self.device = device
        self.images = torch.empty(0, device=self.device)
        self.labels = torch.empty(0, device=self.device)
        self.cls_list = torch.empty(0, device=self.device)
        self.cls_count = torch.empty(0, device=self.device)
        self.cls_train_cnt = torch.empty(0, device=self.device)
        self.previous_idx = torch.empty(0, device=self.device)
        self.others_loss_decrease = torch.empty(0, device=self.device)

    def add_new_class(self, cls_list):
        self.cls_list = torch.tensor(cls_list, device=self.device)
        self.cls_count = torch.cat([self.cls_count, torch.zeros(len(self.cls_list) - len(self.cls_count), device=self.device)])
        self.cls_train_cnt = torch.cat([self.cls_train_cnt, torch.zeros(len(self.cls_list) - len(self.cls_train_cnt), device=self.device)])
        
    def replace_data(self, data, idx=None):
        image, label = data
        if idx is None:
            self.images = torch.cat([self.images, image.unsqueeze(0).to(self.device)])
            self.labels = torch.cat([self.labels, label.unsqueeze(0).to(self.device)])
            self.cls_count[(self.cls_list == label.item()).nonzero().squeeze()] += 1
            if self.cls_count[(self.cls_list == label.item()).nonzero().squeeze()] == 1:
                self.others_loss_decrease = torch.cat([self.others_loss_decrease, torch.tensor([0], device=self.device)])
            else:
                indice = (self.labels == label).nonzero().squeeze()
                self.others_loss_decrease = torch.cat([self.others_loss_decrease, torch.mean(self.others_loss_decrease[indice[:-1]]).unsqueeze(0)])
        else:
            self.cls_count[(self.cls_list == self.labels[idx]).nonzero().squeeze()] -= 1
            self.images[idx] = image
            self.labels[idx] = label
            self.cls_count[(self.cls_list == label).nonzero().squeeze()] += 1
            if self.cls_count[(self.cls_list == label).nonzero().squeeze()] == 1:
                self.others_loss_decrease[idx] = torch.mean(self.others_loss_decrease)
            else:
                indice = (self.labels == label).nonzero().squeeze()
                self.others_loss_decrease[idx] = torch.mean(self.others_loss_decrease[indice[indice != idx]])

    def update_gss_score(self, score, idx=None):
        if idx is None:
            self.score.append(score)
        else:
            self.score[idx] = score

    @torch.no_grad()
    def get_batch(self, batch_size, use_weight=False):
        if use_weight:
            weight = self.get_weight()
            indices = torch.multinomial(weight, batch_size)
        else:
            indices = torch.randperm(self.labels.size(0))[:batch_size]
        images = []
        labels = []
        for i in indices:
            images.append(self.images[i])
            labels.append((self.cls_list == self.labels[i]).nonzero().squeeze().item())
            self.cls_train_cnt[(self.cls_list == self.labels[i]).nonzero().squeeze().item()] += 1
        self.previous_idx = torch.cat([self.previous_idx, indices.to(self.previous_idx.device)])
        return torch.stack(images).to("cpu"), torch.LongTensor(labels).to("cpu")
    
    def update_loss_history(self, loss, prev_loss, ema_ratio=0.90, dropped_idx=None):
        if dropped_idx is None:
            loss_diff = torch.mean(loss - prev_loss)
        elif len(prev_loss) > 0:
            mask = torch.ones(len(loss), dtype=bool)
            mask[dropped_idx] = False
            loss_diff = torch.mean((torch.tensor(loss[:len(prev_loss)] - prev_loss))[mask[:len(prev_loss)]])
        else:
            loss_diff = 0
        difference = loss_diff - torch.mean(self.others_loss_decrease[self.previous_idx.to(torch.int64)]) / len(self.previous_idx)
        self.others_loss_decrease[self.previous_idx.to(torch.int64)] -= (1 - ema_ratio) * difference
        self.previous_idx = torch.empty(0, device=self.device)
    
    def get_weight(self):
        weight = np.zeros(len(self.images))
        weight = torch.zeros(self.images.size(0))
        for cls in self.cls_list:
            weight[(self.labels == cls).nonzero().squeeze()] = 1 / (self.labels == cls).nonzero().numel()
        return weight

    def __len__(self):
        return len(self.labels)