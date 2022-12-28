import torch
from torch.utils.data.sampler import Sampler
from typing import Optional, Sized

class MemorySampler(Sampler):
    def __init__(self, data_source: Optional[Sized], num_tasks, m, n, rnd_seed, cur_iter) -> None:
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.generator  = torch.Generator().manual_seed(rnd_seed)
        self.n  = n
        self.m  = m
        self.disjoint_num   = len(self.classes) * n / 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        # Divide classes into N% of disjoint and (100 - N)% of blurry
        class_order         = torch.randperm(len(self.classes), generator=self.generator)
        self.disjoint_classes   = class_order[:self.disjoint_num]
        self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
        self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
        self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

        # Get indices of disjoint and blurry classes
        self.disjoint_indices   = [[] for _ in range(num_tasks)]
        self.blurry_indices     = [[] for _ in range(num_tasks)]
        for i in range(len(self.targets)):
            for j in range(num_tasks):
                if self.targets[i] in self.disjoint_classes[j]:
                    self.disjoint_indices[j].append(i)
                elif self.targets[i] in self.blurry_classes[j]:
                    self.blurry_indices[j].append(i)

        # Randomly shuffle M% of blurry indices
        blurred = []
        for i in range(num_tasks):
            blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
            self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
        blurred = torch.tensor(blurred)
        blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()

        num_blurred = len(blurred) // num_tasks
        for i in range(num_tasks):
            self.blurry_indices[i] += blurred[:num_blurred]
            blurred = blurred[num_blurred:]
        
        self.indices = [[] for _ in range(num_tasks)]
        for i in range(num_tasks):
            print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
            self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
            self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
        self.task = cur_iter
    
    def __iter__(self):
        return iter(self.indices[self.task])

    def __len__(self):
        return len(self.indices[self.task])

    def set_task(self, cur_iter):
        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter


class OnlineTestSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], exposed_class, rnd_seed) -> None:
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.generator  = torch.Generator().manual_seed(rnd_seed)
        self.exposed_class  = exposed_class
        self.indices    = [i for i in range(self.data_source.__len__()) if self.targets[i] in self.exposed_class]
        
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)