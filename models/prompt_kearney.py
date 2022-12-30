
import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt(nn.Module):
    def __init__(self,
                 pool_size=10,
                 selection_size=5,
                 prompt_len=5,
                 dimension=768,
                 _diversed_selection = True,
                 _batchwise_selection= False,
                 **kwargs):
        super().__init__()

        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimension      = dimension
        self._diversed_selection  = _diversed_selection
        self._batchwise_selection = _batchwise_selection

        self.key     = nn.Parameter(torch.randn(pool_size, dimension, requires_grad= True))
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimension, requires_grad= True))
        
        torch.nn.init.uniform_(self.key,     -1, 1)
        torch.nn.init.uniform_(self.prompts, -1, 1)

        # self.register_buffer('frequency', torch.ones (pool_size))
        # self.register_buffer('counter',   torch.zeros(pool_size))
    
    def forward(self, query : torch.Tensor,  **kwargs):

        B, D = query.shape
        assert D == self.dimension, f'Query dimension {D} does not Cdist prompt dimension {self.dimension}'

        # Select prompts
        Cdist= 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        
        # if self.training and self._diversed_selection:
        #     topk = Cdist * F.normalize(self.frequency, p=1, dim=-1)
        # else:
        #     topk = Cdist
        
        _ ,topk = Cdist.topk(self.selection_size, dim=-1, largest=False, sorted=True)

        # Batch-wise prompt selection
        # if self._batchwise_selection:
        #     idx, counts = topk.unique(sorted=True, return_counts=True)
        #     _,  mosts  = counts.topk(self.selection_size, largest=True, sorted=True)
        #     topk = idx[mosts].clone().expand(B, -1)

        # Frequency counter
        # self.counter += torch.bincount(topk.reshape(-1).clone(), minlength = self.pool_size)

        # selected prompts
        selection = self.prompts.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len, self.dimension))
        # print("prompt selection shape:",selection.shape)
        similarity = Cdist.gather(1, topk)
        # print("similarity selection shape:",similarity.shape)

        return similarity.mean(), selection
    
    # def update(self):
    #     if self.training:
    #         self.frequency += self.counter
    #     counter = self.counter.clone()
    #     self.counter *= 0
    #     if self.training:
    #         return self.frequency - 1
    #     else:
    #         return counter