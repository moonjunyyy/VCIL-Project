
#todo ##################################################################################################
#todo #                                                                                                #
#todo #   "DIB:Dunk In Basket: fancy movement for continual learning (Si-Blurry or i-Blurry Scenario)" #
#todo #                                                                                                #
#todo ##################################################################################################

#todo Prompt를 통한 pretrain parameter control방식이 L2P와는 차이가 있으면 좋을 듯 한데....
#todo L2P and Dual Prompt와는 다른 Prompt Tuning방식이 필요하다 

#todo  --> weight를 저장해서 사용하자 
#! 기존 Promtp는 결국 input의 정보량을 늘려주는 것 잘 예측할 수 있도록 추가적인 정보를 넣어주는 것
#! Our method는 각 Class의 정보가 서로 다른 곳에 위치할 수 있도록 Mapping시켜주자
#! Prompt 아니어도 된다!! advisor??
#todo 새로운 Class가 들어오면 다른 CLass sample들과 다른 곳에 맵핑 될 수 있도록 하는 Layer가 필요하다!! 
#todo Layer는 쓰고 버릴 수 있으면 더 좋을 텐데 불가능 할듯??
#todo 그냥 애초에 섹션을 나누어서 CLass들이 겹치지 않고 Drift가 발생해도 서로 overlap 없도록 구역을 만들자!!!
#todo Linear Layer 3개 정도 쌓아서 OVA NET Fusion!!
#! Embedding space에서 Mapping 하여 각 Class들이 서로 완전히 다른 곳에 위치 할 수 있도록하자

#! 유사한 정보들은 가져 갈 수 있으면 좋다??
#!      --> No! ours는 유사한 정보를 이용하도록 학습하는 것 X
#!      --> Ours Method는 Embedding Space를 독립적으로 Class마다 가질 수 있도록 구축하는 것
#!      --> 유사한 정보의 사용은 Pretrained만을 이용한다.
#!      --> 유사한 정보는 결국 Forgetting 혹은 degradation을 심화시킬 수 있는 요소라고 생각한다.

#? Class마다 Basket을 만들고 해당 CLass의 샘플을 해당 Class Basket에 Dunk in하자!
#? Class마다 Basket parameter 생성: 겹치면 안되고 Margin이 충분해야한다.!!
#?          --> Entropy Minimization등의 OVANet based Technique 필요 
#? 각 Sample들은 Basket을 통해서 안정적으로 Mapping이 되야한다!!!
#? Inference시에 명확하고 심플한 Basket Prediction이 필요하다 
#? --> Crucial!!!! HOW??? : Uncertainty or Entropy based!!

from typing import TypeVar
import copy
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs

from models.vit import _create_vision_transformer



logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

T = TypeVar('T', bound = 'nn.Module')

default_cfgs['vit_base_patch16_224_l2p'] = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        num_classes=21843)

# Register the backbone model to timm
@register_model
def vit_base_patch16_224_l2p(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_l2p', pretrained=pretrained, **model_kwargs)
    return model


class Ours(nn.Module):
    def __init__(self,
                 pool_size      : int   = 1,
                 selection_size : int   = 1,
                 prompt_len     : int   = 20,    #* prompt len for multu layer
                 n_task = 5,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 lambd          : float = 0.5,
                 _batchwise_selection  : bool = False,
                 _diversed_selection   : bool = True,
                 **kwargs):

        super().__init__()
        
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        if pool_size < selection_size:
            raise ValueError('pool_size must be larger than selection_size')

        self.prompt_len     = prompt_len
        self.selection_size = selection_size
        self.lambd          = lambd
        self._batchwise_selection = _batchwise_selection
        self.class_num            = class_num
        
        self.main_cnt = 0.
        self.sub_cnt =0.
        self.deep_layer = [2,3,4]
        
        # self.task_id =0
        
        # self.simmilarity=0.
        # self.num_heads=12
        # self.device='cpu'
        self.add_module('backbone', timm.models.create_model(backbone_name, pretrained=True, num_classes=class_num,
                                                             drop_rate=0.,drop_path_rate=0.,drop_block_rate=None))
        
        self.n_task = n_task
        
        # self.sub_key     = nn.Parameter(torch.randn(1, self.backbone.embed_dim))
        # self.sub_prompt = nn.Parameter(torch.randn(1, self.prompt_len, self.backbone.embed_dim))
        
        # self.main_key     = nn.Parameter(torch.randn(self.n_task, self.backbone.embed_dim))
        # self.main_prompts = nn.Parameter(torch.randn(self.n_task,1,2, len(self.deep_layer), self.prompt_len, self.backbone.embed_dim))
        self.main_key     = nn.Parameter(torch.randn(1, self.backbone.embed_dim))
        self.main_prompts = nn.Parameter(torch.randn(1,1,2, len(self.deep_layer), self.prompt_len, self.backbone.embed_dim))
        
        #* weight initialization
        torch.nn.init.uniform_(self.main_key,     -1, 1)
        torch.nn.init.uniform_(self.main_prompts, -1, 1)
        
        # self.criterion = nn.CrossEntropyLoss()
        # torch.nn.init.uniform_(self.main_key, -1, 1)
        # torch.nn.init.uniform_(self.main_prompts, -1, 1)
        #?-----------------------------------------------------------
        #? RuntimeError: you can only change requires_grad flags of leaf variables.
        #? If you want to use a computed variable in a subgraph that doesn't require differentiation use var_no_grad = var.detach().
        
        # for idx in range(self.main_prompt.shape[0]):
        #     self.main_prompt[idx].requires_grad = False
        # self.main_prompt[0].requires_grad=True
        
        # for idx in range(self.main_key.shape[0]):
        #     self.main_key[idx].requires_grad=False
        # self.main_key[0].requires_grad=True
        #?-----------------------------------------------------------
        for name, param in self.backbone.named_parameters():
                param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

    
    #! Train때와 Test시 구별 해야함
    def forward(self, inputs, test=False, **kwargs):
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()
            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0]
        
        if test:
            #* num_main = self.task_id +1
            #* cand_keys = torch.cat([self.main_key[:num_main],self.sub_key],dim=0)
            m_prompts=[]
            s_prompts=[]
            # main_idx = []
            # sub_idx = []
            
            query_norm = self.l2_normalize(query,dim=1)
            #* key_norm = self.l2_normalize(cand_keys,dim=1)
            key_norm = self.l2_normalize(self.main_key,dim=1)
            similarity = torch.matmul(query_norm, key_norm.t()) # B, keys
            # simmilarity = F.cosine_similarity(query, cand_keys, dim=1)
            topk = similarity.topk(self.selection_size, dim=1)[1]
            
            # for sample_i, select in enumerate(topk):
            #     main_idx.append(sample_i)
            #     m_prompts.append(select)
                # if select.item()<= self.task_id:
                #     main_idx.append(sample_i)
                #     m_prompts.append(select)
                # else:
                #     sub_idx.append(sample_i)
                #     s_prompts.append(select)
            
            # main_idx = torch.tensor(main_idx)
            # m_prompts = torch.tensor(m_prompts)
            # sub_idx = torch.tensor(sub_idx)
            
            m_prompts = self.main_prompts[topk].squeeze(1)
            # m_prompts = self.main_prompts[topk]
            main_key = self.main_key[topk]
            # print("[Test] m-Prompts:",m_prompts.shape)
            # print("[Test] main_key:",main_key.shape)
        else:   #* train
            #* split main and sub class sample
            main_idx,sub_idx = self.split_samples(query)
            self.main_cnt += main_idx.sum().item()
            self.sub_cnt += sub_idx.sum().item()
            
            # print("main",main_idx.sum())
            # print("sub",sub_idx.sum())
            # print()
            
            m_prompts = self.main_prompts[-1] #* 1,2,3,20,768
            main_key = self.main_key[-1].unsqueeze(0)
            
            one,dual,num_layer,p_length,dim = m_prompts.shape
            m_prompts = m_prompts.unsqueeze(0).expand(x.shape[0],one,dual,num_layer,p_length,dim) #* B,1,dual,Num_layer,prompt_len,dim
        

        x,sim = self.main_prompt_forward(x,query,m_prompts,main_key=main_key)
        self.simmilarity = sim.mean()
        feats = x[:,0]
        feats = self.backbone.fc_norm(feats)
        x = self.backbone.fc(feats)
        
        
        if test:
            return x
        else:
            return x,feats,query,main_idx,sub_idx
        #todo ==========================================================
    
    def l2_normalize(self,x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    
    def split_samples(self,query):
        dist_table = self.l2_sim(query)
        mdist = dist_table.mean(dim=1)
        # print("max:",mdist.max().item())
        # print("min:",mdist.min().item())
        # print("avg:",mdist.mean().item())
        max_min = mdist.max().item()-mdist.min().item()
        max_avg = mdist.max().item()-mdist.mean().item()
        avg_min = mdist.mean().item()-mdist.min().item()
        # print("Max_Min_discrepency",max_min)
        # print("Max_AVG_discrepency",max_avg)
        # print("AVG_Min_discrepency",avg_min); print()
        
        
        if (max_avg - avg_min) <0:
            # print("Mean_Max")
            n_dist,threshold = self.norm_proc(mdist)
            hs_dist = self.hard_sigmoid(n_dist,threshold)
            # main_idx = hs_dist < 0.5
            main_idx = hs_dist < threshold
            sub_idx = ~main_idx
        else:
            main_idx = mdist > 0.
            sub_idx = ~main_idx
        
        return main_idx,sub_idx
    
    def l2_sim(self,feature1):
        feature2 = feature1
        Feature = feature1.expand(feature1.size(0), feature1.size(0), feature1.size(1)).transpose(0, 1)
        return torch.norm(Feature - feature2, p=2, dim=2)
    
    def hard_sigmoid(self,x,threshold):
        K=1.5
        return 1 / (1 +torch.exp(-K*(x-threshold)))
    
    def min_max(self,x):
        min = x.min()
        max = x.max()
        denom = max-min
        minmax= []
        for i in range(len(x)):
            minmax.append( (x[i] -min) / denom)
        minmax = torch.tensor(minmax)
        return minmax
    
    def norm_proc(self,mdist):
        n_dist = self.min_max(mdist)
        median_score = n_dist.sort()[0][int(round(len(n_dist)/2))]
        mean_score = n_dist.mean()
        # print("max:",n_dist.max().item())
        # print("median:",median_score.item())
        # print("mean:",mean_score.item())
        threshold = torch.tensor([median_score,mean_score]).max().item()
        return n_dist,threshold
    

    def expand_prompt(self,device):
        #* self.main_key     = nn.Parameter(torch.randn(1, self.backbone.embed_dim))
        #* self.main_prompts = nn.Parameter(torch.randn(1,1,2, len(self.deep_layer), self.prompt_len, self.backbone.embed_dim))
        
        prompt_num,one,dual,num_layer,prompt_len, dim = self.main_prompts.shape
        prev_prompt_data = self.main_prompts.data
        
        self.main_prompts = nn.Parameter(torch.randn(prompt_num+1,one,dual, num_layer, prompt_len, dim, device=device))
        torch.nn.init.uniform_(self.main_prompts, -1, 1)
        self.main_prompts[:-1].data = prev_prompt_data
        # self.main_prompts[-1].data = prev_prompt_data[-1]
        
        key_num,key_dim = self.main_key.shape
        prev_key_data = self.main_key.data
        
        self.main_key = nn.Parameter(torch.randn(key_num+1, key_dim,device=device))
        torch.nn.init.uniform_(self.main_key,-1, 1)
        self.main_key[:-1].data = prev_key_data
        # self.main_key[-1].data = prev_key_data[-1]
        
        
        print("[Expand Prompt Status]")
        print("main_prompt:",self.main_prompts.shape)
        print("main_key:",self.main_key.shape)
        print()

    def sub_prompt_forward(self,sub_x,sub_query,prompts):
        #* prompt attach to input tokens
        B,N,D = sub_x.size()
        simmilarity = F.cosine_similarity(sub_query.unsqueeze(1), self.sub_key, dim=-1)

        # print("sub_prompt:",prompts.shape)  #* 1,5,768 --> pool_size, length,dim
        
        prompts = prompts.contiguous().expand(B, self.selection_size * self.prompt_len, D)
        prompts = prompts + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
        # print("sub_prompt:",prompts.shape)
        
        sub_x = torch.cat((sub_x[:,0].unsqueeze(1), prompts, sub_x[:,1:]), dim=1)
        sub_x = self.backbone.blocks(sub_x)
        sub_x = self.backbone.norm(sub_x)
        
        return sub_x,simmilarity
    
    def main_prompt_forward(self,main_x,main_query,prompts,main_key):
        #* prompt attach to MSA
        B,N,D = main_x.size()
        simmilarity = torch.cosine_similarity(main_query, main_key, dim=1)
        
        batch_size, one,dual,num_layer, length, embed_dim = prompts.shape
        prompts = prompts.contiguous().view(batch_size,dual*num_layer,length,embed_dim)
        
        #todo MSA attach
        for idx, block in enumerate(self.backbone.blocks):
            # if idx in self.deep_layer:
            #todo self.pre_attn(self.backbone.norm1(x), prompts)
            #! 수정 시작하자!
            
            q = block.norm1(main_x)
            k = q.clone()
            v = q.clone()
            if idx in self.deep_layer:
                # print("Prompts:",prompts.shape)
                k = torch.cat([prompts[:,self.deep_layer.index(idx)*2+0], k], dim=1)
                v = torch.cat([prompts[:,self.deep_layer.index(idx)*2+1], v], dim=1)
            #?---------------------------------------
            #? MSA Layer forward start
            attn   = block.attn
            weight = attn.qkv.weight
            bias   = attn.qkv.bias
            B, N, C = q.shape
            q = F.linear(q, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = k.shape
            k = F.linear(k, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = v.shape
            v = F.linear(v, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attn = block.attn.attn_drop(attn)

            attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
            attn = block.attn.proj(attn)
            attn = block.attn.proj_drop(attn)
            #?---------------------------------------
            #? MSA Layer forward done
            
            main_x = main_x + block.drop_path1(block.ls1(attn))
            main_x = main_x + block.drop_path2(block.ls2(block.mlp(block.norm2(main_x))))
            # else:
            #     main_x = block(main_x)
        #todo================================================================
        main_x = self.backbone.norm(main_x)
        
        return main_x,simmilarity
    
    
    def train(self: T, mode : bool = True, **kwargs):
        ten = super().train()
        self.backbone.eval()
        return ten
    
    def eval(self: T, mode : bool = True, **kwargs):
        ten = super().eval()
        self.backbone.eval()
        return ten
    
    def loss_fn(self, output, target):
        # B, C = output.size()
        #* return F.cross_entropy(output, target) + (1-self.simmilarity)
        return F.cross_entropy(output, target,reduction='none'), (1-self.simmilarity)

    # def set_device(self,device):
    #     self.device = device
    # def set_info(self,task_id):
    #     #* Task Id is needed to control prompt scope
    #     #* we already make whold prompt
    #     #* if you expand prompts whenever new task are incomed, you don't need this one
    #     self.task_id = task_id
    
    def query_forward(self,inputs):
        # self.eval()
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()
            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0]
        main_idx,sub_idx = self.split_samples(query)
        
        return query,main_idx,sub_idx

    def task_forward(self, x):
        
        query,main_idx,sub_idx = self.query_forward(x)
        # x = self.backbone.patch_embed(x)
        # main_idx,sub_idx = self.split_samples(query)
        # # self.main_cnt += main_idx.sum().item()
        #     # self.sub_cnt += sub_idx.sum().item()
            
        #     # print("main",main_idx.sum())
        #     # print("sub",sub_idx.sum())
        #     # print()
            
        # m_prompts = self.main_prompts[-1] #* 1,2,3,20,768
        # main_key = self.main_key[-1].unsqueeze(0)
        
        # one,dual,num_layer,p_length,dim = m_prompts.shape
        # m_prompts = m_prompts.unsqueeze(0).expand(x.shape[0],one,dual,num_layer,p_length,dim) #* B,1,dual,Num_layer,prompt_len,dim
        

        # x,_ = self.main_prompt_forward(x,query,m_prompts,main_key=main_key)
        # # self.simmilarity = sim.mean()
        # feats = x[:,0]
        # feats = self.backbone.fc_norm(feats)
        # x = self.backbone.fc_norm(feats)
        # x = self.backbone.fc(x)
        return query,main_idx,sub_idx