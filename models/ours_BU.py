
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
                 prompt_len     : int   = 5,    #* prompt len for multu layer
                 n_task = 5,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 lambd          : float = 0.2,
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
        # self.simmilarity=0.
        # self.num_heads=12
        # self.device='cpu'
        self.add_module('backbone', timm.models.create_model(backbone_name, pretrained=True, num_classes=class_num,
                                                             drop_rate=0.,drop_path_rate=0.,drop_block_rate=None))
        
        self.n_task = n_task
        
        self.sub_key     = nn.Parameter(torch.randn(1, self.backbone.embed_dim))
        self.sub_prompt = nn.Parameter(torch.randn(1, self.prompt_len, self.backbone.embed_dim))
        
        self.main_key     = nn.Parameter(torch.randn(self.n_task, self.backbone.embed_dim))
        self.main_prompts = nn.Parameter(torch.randn(self.n_task,1,2, len(self.deep_layer), self.prompt_len, self.backbone.embed_dim))
        
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
        x = self.backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        with torch.no_grad():
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0].clone()
        
        if test:
            num_main = self.task_id +1
            cand_keys = torch.cat([self.main_key[:num_main],self.sub_key],dim=0)
            # print("cand_keys:",cand_keys.shape)
            # print("query:",query.shape)
            m_prompts=[]
            s_prompts=[]
            main_idx = []
            sub_idx = []
            
            
            # a = torch.randn(2,10)
            # b = torch.randn(5,10)
            # norm_a = l2_normalize(a,dim=1)
            # norm_b = l2_normalize(b,dim=1)
            # sim = torch.matmul(a, b.t()) # B, Pool_size
            # print(sim.shape)  #* sample,prompt
            
            query_norm = self.l2_normalize(query,dim=1)
            key_norm = self.l2_normalize(cand_keys,dim=1)
            simmilarity = torch.matmul(query_norm, key_norm.t()) # B, cand_keys
            # simmilarity = F.cosine_similarity(query, cand_keys, dim=1)
            topk = simmilarity.topk(self.selection_size, dim=1)[1]
            
            for sample_i, select in enumerate(topk):
                if select.item()<= self.task_id:
                    main_idx.append(sample_i)
                    m_prompts.append(select)
                else:
                    sub_idx.append(sample_i)
                    s_prompts.append(select)
            
            main_idx = torch.tensor(main_idx)
            m_prompts = torch.tensor(m_prompts)
            sub_idx = torch.tensor(sub_idx)
            s_prompts = torch.tensor(s_prompts)
            # main_idx = torch.tensor(main_idx)
            # main_x = x[main_idx]    #* go to main prompts
            # if len(sub_idx)==0:
                # sub_x = None
            # else:
                # sub_idx = torch.tensor(sub_idx)
                # sub_x = x[sub_idx]      #* go to sub prompts 
            # s_prompts = self.sub_prompt
            
            
            
            #* m_prompts = self.main_prompts[topk].squeeze(1)
            # print("[Test] M_prompts:",m_prompts.shape)
            # m_prompts = self.main_prompts[m_prompts].squeeze(1)
            m_prompts = self.main_prompts[m_prompts]
            # print("[Test] M_prompts:",m_prompts.shape)
            main_key = self.main_key[topk]
            main_x = x[main_idx]
            main_query = query[main_idx]
            
            if sub_idx.shape[0] != 0:
                # m_prompts = self.main_prompts[m_prompts].squeeze(1)
                s_prompts = self.sub_prompt
                # sub_key = self.main_key[topk]
                # sub_query = query[sub_idx]
                sub_x = x[sub_idx]
            else:
                sub_x = None
            
        else:   #* train
            #* split main and sub class sample
            main_idx,sub_idx = self.split_samples(query)
            self.main_cnt += main_idx.sum().item()
            self.sub_cnt += sub_idx.sum().item()
            
            # print("main",main_idx.sum())
            # print("sub",sub_idx.sum())
            
            #todo sample마다 각각의 prompt를 사용하여 학습하기!!
            #todo sub_prompt_forward / main_prompt_forward implementation
            main_x = x[main_idx]    #* go to main prompts
            sub_x = x[sub_idx]      #* go to sub prompts 
            
            s_prompts = self.sub_prompt
            # s_key = self.sub_key
            m_prompts = self.main_prompts[self.task_id]
            main_key = self.main_key[self.task_id].unsqueeze(0)
            # print("main_key[-1]:",main_key.shape)
            
            
            prompt_per_sample = torch.full((main_x.shape[0],1),0)
            m_prompts = m_prompts[prompt_per_sample] #* B,1,dual,Num_layer,prompt_len,dim
            # main_key = self.main_key[-1].unsqueeze(0).expand(main_x.shape[0],-1)
            # print("[Train] M_prompts:",m_prompts.shape)   
            # print("Single_m_prompt:",m_prompts.shape)
            
            # print("m_prompt_per_sample:",m_prompts.shape)
            # print('-'*30)
            main_query = query[main_idx]
        
        

        main_x,main_sim = self.main_prompt_forward(main_x,main_query,m_prompts,main_key=main_key)
        # main_x = main_x[:, 1:self.selection_size * self.prompt_len + 1].clone()
        main_feats = main_x[:, 0]
        # main_feats = main_x.mean(dim=1).squeeze()
        main_x = self.backbone.fc_norm(main_feats)
        main_x = self.backbone.fc(main_x)
        
        if sub_x != None and sub_x.shape[0] != 0:
            sub_x,sub_sim = self.sub_prompt_forward(sub_x,query[sub_idx],s_prompts)
            sub_x = sub_x[:, 1:self.selection_size * self.prompt_len + 1].clone()
            sub_feats = sub_x.mean(dim=1).squeeze()
            sub_x = self.backbone.fc_norm(sub_feats)
            sub_x = self.backbone.fc(sub_x)
            self.simmilarity = main_sim.mean() + sub_sim.mean()
        else:
            sub_x = None
            sub_feats=None
            self.simmilarity = main_sim.mean()
        
        x_dict = {'main':main_x, 'sub':sub_x}
        feat_dict = {'main':main_feats, 'sub':sub_feats}
        return x_dict, feat_dict, main_idx, sub_idx #* logit,feats,main_idx,sub_idx
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
    

    def expand_prompt(self,task_id):
        if task_id==0:
            pass
        else:
            
            tmp_prompt_data = copy.deepcopy(self.sub_prompt.data)
            tmp_key_data = copy.deepcopy(self.sub_key.data)
            # s = s[:,None,None,None,:,:]
            tmp_prompt_data = tmp_prompt_data[:,None,None,None,:,:].squeeze(0)
            # s = s.expand(m[0].shape); print()
            tmp_prompt_data = tmp_prompt_data.expand(self.main_prompts[task_id].data)
            self.main_key[task_id].data = tmp_key_data
            self.main_prompts[task_id].data = tmp_prompt_data
            
            # for p in self.main_prompt.parameters():
            #     p.requires_grad = False
            # for p in self.main_key.parameters():
            #     p.requires_grad = False
            
            # for p in self.main_prompt[task_id].parameters():
            #     p.requires_grad = True
            # for p in self.main_key[task_id].parameters():
            #     p.requires_grad = True
            #? =============================================
        
        # if task_id>0:
        #     tmp_prompts = self.main_prompts.cpu().clone().detach()
        #     tmp_key = self.main_key.cpu().clone().detach()
            
        #     sub_prompt = self.sub_prompt.clone()
        #     sub_key = self.sub_key.clone()
        #     prompt_pool_shape = (len(self.deep_layer), 2, task_id+1, self.prompt_len, 
        #                                 self.num_heads, self.backbone.embed_dim // self.num_heads)
            
        #     # self.sub_key     = nn.Parameter(torch.randn(1, self.backbone.embed_dim))
        #     # self.sub_prompt = nn.Parameter(torch.randn(1, self.prompt_len, self.backbone.embed_dim))
            
        #     self.main_key = nn.Parameter(torch.randn(task_id+1, self.backbone.embed_dim,device=self.device))
        #     self.main_prompts = nn.Parameter(torch.randn(task_id+1, self.prompt_len, self.backbone.embed_dim,device=self.device))
            
        #     self.main_key[:task_id].data = tmp_key.data
        #     self.main_prompts[:task_id].data = tmp_prompts.data
            
        # else:
        #     prompt_pool_shape = (len(self.deep_layer), 2, 1, self.prompt_len, 
        #                                 self.num_heads, self.backbone.embed_dim // self.num_heads)

        #     self.main_key = nn.Parameter(torch.randn(task_id+1, self.backbone.embed_dim,device=self.device))
        #     self.main_prompts = nn.Parameter(torch.randn(task_id+1, self.prompt_len, self.backbone.embed_dim,device= self.device))
        
        # self.main_key = self.main_key.to(self.device)
        # self.main_prompts = self.main_prompts.to(self.device)
            

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
    
    # def main_prompt_forward(self,main_x,main_query,prompts):
    #     #* prompt attach to input tokens
    #     B,N,D = main_x.size()
    #     simmilarity = F.cosine_similarity(main_query.unsqueeze(1), self.sub_key, dim=-1)

    #     # print("sub_prompt:",prompts.shape)  #* 1,5,768 --> pool_size, length,dim
        
    #     prompts = prompts.contiguous().expand(B, self.selection_size * self.prompt_len, D)
    #     prompts = prompts + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
    #     # print("sub_prompt:",prompts.shape)
        
    #     main_x = torch.cat((main_x[:,0].unsqueeze(1), prompts, main_x[:,1:]), dim=1)
    #     main_x = self.backbone.blocks(main_x)
    #     main_x = self.backbone.norm(main_x)
        
    #     return main_x,simmilarity
    
    def main_prompt_forward(self,main_x,main_query,prompts,main_key):
        #* prompt attach to MSA
        B,N,D = main_x.size()
        # print("[Mprompt_forward] prompts:",prompts.shape)   #* main_X_num, 1,2,3,5,768 -> (main_X_num,1,2,num_layer,prompt_len,768)
        # simmilarity = F.cosine_similarity(main_query.unsqueeze(1), self.main_key, dim=-1)
        # print("main_query",main_query.shape)
        # print("main_key",main_key.shape)
        simmilarity = F.cosine_similarity(main_query, main_key, dim=1)
        # print("similar",simmilarity.shape)
        
        batch_size, one,dual,num_layer, length, embed_dim = prompts.shape
        # prompts = prompts.reshape(num_layer,batch_size,dual,topk*length,num_heads,heads_embed)
        prompts = prompts.contiguous().view(batch_size,dual*num_layer,length,embed_dim)
        
        #todo MSA attach
        for idx, block in enumerate(self.backbone.blocks):
            if idx in self.deep_layer:
                #todo self.pre_attn(self.backbone.norm1(x), prompts)
                #! 수정 시작하자!
                
                norm_x = block.norm1(main_x)
                # xk = xq.clone()
                # xv = xq.clone()
                #?---------------------------------------
                #? MSA Layer forward start
                B, N, C = norm_x.shape
                # qkv = block.attn.qkv(main_x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                qkv = block.attn.qkv(norm_x).reshape(B,N,3,-1)
                # print("qkv:",qkv.shape)
                #* q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
                q, k, v = qkv.unbind(2)
                # print("q:",q.shape)
                # print("k:",k.shape)
                # print("v:",v.shape)

                #? if prompt is not None:
                    # prefix key, value
                # prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
                # key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
                # value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads

                # expected_shape = (B, self.num_heads, C // self.num_heads)
                
                # assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
                # assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

                k = torch.cat([prompts[:,self.deep_layer.index(idx)*2+0], k], dim=1)
                v = torch.cat([prompts[:,self.deep_layer.index(idx)*2+1], v], dim=1)
                # print("prefix_k",k.shape)
                # print("prefix_v",v.shape)
                B,N,C = q.shape

                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1)
                attn = block.attn.attn_drop(attn)

                attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
                attn = block.attn.proj(attn)
                attn = block.attn.proj_drop(attn)
                # print('attn:',attn.shape)
                # print('main_x',main_x.shape)
                #?---------------------------------------
                #? MSA Layer forward done
                
                main_x = main_x + block.drop_path1(block.ls1(attn))
                main_x = main_x + block.drop_path2(block.ls2(block.mlp(block.norm2(main_x))))
            else:
                main_x = block(main_x)
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
        return F.cross_entropy(output, target) + self.lambd * (1-self.simmilarity)

    def set_device(self,device):
        self.device = device
    def set_info(self,task_id):
        #* Task Id is needed to control prompt scope
        #* we already make whold prompt
        #* if you expand prompts whenever new task are incomed, you don't need this one
        self.task_id = task_id







#!=======================================================================================================================================
# from typing import TypeVar, Iterable
# import timm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import logging
# from torch.utils.tensorboard import SummaryWriter
# import timm
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg, default_cfgs

# from models.vit import _create_vision_transformer

# logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")

# T = TypeVar('T', bound = 'nn.Module')

# default_cfgs['vit_base_patch16_224_l2p'] = _cfg(
#         url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
#         num_classes=21843)

# # Register the backbone model to timm
# @register_model
# def vit_base_patch16_224_l2p(pretrained=False, **kwargs):
#     """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
#     """
#     model_kwargs = dict(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224_l2p', pretrained=pretrained, **model_kwargs)
#     return model

# class Prompt(nn.Module):
#     def __init__(self,
#                  pool_size            : int,
#                  selection_size       : int,
#                  prompt_len           : int,
#                  dimention            : int,
#                  _diversed_selection  : bool = False,
#                  _batchwise_selection : bool = False,
#                  **kwargs):
#         super().__init__()

#         self.pool_size      = pool_size
#         self.selection_size = selection_size
#         self.prompt_len     = prompt_len
#         self.dimention      = dimention
#         self._diversed_selection  = _diversed_selection
#         self._batchwise_selection = _batchwise_selection

#         self.key     = nn.Parameter(torch.randn(pool_size, dimention, requires_grad= True))
#         self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimention, requires_grad= True))
        
#         torch.nn.init.uniform_(self.key,     -1, 1)
#         torch.nn.init.uniform_(self.prompts, -1, 1)

#         self.register_buffer('frequency', torch.ones (pool_size))
#         self.register_buffer('counter',   torch.zeros(pool_size))
    
#     def forward(self, query : torch.Tensor, **kwargs):

#         B, D = query.shape
#         assert D == self.dimention, f'Query dimention {D} does not match prompt dimention {self.dimention}'
#         # Select prompts
#         match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
#         if self.training and self._diversed_selection:
#             topk = match * F.normalize(self.frequency, p=1, dim=-1)
#         else:
#             topk = match
#         _ ,topk = topk.topk(self.selection_size, dim=-1, largest=False, sorted=True)
#         # Batch-wise prompt selection
#         if self._batchwise_selection:
#             idx, counts = topk.unique(sorted=True, return_counts=True)
#             _,  mosts  = counts.topk(self.selection_size, largest=True, sorted=True)
#             topk = idx[mosts].clone().expand(B, -1)
#         # Frequency counter
#         self.counter += torch.bincount(topk.reshape(-1).clone(), minlength = self.pool_size)
#         # selected prompts
#         selection = self.prompts.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len, self.dimention).clone())
#         simmilarity = match.gather(1, topk)
#         # get unsimilar prompts also 
#         return simmilarity, selection

#     def update(self):
#         if self.training:
#             self.frequency += self.counter
#         counter = self.counter.clone()
#         self.counter *= 0
#         if self.training:
#             return self.frequency - 1
#         else:
#             return counter

# class DualPrompt(nn.Module):
#     def __init__(self,
#                  pos_g_prompt   : Iterable[int] = (0, 1),
#                  len_g_prompt   : int   = 5,
#                  pos_e_prompt   : Iterable[int] = (2, 3, 4),
#                  len_e_prompt   : int   = 20,
#                  prompt_func    : str   = None,
#                  task_num       : int   = 10,
#                  class_num      : int   = 100,
#                  lambd          : float = 1.0,
#                  backbone_name  : str   = None,
#                  **kwargs):
#         super().__init__()

#         if backbone_name is None:
#             raise ValueError('backbone_name must be specified')

#         self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype=torch.int64))
#         self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
#         self.register_buffer('similarity', torch.zeros(1))
#         self.register_buffer('mask', torch.zeros(class_num))
        
#         self.lambd      = lambd
#         self.class_num  = class_num

#         self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
#         for name, param in self.backbone.named_parameters():
#             param.requires_grad = False
#         self.backbone.head.weight.requires_grad = True
#         self.backbone.head.bias.requires_grad   = True

#         self.tasks = []

#         self.len_g_prompt = len_g_prompt
#         self.len_e_prompt = len_e_prompt
#         g_pool = 1
#         e_pool = task_num
#         self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
#         self.e_length = len(pos_e_prompt) if pos_e_prompt else 0
        
#         if prompt_func == 'prompt_tuning':
#             self.prompt_func = self.prompt_tuning
#             self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(g_pool, 1, self.g_length * self.len_g_prompt, self.backbone.num_features, batchwise_selection = False)
#             self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(e_pool, 1, self.e_length * self.len_e_prompt, self.backbone.num_features, batchwise_selection = False)

#         elif prompt_func == 'prefix_tuning':
#             self.prompt_func = self.prefix_tuning
#             self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(g_pool, 1, 2 * self.g_length * self.len_g_prompt, self.backbone.num_features, batchwise_selection = False)
#             self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(e_pool, 1, 2 * self.e_length * self.len_e_prompt, self.backbone.num_features, batchwise_selection = False)

#         else: raise ValueError('Unknown prompt_func: {}'.format(prompt_func))
        
#         self.task_num = task_num
#         self.task_id = -1 # if _convert_train_task is not called, task will undefined

#     def prompt_tuning(self,
#                       x        : torch.Tensor,
#                       g_prompt : torch.Tensor,
#                       e_prompt : torch.Tensor,
#                       **kwargs):

#         B, N, C = x.size()
#         g_prompt = g_prompt.contiguous().view(B, self.g_length, self.len_g_prompt, C)
#         e_prompt = e_prompt.contiguous().view(B, self.e_length, self.len_e_prompt, C)
#         g_prompt = g_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.g_length, self.len_g_prompt, C)
#         e_prompt = e_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.e_length, self.len_e_prompt, C)

#         for n, block in enumerate(self.backbone.blocks):
#             pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
#             if pos_g.numel() != 0:
#                 x = torch.cat((x, g_prompt[:, pos_g].unsqueeze(0).expand(B,-1,-1)), dim = 1)

#             pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
#             if pos_e.numel() != 0:
#                 x = torch.cat((x, e_prompt[:, pos_e].unsqueeze(0).expand(B,-1,-1)), dim = 1)
#             x = block(x)
#         return x
    
#     def prefix_tuning(self,
#                       x        : torch.Tensor,
#                       g_prompt : torch.Tensor,
#                       e_prompt : torch.Tensor,
#                       **kwargs):

#         B, N, C = x.size()
#         g_prompt = g_prompt.contiguous().view(B, 2 * self.g_length, self.len_g_prompt, C)
#         e_prompt = e_prompt.contiguous().view(B, 2 * self.e_length, self.len_e_prompt, C)
#         # g_prompt = g_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, 2 * self.g_length, self.len_g_prompt, C)
#         # e_prompt = e_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, 2 * self.e_length, self.len_e_prompt, C)

#         for n, block in enumerate(self.backbone.blocks):

#             xq = block.norm1(x)
#             xk = xq.clone()
#             xv = xq.clone()

#             pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
#             if pos_g.numel() != 0:
#                 xk = torch.cat((xk, g_prompt[:, pos_g * 2 + 0]), dim = 1)
#                 xv = torch.cat((xv, g_prompt[:, pos_g * 2 + 1]), dim = 1)

#             pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
#             if pos_e.numel() != 0:
#                 xk = torch.cat((xk, e_prompt[:, pos_e * 2 + 0]), dim = 1)
#                 xv = torch.cat((xv, e_prompt[:, pos_e * 2 + 1]), dim = 1)
            
#             attn   = block.attn
#             weight = attn.qkv.weight
#             bias   = attn.qkv.bias
            
#             B, N, C = xq.shape
#             xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
#             _B, _N, _C = xk.shape
#             xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
#             _B, _N, _C = xv.shape
#             xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

#             attention = (xq @ xk.transpose(-2, -1)) * attn.scale
#             attention = attention.softmax(dim=-1)
#             attention = attn.attn_drop(attention)

#             attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
#             attention = attn.proj(attention)
#             attention = attn.proj_drop(attention)

#             x = x + block.drop_path1(block.ls1(attention))
#             x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

#         return x

#     def forward(self, inputs : torch.Tensor) :
        
#         with torch.no_grad():
#             x = self.backbone.patch_embed(inputs)
#             B, N, D = x.size()

#             cls_token = self.backbone.cls_token.expand(B, -1, -1)
#             token_appended = torch.cat((cls_token, x), dim=1)
#             x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
#             query = self.backbone.blocks(x)
#             query = self.backbone.norm(query)[:, 0]

#         if self.g_prompt is not None:
#             g_p = self.g_prompt.prompts[0].expand(B, -1, -1)
#             # g_s, g_p = self.g_prompt(query)
#         else:
#             g_p = None
#         if self.e_prompt is not None:
#             if self.training:
#                 e_s = 1 - F.cosine_similarity(query.unsqueeze(1), self.e_prompt.key[self.task_id], dim = -1)
#                 e_p = self.e_prompt.prompts[self.task_id].expand(B, -1, -1)
#                 self.e_prompt.counter[self.task_id] += B
#             else:
#                 e_s, e_p = self.e_prompt(query)
#         else:
#             e_p = None
#             e_s = 0

#         x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_p, e_p)
#         x = self.backbone.norm(x)
#         x = self.backbone.head(x[:, 0])

#         # if self.training:
#         #     x = x + self.mask

#         self.similarity = e_s.sum()
#         return x

#     def convert_train_task(self, task : torch.Tensor, **kwargs):
        
#         flag = -1
#         for n, t in enumerate(self.tasks):
#             if torch.equal(t, task):
#                 flag = n
#                 break
#         if flag == -1:
#             self.tasks.append(task)
#             self.task_id = len(self.tasks) - 1
#             if self.training:
#                 if self.task_id != 0:
#                     with torch.no_grad():
#                         self.e_prompt.prompts[self.task_id] = self.e_prompt.prompts[self.task_id - 1].detach().clone()
#                         self.e_prompt.key[self.task_id] = self.e_prompt.key[self.task_id - 1].detach().clone()
#         else :
#             self.task_id = flag

#         self.mask += -torch.inf
#         self.mask[task] = 0
#         return
        
#     def get_count(self):
#         return self.e_prompt.update()

#     def loss_fn(self, output, target):
#         B, C = output.size()
#         return F.cross_entropy(output, target) + self.lambd * self.similarity