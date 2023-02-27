from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import logging
import copy
import time
import datetime

import gc
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from methods._trainer import _Trainer

from utils.train_utils import select_optimizer, select_scheduler

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs
from models.vit import _create_vision_transformer


logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

T = TypeVar('T', bound = 'nn.Module')

default_cfgs['vit_base_patch16_224'] = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        num_classes=21843)

# Register the backbone model to timm
@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

class Ours(_Trainer):
    def __init__(self, **kwargs):
        super(Ours, self).__init__(**kwargs)
        
        self.use_mask    = kwargs.get("use_mask")
        self.use_contrastiv  = kwargs.get("use_contrastiv")
        self.use_last_layer  = kwargs.get("use_last_layer")
        
        self.alpha  = kwargs.get("alpha")
        self.gamma  = kwargs.get("gamma")
        self.use_base_CE = kwargs.get("use_base_ce")
        self.use_CP_CE = kwargs.get("use_compensation_ce")
    
    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        del(images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)
        
        self.optimizer.zero_grad()
        logit, loss = self.model_forward(x, y)
        _, preds = logit.topk(self.topk, 1, True, True)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            feature, mask = self.model_without_ddp.forward_features(x)
            logit = self.model_without_ddp.forward_head(feature)
            if self.use_mask:
                logit = logit * mask
            logit = logit + self.mask
            loss = self.loss_fn(feature, mask, y)
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

                logit = self.model(x)
                logit = logit + self.mask
                loss = F.cross_entropy(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.mean().item()
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
            
    def online_before_task(self, task_id):
        pass

    def online_after_task(self, cur_iter):
        pass

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, True)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def _compute_grads(self, feature, y, mask):
        head = copy.deepcopy(self.model_without_ddp.backbone.fc)
        head.zero_grad()
        logit = head(feature.detach())
        if self.use_mask:
            logit = logit * mask.clone().detach()
        logit = logit + self.mask
        
        sample_loss = F.cross_entropy(logit, y, reduction='none')
        sample_grad = []
        for idx in range(len(y)):
            sample_loss[idx].backward(retain_graph=True)
            _g = head.weight.grad[y[idx]].clone()
            sample_grad.append(_g)
            head.zero_grad()
        sample_grad = torch.stack(sample_grad)    #B,dim
        
        head.zero_grad()
        batch_loss = F.cross_entropy(logit, y, reduction='mean')
        batch_loss.backward(retain_graph=True)
        total_batch_grad = head.weight.grad[:len(self.exposed_classes)].clone()  # C,dim
        idx = torch.arange(len(y))
        batch_grad = total_batch_grad[y[idx]]    #B,dim
        
        return sample_grad, batch_grad
    
    def _get_ignore(self, sample_grad, batch_grad):
        # ign_score = torch.max(1. - torch.cosine_similarity(sample_grad, batch_grad, dim=1), torch.zeros(1, device=self.device)) #B
        ign_score = (1. - torch.cosine_similarity(sample_grad, batch_grad, dim=1)) #B
        return ign_score

    def _get_compensation(self, y, sample_g):
        head_w = self.model_without_ddp.backbone.fc.weight[y].clone().detach()
        # cps_score = torch.max(1 - torch.cosine_similarity(head_w, sample_g, dim=1), torch.ones(1, device=self.device)) # B
        cps_score = (1. - torch.cosine_similarity(head_w, -sample_g, dim=1)) #B
        return cps_score

    def _get_score(self, feat, y, mask):
        sample_grad, batch_grad = self._compute_grads(feat, y, mask)
        ign_score = self._get_ignore(sample_grad, batch_grad)
        cps_score = self._get_compensation(y, sample_grad)
        return ign_score, cps_score
    
    def loss_fn(self, feature, mask, y):
        # logit = self.model_without_ddp.forward_head(feature)
        # logit = logit * mask
        # logit = logit + self.mask
        # mask_loss = F.cross_entropy(logit, y)

        ign_score, cps_score = self._get_score(feature, y, mask)
        logit = self.model_without_ddp.forward_head(feature * (1 + cps_score.unsqueeze(1)**self.alpha))
        logit = logit * mask
        logit = logit + self.mask
        loss = F.cross_entropy(logit, y, reduction='none')
        loss = (1 + ign_score ** self.gamma) * loss
        return loss.mean() + self.model_without_ddp.get_similarity_loss()
    
    def report_training(self, sample_num, train_loss, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))} | "
            f"N_Prompts {self.model_without_ddp.e_prompts.size(0)} | "
            f"N_Exposed {len(self.exposed_classes)} | "
            f"Counts {self.model_without_ddp.count.to(torch.int64).tolist()}"
        )

    def setup_distributed_model(self):
        super().setup_distributed_model()
        self.model_without_ddp.use_mask = self.use_mask
        self.model_without_ddp.use_contrastiv = self.use_contrastiv
        self.model_without_ddp.use_last_layer = self.use_last_layer
