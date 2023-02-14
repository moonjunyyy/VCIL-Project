import torch_optimizer
from easydict import EasyDict as edict
from torch import optim

from models import mnist, cifar, imagenet
import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, _create_vision_transformer, default_cfgs
from timm.models import create_model
from models.vit import _create_vision_transformer
from models.L2P import L2P
from models.dualprompt import DualPrompt
from models.ours import Ours

default_cfgs['vit_base_patch16_224'] = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        num_classes=21843)
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

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

def select_optimizer(opt_name, lr, model):

    if opt_name == "adam":
        # print("opt_name: adam")
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif opt_name == "radam":
        opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    return opt

# def select_optimizer_with_extern_params(opt_name, lr, model, extern_param=None):

    # if opt_name == "adam":
    #     params = [param for name, param in extern_param.named_parameters() if 'fc.' not in name]
    #     opt = optim.Adam(params, lr=lr, weight_decay=0)
    #     opt.add_param_group({'params': model.fc.parameters()})
    # elif opt_name == "radam":
    #     params = [param for name, param in extern_param.named_parameters() if 'fc.' not in name]
    #     opt = torch_optimizer.RAdam(params, lr=lr, weight_decay=0.00001)
    #     opt.add_param_group({'params': model.fc.parameters()})
    # elif opt_name == "sgd":
    #     params = [param for name, param in extern_param.named_parameters() if 'fc.' not in name]
    #     opt = optim.SGD(
    #         params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
    #     )
    #     opt.add_param_group({'params': model.fc.parameters()})
    # else:
    #     raise NotImplementedError("Please select the opt_name [adam, sgd]")
    # return opt

def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2)
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80, 90], gamma=0.1)
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler

# def select_model(model_name, dataset, num_classes=None):

#     opt = edict(
#         {
#             "depth": 18,
#             "num_classes": num_classes,
#             "in_channels": 3,
#             "bn": True,
#             "normtype": "BatchNorm",
#             "activetype": "ReLU",
#             "pooltype": "MaxPool2d",
#             "preact": False,
#             "affine_bn": True,
#             "bn_eps": 1e-6,
#             "compression": 0.5,
#         }
#     )

# #! cifar and imageNet --> ViT model 추가!!
#     if "mnist" in dataset:
#         model_class = getattr(mnist, "MLP")
#     elif "cifar" in dataset:
#         model_class = getattr(cifar, "ResNet")
#     elif "imagenet" in dataset:
#         model_class = getattr(imagenet, "ResNet")
#     elif "vit" in dataset:
#         pass
#     else:
#         raise NotImplementedError(
#             "Please select the appropriate datasets (mnist, cifar10, cifar100, imagenet)"
#         )

#     #* vit method(L2p) --> cifar_vit,vision_transformer

#     if model_name == "resnet18":
#         opt["depth"] = 18
#     elif model_name == "resnet32":
#         opt["depth"] = 32
#     elif model_name == "resnet34":
#         opt["depth"] = 34
#     elif model_name == "mlp400":
#         opt["width"] = 400
#     elif model_name == "vit":
#         opt["depth"] = 12
#     elif model_name == "L2P":
#         opt["depth"] = 12
#     else:
#         raise NotImplementedError(
#             "Please choose the model name in [resnet18, resnet32, resnet34]"
#         )

#     if model_name == "vit":
#         model = timm.create_model(
#                             "vit_base_patch16_224",pretrained=True,num_classes=num_classes,
#                             drop_rate=0.,drop_path_rate=0.,drop_block_rate=None,)
#     elif model_name == "L2P":
#         model = L2P(backbone_name="vit_base_patch16_224", class_num=num_classes)
#     elif model_name == "resnet18":
#         model = timm.create_model('resnet18', num_classes=num_classes)
#     elif model_name == "resnet32":
#         model = timm.create_model('resnet32', num_classes=num_classes)
#     elif model_name == "resnet34":
#         model = timm.create_model('resnet34', num_classes=num_classes)
#     else:
#         model = model_class(opt)

#     print("[Selected Model]:", model_name )
#     return model
def select_model(model_name, dataset, num_classes=None,selection_size=None):
    
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
        }
    )

#! cifar and imageNet --> ViT model 추가!!
    if "mnist" in dataset:
        model_class = getattr(mnist, "MLP")
    elif "cifar" in dataset:
        model_class = getattr(cifar, "ResNet")
    elif "imagenet" in dataset:
        model_class = getattr(imagenet, "ResNet")
    elif "vit" in dataset:
        pass
    else:
        raise NotImplementedError(
            "Please select the appropriate datasets (mnist, cifar10, cifar100, imagenet)"
        )

    #* vit method(L2p) --> cifar_vit,vision_transformer

    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet34":
        opt["depth"] = 34
    elif model_name == "mlp400":
        opt["width"] = 400
    elif model_name == "vit":
        opt["depth"] = 12
    elif model_name == "L2P":
        opt["depth"] = 12
    elif model_name == "DualPrompt":
        opt["depth"] = 12
    elif model_name == "ours":
        opt["depth"] = 12
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    if model_name == "vit":
        model = timm.create_model(
                            "vit_base_patch16_224",pretrained=True,num_classes=num_classes,
                            drop_rate=0.,drop_path_rate=0.,drop_block_rate=None,)
    elif model_name == "L2P":
        model = L2P(backbone_name="vit_base_patch16_224", class_num=num_classes)
    elif model_name == "DualPrompt":
        model = DualPrompt(backbone_name="vit_base_patch16_224", class_num=num_classes)
    elif model_name == "ours":
        model = Ours(backbone_name="vit_base_patch16_224", class_num=num_classes, selection_size = selection_size)
    # elif model_name == "resnet18":
    #     model = timm.create_model('resnet18', num_classes=num_classes)
    # elif model_name == "resnet32":
    #     model = timm.create_model('resnet32', num_classes=num_classes)
    # elif model_name == "resnet34":
    #     model = timm.create_model('resnet34', num_classes=num_classes)
    else:
        model = model_class(opt)

    print("[Selected Model]:", model_name )
    return model