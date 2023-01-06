import logging.config
import os
import random
import pickle
from collections import defaultdict

import numpy as np
import torch
from randaugment import RandAugment
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms

from configuration import config
from utils.onlinesampler import OnlineSampler, OnlineTestSampler
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method
from datasets import *

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

datasets = {
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "svhn": SVHN,
        "fashionmnist": FashionMNIST,
        "mnist": MNIST,
        "tinyimagenet": TinyImageNet,
        "notmnist": NotMNIST,
        "cub200": CUB200,
        "imagenet": ImageNet
    }

def main():
    # Get Configurations
    args = config.base_parser()
    
    # Set the logger
    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"{args.log_path}/logs/{args.dataset}/{args.note}", exist_ok=True)
    os.makedirs(f"{args.log_path}/tensorboard/{args.dataset}/{args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'{args.log_path}/logs/{args.dataset}/{args.note}/seed_{args.rnd_seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    writer = SummaryWriter(f'{args.log_path}/tensorboard/{args.dataset}/{args.note}/seed_{args.rnd_seed}')

    logger.info(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("Augmentation on GPU not available!")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # Transform Definition
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)
    if args.model_name == 'vit':
        inp_size = 224
    train_transform = []
    if "cutout" in args.transforms:
        train_transform.append(Cutout(size=16))
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("cutout not supported on GPU!")
    if "randaug" in args.transforms:
        train_transform.append(RandAugment())
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("randaug not supported on GPU!")
    if "autoaug" in args.transforms:
        if 'cifar' in args.dataset:
            train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
        elif 'imagenet' in args.dataset:
            train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
        elif 'svhn' in args.dataset:
            train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('svhn')))
            
    train_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            *train_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    logger.info(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    logger.info(f"[1] Select a CIL method ({args.mode})")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    method = select_method(
        args, criterion, device, train_transform, test_transform, n_classes
    )

    logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
    task_records = defaultdict(list)
    eval_results = defaultdict(list)
    samples_cnt = 0

    test_dataset = get_test_datalist(args.dataset)
    train_dataset   = datasets[args.dataset](root=args.data_dir, train=True,  download=True, 
                                             transform=transforms.ToTensor())
    test_dataset    = datasets[args.dataset](root=args.data_dir, train=False, download=True, transform=test_transform)
    train_sampler   = OnlineSampler(train_dataset, args.n_tasks, args.m, args.n, args.rnd_seed, 0)

    num_eval = args.eval_period
    features = []
    for cur_iter in range(args.n_tasks):
        if args.mode == "joint" and cur_iter > 0:
            return
        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        logger.info("[2-1] Prepare a datalist for the current task")

        train_sampler.set_task(cur_iter)
        train_dataloader= DataLoader(train_dataset, batch_size=args.batchsize, sampler=train_sampler, num_workers=4)

        # Reduce datalist in Debug mode
        # if args.debug:
        #     train_dataloader = train_dataloader[:2000]
        #     test_dataloader  = test_dataloader[:2000]
        method.online_before_task(cur_iter)
        for i, data in enumerate(train_dataloader):
            if args.debug and i == 2000 : break
            samples_cnt += args.batchsize
            method.online_step(data, samples_cnt, args.n_worker)
            if samples_cnt > num_eval:
            # if samples_cnt % args.eval_period == 0:
                num_eval += args.eval_period
                test_sampler = OnlineTestSampler(test_dataset, method.exposed_classes)
                test_dataloader = DataLoader(test_dataset, batch_size=512, sampler=test_sampler, num_workers=4)
                eval_dict = method.online_evaluate(test_dataloader, samples_cnt)
                eval_results["test_acc"].append(eval_dict['avg_acc'])
                eval_results["avg_acc"].append(eval_dict['cls_acc'])
                eval_results["data_cnt"].append(samples_cnt)
        method.online_after_task(cur_iter)
        
        test_sampler = OnlineTestSampler(test_dataset, method.exposed_classes)
        test_dataloader = DataLoader(test_dataset, batch_size=512, sampler=test_sampler, num_workers=4)
        if args.mode == "ViT":
            eval_dict = method.evaluation_with_feature(test_dataloader, samples_cnt)
        else:
            eval_dict = method.evaluation(test_dataloader, samples_cnt)
        task_acc = eval_dict['avg_acc']

        if args.mode == "ViT":
            features.append(eval_dict['embedding'])

        logger.info("[2-4] Update the information for the current task")
        task_records["task_acc"].append(task_acc)
        task_records["cls_acc"].append(eval_dict["cls_acc"])

        logger.info("[2-5] Report task result")
        writer.add_scalar("Metrics/TaskAcc", task_acc, cur_iter)

    if args.mode == "ViT":
        # Tsne visualization feature
        for class_idx in range(n_classes):
            class_feature = []
            for i in range(args.n_tasks):
                class_feature.append(features[i][class_idx])
            class_feature = np.concatenate(class_feature, axis=0)
            X_2d = TSNE(n_components=2).fit_transform(class_feature)
            plt.figure(figsize=(10, 10))
            for i in range(args.n_tasks):
                plt.scatter(X_2d[i*1000:(i+1)*1000, 0], X_2d[i*1000:(i+1)*1000, 1])
            plt.savefig(f'{args.log_path}/logs/{args.dataset}/{args.note}/seed_{args.rnd_seed}_tsne_{class_idx}.png')

    np.save(f"{args.log_path}/logs/{args.dataset}/{args.note}/seed_{args.rnd_seed}.npy", task_records["task_acc"])

    if args.mode == 'gdumb':
        eval_results, task_records = method.evaluate_all(test_dataset, args.memory_epoch, args.batchsize, args.n_worker)
    if args.eval_period is not None:
        np.save(f'{args.log_path}/logs/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval.npy', eval_results['test_acc'])
        np.save(f'{args.log_path}/logs/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval_time.npy', eval_results['data_cnt'])

    # Accuracy (A)
    A_auc = np.mean(eval_results["test_acc"])
    A_avg = np.mean(task_records["task_acc"])
    A_last = task_records["task_acc"][args.n_tasks - 1]

    # Forgetting (F)
    cls_acc = np.array(task_records["cls_acc"])
    acc_diff = []
    for j in range(n_classes):
        if np.max(cls_acc[:-1, j]) > 0:
            acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
    F_last = np.mean(acc_diff)

    logger.info(f"======== Summary =======")
    logger.info(f"A_auc {A_auc} | A_avg {A_avg} | A_last {A_last} | F_last {F_last}")


if __name__ == "__main__":
    main()
