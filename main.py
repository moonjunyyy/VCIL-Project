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
# from utils.method_manager import select_method
from datasets import *

import time

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from methods.er_baseline import ER
from methods.clib import CLIB
from methods.L2P import L2P

methods = { "er": ER, "clib":CLIB, 'L2P':L2P }
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
def main():
    # Get Configurations
    args = config.base_parser()
    print(args)
    trainer = methods[args.mode](**vars(args))
    trainer.run()

if __name__ == "__main__":
    main()
    time.sleep(60)
