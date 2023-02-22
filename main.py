# import os
# import time
# import torch

# from configuration import config
# from datasets import *

# from methods.er_baseline import ER
# from methods.clib import CLIB
# from methods.L2P import L2P
# from methods.our_method import Ours
# # from methods.L2P_rebuild import L2P

# torch.autograd.set_detect_anomaly(True)
# methods = { "er": ER, "clib":CLIB, 'L2P':L2P,'Ours':Ours }
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# import torch
import os
import time
import torch
from configuration import config
from datasets import *
from methods.er_baseline import ER
from methods.clib import CLIB
from methods.L2P import L2P
from methods.rainbow_memory import RM
from methods.Finetuning import FT
from methods.ewc import EWCpp
from methods.lwf import LwF
from methods.ours import Ours
from methods.ours_test import Ours_test
from methods.ours_test import baseline
from methods.ours_total import Ours_total
from methods.dualprompt import DualPrompt

import random
import torch
import numpy as np

# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.enabled = False
methods = { "er": ER, "clib":CLIB, 'L2P':L2P, 'rm':RM, 'Finetuning':FT, 'ewc++':EWCpp,
           'lwf':LwF,
           'ours':Ours, 'ours_test':Ours_test, 'DualPrompt':DualPrompt, 'baseline':baseline,
           'Ours_total':Ours_total}
# torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# import torch.backends.cudnn as cudnn
def main():
    # Get Configurations
    args = config.base_parser()
    print(args)
    rnd_seed = args.rnd_seed
    random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(rnd_seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(rnd_seed)
    torch.cuda.manual_seed_all(rnd_seed) # if use multi-GPU
    
    trainer = methods[args.mode](**vars(args))
    trainer.run()

if __name__ == "__main__":
    main()
    time.sleep(60)
