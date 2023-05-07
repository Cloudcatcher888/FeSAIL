from typing import DefaultDict
import numpy as np
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss


import os
import random

class Config():
    batch_win_size = 10
    num_categorial_feat = 26
	task_num = 20
	batch_size = 128
	EMD_DIM = 16
	HID_DIM = 128
	weight_decay = 1e-3
	lr = 1e-3
	epoch_num = 20
	PATIENCE = 2
	if_random_sample = 0
	old_sample_rate = 0
	test_on_which = 2 #0 for old, 1 for all, 2 for unold
	equal_sample = 0 #1 for equal sample numbers for unsample and random/ours


def cuda_setting():
	memory_needed = 2000
	device = torch.device('cpu')
	import pynvml
	pynvml.nvmlInit()
	if torch.cuda.is_available():
	    for i in range(4): 
	        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
	        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
	        free_mem = float(meminfo.free/1024**2)
	        print(free_mem)
	        if free_mem > memory_needed:
 	           device = torch.device(f'cuda:{i}')
	            break
	# device = torch.device('cpu')
	print(f'use {device}')
	random.seed(1) 
	np.random.seed(1) 
	torch.manual_seed(1) 
	torch.cuda.manual_seed(1) 
	torch.cuda.manual_seed_all(1)
    