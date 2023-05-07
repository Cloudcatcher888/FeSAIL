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

from utils import Config

import wandb


import collections





import math
import scipy.stats
from torch.optim import Adagrad
print(f'num_feat:{num_categorial_feat}')
examplar_size = data_num//task_num
criterion = nn.BCELoss()
wconfig = {
    "weight_decay":weight_decay,
    "batch_win_size":Config.batch_win_size,
    "lr":lr,
    "pretrain_epoch":epoch_num,
    "finetune_epoch":epoch_num,
    "patience":PATIENCE,
    "task_num":task_num,
    "batch_size":batch_size,
    "hidden_dim":HID_DIM,
    "device":device
}
wandb.init(project="old_sampler", entity="ludlethwang",config=wconfig,name=f"icg")


#grid search
for ifrandom, oldrate in [(0,2)]:
    model = BaseModel(sizes).to(device)
    embeds = [eb.weight.detach() for eb in model.embeddings]
    print(f'model param nums: {sum([param.numel() for param in model.parameters()])}')
    print(f"batch_size:{batch_size}")
    optimizer = Adagrad(
        model.parameters(),
        lr=lr, 
        initial_accumulator_value=1e-8,
        weight_decay=weight_decay
    )
    examplar_features, examplar_labels = [f[:int(data_num/task_num*Config.batch_win_size)] for f in categorial_features], labels[:int(data_num/task_num*Config.batch_win_size)]     
    guards = [torch.zeros([sizes[_]+1],dtype=torch.long) for _ in range(num_categorial_feat)]
    overall_auc = []
    overall_logloss = []
    overall_auc_old = []
    overall_logloss_old = []
    overall_auc_notold = []
    overall_logloss_notold = []
    for task_idx in range(Config.batch_win_size, task_num-1):
        global_auc = 0.0
        global_logloss = 10.0
        global_auc_old = 0.0
        global_logloss_old = 10.0
        global_auc_notold = 0.0
        global_logloss_notold = 10.0
        global_patience = PATIENCE
        patience = global_patience  
        print(f'running task {task_idx}')
        dataloader,test_loader,guards,examplar_features,examplar_labels = olddata_sampler(
            task_idx,ifrandom,oldrate,1,examplar_features,examplar_labels,guards
            )
        _,test_loader_old,__,___,____ = olddata_sampler(task_idx,ifrandom,oldrate,0,None,None,None)
        _,test_loader_notold,__,___,____ = olddata_sampler(task_idx,ifrandom,oldrate,2,None,None,None)
        for epoch in range(epoch_num):
            model.train()
            total_loss = []
            for features, batchlabel in dataloader:
                features = features.to(device)
                batchlabel = batchlabel.to(device)
                prediction =model(features)
                if if_guard:
                    last_embeds = embeds
                    embeds = [eb.weight.detach().clone() for eb in model.embeddings]
                    reg_l = model.reg(guards,last_embeds)
                    # print(reg_l)
                else:
                    reg_l = 0
                loss = criterion(prediction.view(-1),batchlabel)+reg_l
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
            train_loss = sum(total_loss)/len(total_loss)
            if (epoch+1)%2==0:
                print(epoch,"train loss:",train_loss)
                with torch.no_grad():
                    model.eval()
                    auc,logloss = evaluate(task_idx+1,model)
                    print("evaluation auc,logloss on", task_idx+1,"is",auc,logloss)
                    auc_old,logloss_old = evaluate_spec(test_loader_old,model)
                    print("evaluation auc,logloss on", task_idx+1,"is",auc_old,logloss_old)
                    auc_notold,logloss_notold = evaluate_spec(test_loader_notold,model)
                    print("evaluation auc,logloss on", task_idx+1,"is",auc_notold,logloss_notold)
                    if auc_old<global_auc_old or logloss_old>global_logloss_old:
                        if not patience:
                            overall_auc.append(global_auc)
                            overall_logloss.append(global_logloss)
                            overall_auc_old.append(global_auc_old)
                            overall_logloss_old.append(global_logloss_old)
                            overall_auc_notold.append(global_auc_notold)
                            overall_logloss_notold.append(global_logloss_notold)
                            break
                        patience-=1
                    else:
                        patience = global_patience
                        global_auc, global_logloss = auc, logloss
                        global_auc_old, global_logloss_old = auc_old, logloss_old
                        global_auc_notold, global_logloss_notold = auc_notold, logloss_notold
            
                    if epoch == epoch_num-1:
                        overall_auc.append(global_auc)
                        overall_logloss.append(global_logloss)
                        overall_auc_old.append(global_auc_old)
                        overall_logloss_old.append(global_logloss_old)
                        overall_auc_notold.append(global_auc_notold)
                        overall_logloss_notold.append(global_logloss_notold)
        wandb.log({"train loss":train_loss})
        wandb.log({"auc":global_auc})
        wandb.log({"logloss":global_logloss})

    wandb.log({"train loss":train_loss})
    wandb.log({"auc":sum(overall_auc)/len(overall_auc)})
    wandb.log({"logloss":sum(overall_logloss)/len(overall_logloss)})             
        
        
        