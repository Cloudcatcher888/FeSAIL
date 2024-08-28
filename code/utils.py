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
	


# definition of dataset
from torch.utils.data import Dataset, DataLoader
class CriteoDataset(Dataset):
    def __init__(self, categorial_features, labels,start, end):
    	print(os.path.abspath('.'))
		path = 	os.path.abspath('/home/wangzhikai/work/playground/recommendation_learning/models/')
		categorial_features = np.load(os.path.join(path,'cate_array_thr10.npy'))
		labels = np.load(os.path.join(path,'label_array_thr10.npy'))
		sizes = np.load(os.path.join(path,'size_array_thr10.npy'))

		categorial_features = categorial_features.tolist()
		for field in categorial_features:
	 	    print(len(set(field)))
		labels = labels.tolist()
		sizes = sizes.tolist()
		data_num = len(labels)
 	   self.categorial_features = [f[start:end] for f in categorial_features]    				
 	   self.labels = labels[start:end]
        
    def __getitem__(self, idx):
        d = torch.tensor([f[idx] for f in self.categorial_features],dtype=torch.int64)
        la = torch.tensor(int(self.labels[idx]),dtype=torch.float)
        return d, la
    
    def __len__(self):
        return len(self.labels)
        
def evaluate(task_idx,model):
    # evaluate the model on $task_idx th task	

    # count = 0
    # tse = 0
    start = int(data_num/task_num*task_idx)
    end = int(data_num/task_num*(task_idx+1))
    dataset = CriteoDataset(categorial_features,labels,start,end)
    print(len(dataset))
    loader =  DataLoader(dataset,shuffle=False,batch_size=batch_size)
    t_label = []
    t_predict = []
    for features, label in loader:
        features = features.to(device)
        # label = label.to(device)
        prediction =model(features)
        # se = (prediction.view(-1)-label)**2
        # count += label.shape[0]
        # tse += se.sum().cpu().item()
        t_label.extend(label.numpy().tolist())
        t_predict.extend(prediction.view(-1).cpu().numpy().tolist())
    auc = roc_auc_score(np.array(t_label),np.array(t_predict))
    logloss = log_loss(np.array(t_label),np.array(t_predict))
    # rmse = math.sqrt(tse / count)

    return auc,logloss

def evaluate_spec(test_loader,model):
    # evaluate on old/not old samples in $task_idx th task
    # count = 0
    # tse = 0
    loader =  test_loader
    t_label = []
    t_predict = []
    for features, label in loader:
        features = features.to(device)
        # label = label.to(device)
        prediction =model(features)
        # se = (prediction.view(-1)-label)**2
        # count += label.shape[0]
        # tse += se.sum().cpu().item()
        t_label.extend(label.numpy().tolist())
        t_predict.extend(prediction.view(-1).cpu().numpy().tolist())
    auc = roc_auc_score(np.array(t_label),np.array(t_predict))
    logloss = log_loss(np.array(t_label),np.array(t_predict))
    # rmse = math.sqrt(tse / count)

    return auc,logloss





def olddata_sampler(task_idx,if_random_sample,old_sample_rate, test_on_which,examplar_features,examplar_labels,guards):
	
    # the definition of FeSAIL
    from copy import copy
    temp_guard = [{id:0 for id in range(sizes[_]+1)} for _ in range(num_categorial_feat)]
    tguard2 = [{id:0 for id in range(sizes[_]+1)} for _ in range(num_categorial_feat)]
    addset = set()
    addset_examplar = set()
    addset_examplar_wcp = set()
    examplar_feat_dicts = []
    coexist_dict = DefaultDict(set)
    addset2 = []
    start = int(data_num/task_num*task_idx)
    end = int(data_num/task_num*(task_idx+1))
    sample_features, sample_label = [f[start:end] for f in categorial_features], labels[start:end]
    sample_features_fix = [_[:] for _ in sample_features]
    nxt_features, nxt_label = [f[end:2*end-start] for f in categorial_features],labels[end:2*end-start]
    nxt_test_features, nxt_test_label = [[] for _ in range(num_categorial_feat)], []
    nxt_idxs = []
    history_features, history_label = [f[:end] for f in categorial_features], labels[:end]
    for i in range(num_categorial_feat):
        # print(len(sample_features[i]))
        # print(len(sample_features_fix[i]))
        feat_set = set(sample_features[i])        
        feat_set_fix = set(sample_features_fix[i])

        nxt_feat_dict = {nxt_features[i][idx]:[] for idx in range(end-start)}
        for idx in range(end-start):
            nxt_feat_dict[nxt_features[i][idx]].append(idx)
#         all_set = set(list(range(sizes[i])))
#         print(all_set-feat_set)
        p = old_sample_rate#/(task_idx+0.01)
        rd = if_random_sample
        if old_sample_rate == 1:
            for idx in range(start):        
                if history_features[i][idx] not in feat_set:
                    addset.add(idx)
        for idx in range(start):
            if history_features[i][idx] in nxt_feat_dict and history_features[i][idx] not in feat_set_fix:
                addset2.append(idx)
                nxt_idxs.extend(nxt_feat_dict[history_features[i][idx]]) 
        if old_sample_rate == 2 and examplar_labels is not None:
            for idx in range(len(examplar_labels)):
                if examplar_features[i][idx] not in feat_set:
                    addset_examplar.add(idx) 
                    temp_guard[i][examplar_features[i][idx]]+=1
    
            
    
    if test_on_which == 0:
        for nxt_idx in list(set(nxt_idxs)):
            for j in range(num_categorial_feat):
                nxt_test_features[j].append(nxt_features[j][nxt_idx])
            nxt_test_label.append(nxt_label[nxt_idx])
    if test_on_which == 2:
        nxt_idxs = list(set(range(len(nxt_label)))-set(nxt_idxs))
        for nxt_idx in nxt_idxs:
            for j in range(num_categorial_feat):
                nxt_test_features[j].append(nxt_features[j][nxt_idx])
            nxt_test_label.append(nxt_label[nxt_idx])
    
    if if_random_sample== 0 and old_sample_rate == 0:
        for idx in range(start-len(addset),start):
            for j in range(num_categorial_feat):
                sample_features[j].append(history_features[j][idx])
            sample_label.append(history_label[idx])
    if if_random_sample== 1 and old_sample_rate == 0:
        rp = len(addset)/start
        for idx in range(start):
            u = random.random()
            if u<rp:
                for j in range(num_categorial_feat):
                    sample_features[j].append(history_features[j][idx])
                sample_label.append(history_label[idx])
    if old_sample_rate == 1:
        for idx in list(addset):
            for j in range(num_categorial_feat):
                sample_features[j].append(history_features[j][idx])
            sample_label.append(history_label[idx])
    if old_sample_rate == 2:
        if guards:
            for feat in range(num_categorial_feat):
                for id in range(1,sizes[feat]+1):
                    if temp_guard[feat][id]:
                        guards[feat][id]+=1
                        tguard2[feat][id]+=1
                        # print(feat,id,guards[feat][id])
                    else:
                        guards[feat][id]=0
                        tguard2[feat][id]=0
        if examplar_labels is not None:
            addset_examplar_temp = addset_examplar.copy()
            choosed = DefaultDict(int)
            covered = [set() for _ in range(num_categorial_feat)]
            Ws = DefaultDict(int)
            w_max = 0
            w_argmax = random.choice(list(addset_examplar_temp))
            for ex_idx in addset_examplar_temp:
                w = 0                
                for i in range(num_categorial_feat):
                    f = examplar_features[i][ex_idx]
                    if f not in covered[i]:
                        g = tguard2[i][f]
                        if g:
                            w += 1/g
                            Ws[ex_idx] += 1/g
                if w > w_max:
                    w_argmax = ex_idx
                    w_max = w
            addset_examplar_wcp.add(w_argmax)
            for i in range(num_categorial_feat):
                covered[i].add(examplar_features[i][w_argmax])
            Ws_fix = Ws.copy()
            
            ex_feat_dicts = [
                {examplar_features[i][idx]:[] for idx in addset_examplar_temp}
                for i in range(num_categorial_feat)
            ]
            for idx in addset_examplar:
                for i in range(num_categorial_feat):
                    ex_feat_dicts[i][examplar_features[i][idx]].append(idx)
            for s in range(min(examplar_size-1,len(addset_examplar)-2)):  
                if s%10000==0:
                    print(s)
                w_max = 0
                neigh = []
                for i in range(num_categorial_feat):
                    f = examplar_features[i][w_argmax]
                    if tguard2[i][f]:
                        neigh.extend(ex_feat_dicts[i][f])
                neigh = set(neigh)           
                for ex_idx in neigh:
                    if ex_idx in addset_examplar_temp:
                        w = 0
                        Ws[ex_idx]=0
                        for i in range(num_categorial_feat):
                            f = examplar_features[i][ex_idx]
                            if f not in covered[i]:
                                g = tguard2[i][f]
                                if g:
                                    w += 1/g
                                    Ws[ex_idx] += 1/g
                v=list(Ws.values())
                w_max = max(v)
                w_argmax = list(Ws.keys())[v.index(w_max)]    

                # for ex_idx in addset_examplar:
                #     if Ws[ex_idx] > w_max:
                #         w_argmax = ex_idx
                #         w_max = Ws[ex_idx]
                if w_max<0.1:
                    print("freshed")
                    covered = [set() for _ in range(num_categorial_feat)]
                    Ws = Ws_fix.copy()
                    w_argmax = random.choice(list(addset_examplar_temp))
                addset_examplar_wcp.add(w_argmax)
                addset_examplar_temp.remove(w_argmax)
                Ws.pop(w_argmax)
                Ws_fix.pop(w_argmax)

                for i in range(num_categorial_feat):
                    covered[i].add(examplar_features[i][w_argmax])
                    

        for idx in list(addset_examplar_wcp):
            for j in range(num_categorial_feat):
                sample_features[j].append(examplar_features[j][idx])
            sample_label.append(examplar_labels[idx])
        if examplar_labels is not None:
            examplar_features, examplar_labels = [_[:] for _ in sample_features], sample_label
        
    dataset = CriteoDataset(sample_features, sample_label, 0, len(sample_label))
    test_set = CriteoDataset(nxt_test_features,nxt_test_label,0, len(nxt_test_label))
    print(len(dataset))
    print('len of test set:',len(test_set))
    wandb.log({"test_number":len(test_set)})
    return (
        DataLoader(dataset,shuffle=True,batch_size=batch_size,drop_last=True),
        DataLoader(test_set,shuffle=False, batch_size=batch_size),
        guards,examplar_features,examplar_labels
    )
    
