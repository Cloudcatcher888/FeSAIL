---
typora-root-url: ../FeSAIL
---

# FeSAIL

## Introduction 
This is the official implementation for paper "Feature Staleness Aware Incremental Learning for CTR Prediction", Zhikai, Wang Yanyan, Shen Zibin, Zhang and Kangyi, Lin, in IJCAI 2023.

## Abstract
Click-through Rate (CTR) prediction in real-world recommender systems often deals with billions of user interactions every day. To improve the training efficiency, it is common to update the CTR prediction model incrementally using the new incremental data and a subset of historical data. However, the feature embeddings of a CTR prediction model often get stale when the corresponding features do not appear in current incremental data. In the next period, the model would have a performance degradation on samples containing stale features, which we call the feature staleness problem. To mitigate this problem, we propose a Feature Staleness Aware Incremental Learning method for CTR prediction (FeSAIL) which adaptively replays samples containing stale features. We first introduce a staleness aware sampling algorithm (SAS) to sample a fixed number of stale samples with high sampling efficiency. We then introduce a staleness aware regularization mechanism (SAR) for a fine-grained control of the feature embedding updating. We instantiate FeSAIL with a general deep learningbased CTR prediction model and the experimental results demonstrate FeSAIL outperforms various state-of-the-art methods on four benchmark datasets.

## Architecture

![](/overview.jpg)


## Requirement

```
pytorch == 1.14
python == 3.7
```

## Instruction
1, Unzip the datasets into the root dir first.

2, You can run the code by: 
```
python code/main.py
```

3, You can change ifrandom into 1 as random selection, and change oldrate into 1 to activate old sampling.

4, result on Criteo:

![](/ablation1.jpg)

# Reference

Please cite our paper if you use this code.

```
@inproceedings{wang2023fesail,
  title={Feature Staleness Aware Incremental Learning for CTR Prediction},
  author={Zhikai Wang and Yanyan Shen},
  booktitle={IJCAI},
  year={2023}
}
```