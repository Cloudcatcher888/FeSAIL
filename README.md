---
typora-root-url: ../FeSAIL
---

# FeSAIL

## Introduction 
This is the official implementation for paper "Feature Staleness Aware Incremental Learning for CTR Prediction", Zhikai, Wang Yanyan, Shen Zibin, Zhang and Kangyi, Lin, in IJCAI 2023.

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
