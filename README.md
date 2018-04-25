# Attentive Interactive Convolutional Matching for Community Question Answering in Social Multimedia


## Abstract

Nowadays, community-based question answering (CQA) services have accumulated millions of users to share valuable knowledge.
An essential function in CQA tasks is the accurate matching of answers w.r.t given questions.
Existing methods usually ignore the redundant, heterogeneous, and multi-modal properties of CQA systems.
In this paper, we propose a multi-modal attentive interactive convolutional matching method (MMAICM) to model the multi-modal content and social context jointly for questions and answers in a unified framework for CQA retrieval, which explores the redundant, heterogeneous, and multi-modal properties of CQA systems jointly.
A well-designed attention mechanism is proposed to focus on useful word-pair interactions and neglect meaningless and noisy word-pair interactions.
Moreover, a multi-modal interaction matrix method and a novel meta-path based network representation approach are proposed to consider the multi-modal content and social context, respectively.
The attentive interactive convolutional matching network is proposed to infer the relevance between questions and answers, which can capture both the lexical and the sequential information of the contents.
Experiment results on two real-world datasets demonstrate the superior performance of MMAICM compared with other
state-of-the-art algorithms.

## Framework
![](https://github.com/briefcopy/MMAICM/raw/master/images/framework.png)



## Datasets

+ Quoda Dataset: [https://github.com/briefcopy/MMAICM/tree/master/datasets/quora](https://github.com/briefcopy/MMAICM/tree/master/datasets/quora)
+ Zhihu Dataset: [https://github.com/briefcopy/MMAICM/tree/master/datasets/zhihu](https://github.com/briefcopy/MMAICM/tree/master/datasets/zhihu)

The social network structure data and QA content data are organized into python objects by `pickle`.
In terms of the image data, we provide the image urls instead of the raw image data since the image data is large for Github.
The two datasets should be loaded as follows:

```python
#coding=utf-8
import pickle

data_path = "quora/zhihu data file path"
with open(data_path, "rb") as f:
    network = pickle.load(f)
```
