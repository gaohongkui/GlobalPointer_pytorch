# GlobalPointer_pytorch

> 喜欢本项目的话，欢迎点击右上角的star，感谢每一个点赞的你。

## 项目介绍

本项目的模型参考苏剑林的文章[GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://kexue.fm/archives/8373)，并用Pytorch实现。

![GlobalPoniter多头识别嵌套实体示意图](https://kexue.fm/usr/uploads/2021/05/2377306125.png "GlobalPoniter多头识别嵌套实体示意图")

GlobalPointer的设计思路与[TPLinker-NER](https://github.com/gaohongkui/TPLinker-NER)类似，但在实现方式上不同。具体体现在：

1. 加性乘性Attention

TPLinker在Multi-Head上用的是加性Attention：

<!-- $$s_α(i,j)=W_{o,α}tanh(W_{h,α}[h_i,h_j]+b_{h,α})+b_{o,α}$$ -->

![](https://latex.codecogs.com/svg.image?s%20_%20%7B%20%5Calpha%20%7D%20(%20i%20,%20j%20)%20=%20W%20_%20%7B%20o%20,%20%5Calpha%20%7D%20%5Coperatorname%20%7B%20tanh%20%7D%20(%20W%20_%20%7B%20h%20,%20%5Calpha%20%7D%20%5B%20h%20_%20%7B%20i%20%7D%20,%20h%20_%20%7B%20j%20%7D%20%5D%20+%20b%20_%20%7B%20h%20,%20%5Calpha%20%7D%20)%20+%20b%20_%20%7B%20o%20,%20%5Calpha%20%7D)

而GlobalPointer用的是乘性Attention：

<!-- $$s_α(i,j)=q^⊤_{i,α}k_{j,α}$$ -->

![](https://latex.codecogs.com/svg.image?s%20_%20%7B%20%5Calpha%20%7D%20(%20i%20,%20j%20)%20=%20q%20_%20%7B%20i%20,%20%5Calpha%20%7D%20%5E%20%7B%20T%20%7D%20k%20_%20%7B%20j%20,%20%5Calpha%20%7D)

2. 位置编码

GlobalPointer在模型中还加入了一种旋转式位置编码[RoPE](https://kexue.fm/archives/8265)。这是一种“通过绝对位置编码的方式实现相对位置编码”，在本模型中效果明显。

## Usage

### 实验环境

本次实验进行时Python版本为3.6，其他主要的第三方库包括：

* pytorch==1.8.1
* wandb==0.10.26 #for logging the result
* transformers==4.1.1
* tqdm==4.54.1

### 下载预训练模型

请下载Bert的中文预训练模型[bert-base-chinese](https://huggingface.co/bert-base-chinese)存放至 `pretrained_models/`，并在config.py中配置正确的bert_path

### Train

```
python train.py
```

### Evaluation

```
python evaluate.py
```

## 实验结果

默认配置（超参数已在 `config.py` 文件中），数据集是 CLUENER

* 验证集 Best F1：0.7966
