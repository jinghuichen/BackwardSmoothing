# BackwardSmoothing 

This is the official code for our paper [Efficient Robust Training via Backward Smoothing
](https://arxiv.org/abs/2010.01278)(aceepted by AAAI'2022) by [Jinghui Chen](https://jinghuichen.github.io/) (PSU), [Yu Cheng](https://sites.google.com/site/chengyu05) (Microsoft), [Zhe Gan](https://zhegan27.github.io/) (Microsoft),  [Quanquan Gu](http://web.cs.ucla.edu/~qgu/) (UCLA), Jingjing Liu (Tsinghua University).

## Prerequisites
* Python (3.6.9)
* Pytorch (1.7.1)
* CUDA
* numpy


## BackwardSmoothing: A New Method for Efficient Robust Training

 
#### Arguments:
* ```alpha```: step size for perturbation
* ```epsilon```: input space perturbation strength
* ```gamma```: output space perturbation strength
* ```beta```: TRADES robust regularization parameter


### Examples:

* Train Backward Smoothing on CIFAR10 using Resnet-18:
```bash
  $ python3 train_trades_backward.py --arch resnet --dataset cifar10 --beta 10.0 --gamma 1.0 --alpha 0.031 --epsilon 0.031
```

 

## Reference
For technical details and full experimental results, please check [the paper](https://arxiv.org/abs/2010.01278).
```
@inproceedings{chen2022efficient, 
	author = {Chen, Jinghui and Cheng, Yu and Gan, Zhe and Gu, Quanquan and Liu, Jingjing}, 
	title = {Efficient robust training via backward smoothing}, 
	booktitle = {AAAI},
	year = {2022}
}
```
 
 