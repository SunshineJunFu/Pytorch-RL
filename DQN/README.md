# Deep Q Network

## charateristics:

+ Asynchronous learning on GPU / CPU

+ Multiprocess

## Prerequisites

Pytorch     :  >= 0.4.0 

OpenAI Gym  :  >= 1.1.0

## Usage

### Train

```Shell
python ./src/multi_agent.py --train_worker_nums 16  --test_epoch 10 --T_max 1000000
```


### Test

```Shell
python  --resume 1 --pretrain_path ./model/pretrain_model.pkl
```

### Experiment Results

<div align="center">
<img src="https://github.com/fujunustc/Pytorch-RL/raw/master/A3C/imgs/Carpole_V0.png" height="280px" alt="图片说明" >
<img src="https://github.com/fujunustc/Pytorch-RL/raw/master/A3C/imgs/Carpole_V0_eval.gif" height="280px" alt="图片说明" >
</div>

## Reference

### code
1. https://github.com/chenaddsix/pytorch_a3c
2. https://github.com/MorvanZhou/pytorch-A3C

### Paper 

1. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)





