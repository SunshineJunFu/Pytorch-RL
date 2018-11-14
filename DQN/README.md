# Deep Q Network

## charateristics:

+ Asynchronous learning on GPU / CPU

+ Prioritized Experience Replay

+ N step Return

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
<img src="https://github.com/fujunustc/Pytorch-RL/raw/master/DQN/imgs/dqn.png" height="280px" alt="图片说明" >
<img src="https://github.com/fujunustc/Pytorch-RL/raw/master/DQN/imgs/Carpole_V0_dqn.gif" height="280px" alt="图片说明" >
</div>

## Reference

### Paper 

1. [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)





