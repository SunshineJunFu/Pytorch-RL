# Deep Q Network

## charateristics:

+ Asynchronous learning on GPU / CPU

+ Prioritized Experience Replay

+ N step Return

## Prerequisites

+ Pytorch     :  >= 0.4.0 

+ OpenAI Gym  :  >= 1.1.0

## Usage

### Train

```Shell
python ./src/multi_agent.py  --train_worker_nums 1 --Max_epoch 5000 --test_epoch 10 --n_step 5
```


### Test

```Shell
python ./src/evaluation.py   --resume 1 --qnet_pkl_path ./model/agent.pkl
```

### Experiment Results

<div align="center">
<img src="https://github.com/fujunustc/Pytorch-RL/raw/master/DQN/imgs/dqn.png" height="280px" alt="图片说明" >
<img src="https://github.com/fujunustc/Pytorch-RL/raw/master/DQN/imgs/Carpole_V0_dqn.gif" height="280px" alt="图片说明" >
</div>

## How to apply into specific environment 


## Reference

### Paper 

1. [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)





