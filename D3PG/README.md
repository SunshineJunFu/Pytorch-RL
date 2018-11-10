# Distributed Deep Deterministic Policy Gradient 

## charateristics:

+ Shared Priority Experience Replay

+ N step return learning 

+ Asynchorous learning


## Prerequisites

Pytorch     :  >= 0.4.0 

OpenAI Gym  :  >= 1.1.0


## Usage

### Train

```Shell
python ./src/multi_agent.py  --k 1000 --train_worker_nums 16 --Max_epoch 5000 --test_epoch 10 --n_step 5
```


### test

```Shell
python ./src/evaluation.py  --actor_pkl_path ./model/actor.pkl  --critic_pkl_path ./model/critic.pkl 
```

### Experiment Results
<div align="center">
<img src="https://github.com/fujunustc/Pytorch-RL/raw/master/D3PG/imgs/Mountain_test.png" height="280px" alt="图片说明" >
<img src="https://github.com/fujunustc/Pytorch-RL/raw/master/D3PG/imgs/MountainCar.gif" height="280px" alt="图片说明" >
</div>


## Reference

### code

1. https://github.com/ajgupta93/d3pg-pytorch

2. https://github.com/ghliu/pytorch-ddpg

3. https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py

4. https://stackoverflow.com/questions/3671666/sharing-a-complex-object-between-python-processes

### Paper 

1. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

2. [Comparing Deep Reinforcement Learning and Evolutionary Methods
in Continuous Control](https://arxiv.org/pdf/1712.00006.pdf)


