#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 16:51:08
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$


import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
from model import Net
import gym


# CUDA_VISIBLE_DEVICES  #

class train_worker(mp.Process):

	def __init__(self, 	global_model, optimizer, T, lock, res_queue, name, args):
		
		super(train_worker, self).__init__()

		self.global_model = global_model

		self.optimizer = optimizer

		self.T = T 

		self.lock = lock

		self.res_queue = res_queue

		self.name = name

		self.args = args

		# hyper parameters
		self.gamma = self.args.gamma #0.99

		self.beta = self.args.beta #0.01

		self.T_max = self.args.T_max

		self.episode_max_len = self.args.episode_max_len

		# a copy of enviroment

		self.env = gym.make(self.args.env_name).unwrapped

		N_S = self.env.observation_space.shape[0]

		N_A = self.env.action_space.n

		self.agent = Net(N_S, N_A)

		if self.args.resume:

			self.agent.load_state_dict(torch.load(self.args.pretrain_path))

		if self.args.use_gpu and torch.cuda.is_available():

			self.agent = self.agent.cuda().train()

		print('train worker %s start train' % self.name)

	def cal_loss(self, a, r, R, v, p):

		'''
			a: action 
			r: reward
			v: value of state
			done: terminal or not
			p: the distribution		

		'''

		critic_loss , policy_loss = 0, 0

		time_step = len(r)

		R = R.float()

		r = torch.tensor(r).float()

		a = torch.from_numpy(np.array(a))


		if self.args.use_gpu and torch.cuda.is_available():

			R = R.cuda()

			r = r.cuda()

			a = a.cuda()

		for i in reversed(range(time_step)):

			# R ← r_i + γR

			R = r[i] + self.gamma * R

			# Advantage A ← R - V(s_i; θ)

			td_loss = R - v[i]

			# critic loss 

			critic_loss += td_loss.pow(2)

			# Log policy log(π(a_i|s_i; θ)) 

			log_prob  = p[i][0,a[i]].log()

			# dθ ← dθ + ∇θ∙log(π(a_i|s_i; θ))∙A
			policy_loss = policy_loss - ((log_prob * td_loss.detach()) + self.beta * ( -p[i].log() * p[i]).sum(1))

		# average over batch

		total_loss = (critic_loss + policy_loss)

		return total_loss , critic_loss/time_step, policy_loss/time_step #/ time_step


	def adjust_learning_rate(self, lr):
		# Adjusts learning rate
		for param_group in self.optim.param_groups:
			param_group['lr'] = lr
		

	def update_network(self, loss):
		
		self.optimizer.zero_grad()

		loss.backward()

		nn.utils.clip_grad_norm_(self.agent.parameters(), 40)

		for local_param, gnet_parm in zip(self.agent.parameters(), self.global_model.parameters()):

			gnet_parm._grad = local_param.grad

		self.optimizer.step()


	def run(self):

		t = 1

		while self.T.value < self.T_max:

			# reset enviroment
			
			state = self.env.reset()

			# load gnet parameters

			self.agent.load_state_dict(self.global_model.state_dict())

			# start time step #

			t_start = t

			# store episode data #

			a, r, R, v, p = [], [], [], [], []

			done = False

			episode_reward = 0

			episode_loss = 0

			while  not done and t - t_start < self.episode_max_len:

				# perform a_t according to policy 

				prob, v_s, a_t = self.agent.choose_action(state)

				next_state, reward, done, info = self.env.step(a_t)

				# store episode information

				a.append(a_t)

				r.append(reward)

				v.append(v_s)

				p.append(prob)

				t = t + 1

				self.lock.acquire()

				self.T.value = self.T.value + 1

				self.lock.release()

				state = next_state

				# self.env.render()

			if done:

				R = torch.tensor(0)

			else:

				prob, v_s, a_t = self.agent.choose_action(state)

				R = v_s.detach()

			# one episode done perform asynchronous update global d_theta and d_theta_v

			loss, critic_loss, policy_loss = self.cal_loss(a, r, R, v, p)

			self.update_network(loss)

			self.beta = max(1e-3, 1 * (self.T_max - self.T.value) / self.T_max)

			episode_loss = loss.cpu().detach().numpy()

			episode_reward = np.sum(r)

			self.res_queue.put([episode_loss, episode_reward, self.T.value,critic_loss.cpu().detach().numpy(), policy_loss.cpu().detach().numpy()])

		print('train worker %s finish' % self.name)
		

class test_worker(mp.Process):

	def __init__(self, global_model, T, lock, res_queue, name, args):

		super(test_worker, self).__init__()

		self.global_model = global_model

		self.T = T 

		self.lock = lock

		self.res_queue = res_queue

		self.name = name

		self.args = args

		self.T_max = self.args.T_max

		self.best = -1e6

		# a copy of enviroment

		self.env = gym.make(self.args.env_name).unwrapped

		N_S = self.env.observation_space.shape[0]

		N_A = self.env.action_space.n

		self.agent = Net(N_S, N_A)

		if self.args.resume:

			self.agent.load_state_dict(torch.load(self.args.pretrain_path))

		if self.args.use_gpu and torch.cuda.is_available():

			self.agent = self.agent.cuda().train()

		print('test worker %s start train' % self.name)

	def run(self):

		counter = 0

		t = 1

		while True:

			# load gnet parameters #

			self.agent.load_state_dict(self.global_model.state_dict())

			reward_list = []

			for i in range(self.args.test_epoch):

				state = self.env.reset()

				reward_all = 0

				t_start = t

				while True:

					prob, v_s, a_t = self.agent.choose_action(state)

					next_state, reward, done, info = self.env.step(a_t)

					reward_all  += reward

					state = next_state

					t += 1

					if done or t- t_start>self.args.episode_max_len:

						break

				reward_list.append(reward_all)

				self.res_queue.put([reward_all, counter])

				counter += 1

			mean = np.array(reward_list).mean()

			if mean > self.best:

				self.best = mean

				torch.save(self.agent.state_dict(), '%s/pretrain_model.pkl' % self.args.savePath)

			self.lock.acquire()

			value = self.T.value

			self.lock.release()

			if value > self.T_max:

				print('test %s work finish' % self.name)

				break


if __name__ == '__main__':

	pass




