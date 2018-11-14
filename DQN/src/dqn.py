#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-20 21:02:55
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

import numpy as np

import torch
import torch.nn.functional as F

from ReplayMemory import *
from model import *


class DQN(object):

	def __init__(self, global_replayMemory, q_optimizer, args, trainFlag):

		# model parameters #
		
		self.num_action = args.num_action

		self.num_feature = args.num_feature

		self.use_gpu = args.use_gpu & torch.cuda.is_available()

		self.resume = args.resume

		self.qnet_pkl_path = args.qnet_pkl_path 

		# current policy and critic #
		
		self.agent = QNet(self.num_action, self.num_feature)

		self.agent_target = QNet(self.num_action, self.num_feature)


		if self.resume:

			self.agent.load_state_dict(torch.load('%s'%self.qnet_pkl_path))

		else:
			
			self.init_model_parameters(self.agent)

		self.sync_target_parameters(self.agent_target, self.agent)


		# replay memory #		

		self.replay_memory = global_replayMemory

		self.global_counter = 0

		self.update_counter = 0

		# hyper parameters #

		self.tau 		= args.tau

		self.gamma 		= args.gamma ** args.n_step

		self.batch_size = args.batch_size


		if self.use_gpu:

			self.agent = self.agent.cuda()

			self.agent_target = self.agent_target.cuda()

		# optimizer #
		
		self.agent_lr = args.lr

		self.weight_decay = args.weight_decay

		self.q_optimizer = q_optimizer 

		self.trainFlag = trainFlag

		self.decay_lr = args.decay_lr

		self.decay_slot = args.decay_slot

	def init_model_parameters(self, model):

		for name, param in model.named_parameters():

			if 'out' in name:

				torch.nn.init.uniform_(param, -3e-3, 3e-3)

			else:

				torch.nn.init.uniform_(param, -1.0/np.sqrt(param.size()[0]), 1.0/np.sqrt(param.size()[0]))

	def sync_target_parameters(self, target_model, model):

		for target_param, param in zip(target_model.parameters(), model.parameters()):

			target_param.data.copy_(param.data)

	def select_action(self, state):
		"""[summary]
		
		[description]
		
		Args:
			state: [description] 
			shape: (batch_size, xx, xx...)
			dtype
		"""
		# print(state.shape)
		if self.use_gpu:

			state = state.cuda()

		if self.trainFlag:

			self.agent.train()

		else:

			self.agent.eval()

		value = self.agent(state)

		max_index = torch.argmax(value, dim=1, keepdim=True)

		return max_index.detach().cpu().numpy()[0]


	def select_target_action(self, state):
		"""[summary]
		
		[description]
		
		Args:
			state: [description] 
			shape: (batch_size, xx, xx...)
			dtype
		"""
		if self.use_gpu:

			state = state.cuda()

		if self.trainFlag:

			self.agent_target.train()

		else:

			self.agent_target.eval()

		value = self.agent(state)	
		
		max_index = torch.argmax(value, dim=1, keepdim=True)	

		value = torch.gather(value,  1, max_index.long())	

		return value.detach()

	def add_sample(self, sample):

		states =  sample[0][np.newaxis,:] # (1, element dim)

		actions = sample[1][np.newaxis,:]

		rewards = sample[2][np.newaxis,:]

		states_ = sample[3][np.newaxis,:]

		done 	= sample[4][np.newaxis,:]


		batch_state = torch.from_numpy(states).float()

		batch_action = torch.from_numpy(actions).float()

		batch_reward = torch.from_numpy(rewards).float()

		batch_next_state = torch.from_numpy(states_).float()

		batch_done = 1 - torch.from_numpy(done).float()


		if self.use_gpu:
		
			batch_state = batch_state.cuda()

			batch_action = batch_action.cuda()

			batch_reward = batch_reward.cuda()

			batch_next_state = batch_next_state.cuda()

			batch_done = batch_done.cuda()

		current_value = self.agent(batch_state).detach()

		current_Q = torch.gather(current_value, 1, batch_action.long())

		next_q = self.select_target_action(batch_next_state).detach()

		next_Q = next_q * batch_done

		target_Q = batch_reward + (self.gamma * next_Q)

		# critic loss # 
		
		td = F.smooth_l1_loss(current_Q, target_Q, reduce=False)

		# add  priority #
		
		td_cpu = td.detach().cpu().numpy()

		for i in range(td_cpu.shape[0]):

			self.replay_memory.add(abs(td_cpu[i][0]), sample)

	def update_network(self, batch, importance_weight, global_model):
		"""[summary]
		
		[description]
		1. extract state, reward, action, next_state, done information
		2. calculate critic_loss and actor_loss
		3. backward 
		Args:
			batch_data:  (idx, (s, a, r, s_, done))
		"""
		# 1. 

		idx 	= np.array([o[0] for o in batch])

		states =  np.array([o[1][0] for o in batch]) # (Batchsize, element dim)

		actions = np.array([o[1][1] for o in batch])

		rewards = np.array([o[1][2] for o in batch])

		states_ = np.array([o[1][3] for o in batch])

		done 	= np.array([o[1][4] for o in batch])

		batch_state = torch.from_numpy(states).float()

		batch_action = torch.from_numpy(actions).float()

		batch_reward = torch.from_numpy(rewards).float()

		batch_next_state = torch.from_numpy(states_).float()

		batch_done = 1 - torch.from_numpy(done).float()

		w = torch.from_numpy(importance_weight).float().unsqueeze(1)


		if self.use_gpu:
		
			batch_state = batch_state.cuda()

			batch_action = batch_action.cuda()

			batch_reward = batch_reward.cuda()

			batch_next_state = batch_next_state.cuda()

			batch_done = batch_done.cuda()

			w 			= w.cuda()


		# compute current Q value #
		current_value = self.agent(batch_state)

		current_Q = torch.torch.gather(current_value, 1, batch_action.long())

		# compute target Q value modify using cur_state #
		next_q = self.select_target_action(batch_next_state)

		next_Q = next_q * batch_done

		target_Q = batch_reward + (self.gamma * next_Q)

		self.agent.zero_grad()

		td = F.smooth_l1_loss(current_Q, target_Q, reduce=False)

		# update  priority #
		
		td_cpu = td.detach().cpu().numpy()

		for i in range(td_cpu.shape[0]):

			self.replay_memory.update(idx[i], abs(td_cpu[i][0]))

		td = td * w

		td = td.sum()

		td.backward()

		self.copy_gradients(self.agent, global_model.agent)

		self.q_optimizer.step()

		self.global_counter +=1


	def update_target(self, target_model, model):
		
		for target_param, param in zip(target_model.parameters(), model.parameters()):

			target_param.data.copy_(self.tau * param.data + (1- self.tau) * target_param.data)

	def learning_rate_decay(self, optimizer, epoch, lr_decay, lr_decay_epoch):

		if epoch % lr_decay_epoch == 0 and epoch >0:

			for param_group in optimizer.param_groups:

				param_group['lr'] = param_group['lr'] * lr_decay if param_group['lr'] * lr_decay > 1e-5 else 1e-5

	# for multiprocess #

	def train(self, global_model):
		"""[summary]
		
		[description]
		1. transfer model to train 
		2. sample a batch size data from replay buffer
		3. update critic and actor 
		4. update target critic and actor

		Args:
			: [description]
		"""
		# 1.

		if self.replay_memory.get_len() < self.batch_size:
			return 

		batch_data, importance_weight = self.replay_memory.sample(self.batch_size)
	
		#3.
		self.update_network(batch_data, importance_weight, global_model)

		# copy global network weights to local
		
		self.sync_local_global(global_model)

		#4.
		self.update_target(self.agent_target, self.agent)

		self.update_counter     +=1

	def copy_gradients(self, model_local, model_global):
		for param_local, param_global in zip(model_local.parameters(), model_global.parameters()):
			if param_global.grad is not None:
				return
			param_global._grad = param_local.grad


	def sync_grad_with_global_model(self, global_model):
		self.copy_gradients(self.agent, global_model.agent)

	def sync_local_global(self, global_model):

		self.agent.load_state_dict(global_model.agent.state_dict())

	def share_memory(self):

		self.agent.share_memory()


if __name__ == '__main__':

	pass







