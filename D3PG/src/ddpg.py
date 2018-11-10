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


class DDPG(object):

	def __init__(self, global_replayMemory, noise, actor_optimizer, critic_optimzer, args, trainFlag):

		# model parameters #
		
		self.num_action = args.num_action

		self.num_feature = args.num_feature


		self.use_gpu = args.use_gpu & torch.cuda.is_available()

		self.resume = args.resume

		self.actor_pkl_path = args.actor_pkl_path

		self.critic_pkl_path = args.critic_pkl_path

		# current policy and critic #
		
		self.actor = actorNet(self.num_action, self.num_feature)

		self.critic = criticNet(self.num_action, self.num_feature)

		# old policy and critic #

		self.actor_target = actorNet(self.num_action, self.num_feature)

		self.critic_target = criticNet(self.num_action, self.num_feature)

		self.noise = noise


		if self.resume:

			self.actor.load_state_dict(torch.load('%s'%self.actor_pkl_path))

			self.critic.load_state_dict(torch.load('%s'%self.critic_pkl_path))		

		else:
			
			self.init_model_parameters(self.actor)

			self.init_model_parameters(self.critic)

		self.sync_target_parameters(self.actor_target, self.actor)

		self.sync_target_parameters(self.critic_target, self.critic)	

		# replay memory #		

		self.replay_memory = global_replayMemory

		self.global_counter = 0

		self.update_counter = 0

		# hyper parameters #

		self.tau 		= args.tau

		self.gamma 		= args.gamma ** args.n_step

		self.batch_size = args.batch_size


		if self.use_gpu:

			self.actor = self.actor.cuda()

			self.critic = self.critic.cuda()

			self.actor_target = self.actor_target.cuda()

			self.critic_target = self.critic_target.cuda()

		# optimizer #
		
		self.actor_lr = args.actor_lr

		self.critic_lr = args.critic_lr

		self.weight_decay = args.weight_decay

		self.actor_optimizer = actor_optimizer #torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

		self.critic_optimizer = critic_optimzer #torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)

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

			self.actor.train()

			self.critic.train()

			self.actor_target.train()

			self.critic_target.train()
		else:
			self.actor.eval()

			self.critic.eval()

			self.actor_target.eval()

			self.critic_target.eval()

		if self.trainFlag:

			return self.actor(state).detach().cpu().numpy() + self.noise.sample()

		else:

			return self.actor(state).detach().cpu().numpy()

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

			self.actor.train()

			self.critic.train()

			self.actor_target.train()

			self.critic_target.train()
		else:
			self.actor.eval()

			self.critic.eval()

			self.actor_target.eval()

			self.critic_target.eval()

		return self.actor_target(state)

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


		# compute current Q value #
		current_Q = self.critic(batch_state, batch_action)

		# compute target Q value modify using cur_state #
		target_actions = self.select_target_action(batch_next_state)

		next_q = self.critic_target(batch_next_state, target_actions).detach()

		next_Q = next_q * batch_done

		target_Q = batch_reward + (self.gamma * next_Q)

		# critic loss # 
		
		critic_loss = F.smooth_l1_loss(current_Q, target_Q, reduce=False)

		# add  priority #
		
		critic_loss_cpu = critic_loss.detach().cpu().numpy()

		for i in range(critic_loss_cpu.shape[0]):

			self.replay_memory.add(abs(critic_loss_cpu[i][0]), sample)

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
		current_Q = self.critic(batch_state, batch_action)

		# compute target Q value modify using cur_state #
		target_actions = self.select_target_action(batch_next_state)

		next_q = self.critic_target(batch_next_state, target_actions).detach()

		next_Q = next_q * batch_done

		target_Q = batch_reward + (self.gamma * next_Q)

		# 3 update gradient #
		self.critic_optimizer.zero_grad()

		self.critic.zero_grad()

		# critic loss # 
		critic_loss = F.smooth_l1_loss(current_Q, target_Q, reduce=False)

		# update  priority #
		
		critic_loss_cpu = critic_loss.detach().cpu().numpy()

		for i in range(critic_loss_cpu.shape[0]):

			self.replay_memory.update(idx[i], abs(critic_loss_cpu[i][0]))

		critic_loss = critic_loss * w

		critic_loss = critic_loss.sum()

		critic_loss.backward()

		nn.utils.clip_grad_norm_(self.critic.parameters(), 40)

		self.copy_gradients(self.critic, global_model.critic)

		self.critic_optimizer.step()

		self.actor_optimizer.zero_grad()

		self.actor.zero_grad()
		
		# actor loss maximization ==> minimization
		actor_loss = - self.critic(batch_state, self.actor(batch_state))

		actor_loss = actor_loss * w

		actor_loss = actor_loss.sum()

		actor_loss.backward()

		nn.utils.clip_grad_norm_(self.actor.parameters(), 40)

		self.copy_gradients(self.actor, global_model.actor)

		self.actor_optimizer.step()

		self.global_counter +=1

		# self.writer.add_scalar('critic_loss', critic_loss.detach().cpu().numpy(), self.global_counter)

		# self.writer.add_scalar('actor_loss', actor_loss.detach().cpu().numpy(), self.global_counter)

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

		if self.replay_memory.get_len() < self.batch_size: # not use len()
			return 

		batch_data, importance_weight = self.replay_memory.sample(self.batch_size)
	
		self.learning_rate_decay(self.actor_optimizer, self.update_counter,self.decay_lr, self.decay_slot )

		self.learning_rate_decay(self.critic_optimizer, self.update_counter,self.decay_lr, self.decay_slot)

		#3.
		self.update_network(batch_data, importance_weight, global_model)

		# copy global network weights to local
		
		self.sync_local_global(global_model)

		#4.
		self.update_target(self.actor_target, self.actor)

		self.update_target(self.critic_target, self.critic)

		self.update_counter     +=1

	def copy_gradients(self, model_local, model_global):
		for param_local, param_global in zip(model_local.parameters(), model_global.parameters()):
			if param_global.grad is not None:
				return
			param_global._grad = param_local.grad


	def sync_grad_with_global_model(self, global_model):
		self.copy_gradients(self.actor, global_model.actor)
		self.copy_gradients(self.critic, global_model.critic)

	def sync_local_global(self, global_model):

		self.actor.load_state_dict(global_model.actor.state_dict())

		self.critic.load_state_dict(global_model.critic.state_dict())

	def share_memory(self):

		self.actor.share_memory()

		self.critic.share_memory()


if __name__ == '__main__':

	pass







