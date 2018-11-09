#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-06 19:44:37
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

import torch.multiprocessing as mp 

from ddpg import *

from random_process import *

import gym

class train_woker(mp.Process):

	def __init__(self, idx, global_model, global_replayMemory, queue, actor_optimizer, critic_optimzer, args, train_finsh):

		super(train_woker, self).__init__()

		self.idx			= idx 

		self.noise 			= OrnsteinUhlenbeckProcess(0.2)

		self.ddpg 			= DDPG(global_replayMemory, self.noise, actor_optimizer, critic_optimzer, args, 1)

		self.env 			= gym.make(args.env_name)

		self.args 			= args

		self.queue 			= queue

		self.global_model	= global_model

		self.global_replayMemory = global_replayMemory

		self.train_finsh    = train_finsh

		# n-step reward need to be added in next period #

		print('%d train work start' % self.idx)

	def run(self):

		self.ddpg.sync_local_global(self.global_model)

		self.ddpg.sync_target_parameters(self.ddpg.actor_target, self.ddpg.actor)

		self.ddpg.sync_target_parameters(self.ddpg.critic_target, self.ddpg.critic)		

		for i_epoch in range(self.args.Max_epoch):

			#1. reset state of environment and noise #

			state = self.env.reset()

			self.noise.reset_states()

			reward_epoch = 0

			counter = 0

			episode_states = []

			episode_rewards = []

			episode_actions = []

			while counter < 500 :

				action = self.ddpg.select_action(cvtTensor(state))[0]  # shape (action_dim, )

				#2. make action to environment #

				next_state, reward, done, _ = self.env.step(action)

				reward_epoch += reward 
			
			 	#4. store experience to replay memory #

				state_t = state	#(1, x)			

				reward_t = np.array([reward])

				done_t = np.array([0] if not done else [1])

				action_t = action  #norm

				next_state_t = next_state

				episode_actions.append(action_t)

				episode_rewards.append(reward_t)

				episode_states.append(state_t)

				if counter > self.args.n_step:

					cum_reward = 0

					exp_gamma = 1

					for k in range(-self.args.n_step, 0):

						cum_reward += exp_gamma * episode_rewards[k]

						exp_gamma *= self.args.gamma

					sample = (episode_states[-self.args.n_step], episode_actions[-self.args.n_step], cum_reward, next_state_t, done_t)

					self.ddpg.add_sample(sample)

				#5. train model #
				self.ddpg.train(self.global_model)	

				# next iteration #
				state = next_state

				counter += 1

				# finish one epoch #
				if done:

					break

			self.queue.put([self.idx, reward_epoch, i_epoch]) 

		self.train_finsh.value = 1

		print('%d train work finish' % self.idx)

class test_woker(mp.Process):

	def __init__(self, idx, global_model, queue, args, train_finsh_list, test_finish):

		super(test_woker, self).__init__()

		self.idx = idx 

		self.ddpg 			= DDPG(None, None, None, None, args, 0)

		self.env 			= gym.make(args.env_name)

		self.args 			= args

		self.queue 			= queue

		self.global_model	= global_model

		self.train_finsh_list = train_finsh_list

		self.test_finish     = test_finish

		self.best			 = -1e6

		print('%d test work start' % self.idx)

	def run(self):

		cc = 0

		while True:

			self.ddpg.sync_local_global(self.global_model)

			reward_list = []

			for i_epoch in range(self.args.test_epoch):

				#1. reset state of environment and noise #

				state = self.env.reset()

				counter = 0

				reward_epoch = 0

				while counter < 500 :

					# 1. select and perform an action , add noise #	
					action = self.ddpg.select_action(cvtTensor(state))[0]  # shape (action_dim, )

					#2. make action to environment #
					next_state, reward, done, _ = self.env.step(action)

					reward_epoch += reward 

					# next iteration #

					state = next_state

					counter += 1

					# finish one epoch #
					if done:

						break

				self.queue.put([self.idx, reward_epoch, cc*self.args.test_epoch + i_epoch])

				reward_list.append(reward_epoch)

			cc += 1

			mean = np.array(reward_list).mean()

			if mean > self.best:

				self.best = mean 

				torch.save(self.ddpg.actor.state_dict(), '%s/actor.pkl' % self.args.savePath)

				torch.save(self.ddpg.critic.state_dict(),'%s/critic.pkl' % self.args.savePath)

			counter = 0

			for i in range(len(self.train_finsh_list)):

				if self.train_finsh_list[i].value == 1:

					counter += 1

			if counter == len(self.train_finsh_list):

				self.test_finish.value = 1

				break


		print('%d test work finish' % self.idx)