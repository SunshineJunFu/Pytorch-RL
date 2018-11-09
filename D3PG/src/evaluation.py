#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-09 11:49:33
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

from ddpg import *

import argparse

import gym

from utils import *

def parser_usage():

	# 1. create parser #
	
	parser = argparse.ArgumentParser('d3pg for OpenAI environment')


	# model parameters #
	parser.add_argument('--num_action', type=int, required=False, default=1, help='train worker nums')

	parser.add_argument('--num_feature', type=int, required=False, default=2, help='train worker nums')

	parser.add_argument('--use_gpu', type=int, required=False, default= 1, help='enable use gpu')

	parser.add_argument('--resume', type=int, required=False, default= 1, help='use pretrained model')

	parser.add_argument('--actor_pkl_path', type=str, required=False, default= '../model/actor.pkl', help='actor pretrained model')

	parser.add_argument('--critic_pkl_path', type=str, required=False, default= '../model/critic.pkl', help='critic pretrained model')

	# training parameters #

	parser.add_argument('--tau', type=float, required=False, default= 0.001, help='tau')

	parser.add_argument('--gamma', type=float, required=False, default= 0.99, help='discount coefficient')

	parser.add_argument('--batch_size', type=int, required=False, default= 32, help='use pretrained model')	

	parser.add_argument('--actor_lr', type=float, required=False, default= 0.0001, help='actor learning rate')

	parser.add_argument('--critic_lr', type=float, required=False, default= 0.001, help='critic learning rate')	

	parser.add_argument('--weight_decay', type=float, required=False, default= 0.01, help='weight_decay')

	parser.add_argument('--decay_lr', type=float, required=False, default= 0.9, help='lr decay rate')

	parser.add_argument('--decay_slot', type=int, required=False, default= 5000, help='lr decay slot')

	# space parameters #

	parser.add_argument('--k', type=int, required=False, default= 1000, help='number of neighbourhood')

	parser.add_argument('--capacity', type=int, required=False, default= 1000000, help='replay memory size')

	parser.add_argument('--Max_epoch', type=int, required=False, default= 1, help='Max_epoch')

	parser.add_argument('--test_epoch', type=int, required=False, default= 1, help='Max_epoch')

	parser.add_argument('--speed', type=str, required=False, default='Fast', help='pyflann search speed')	

	parser.add_argument('--train_worker_nums', type=int, required=False, default=1, help='train worker nums')


	parser.add_argument('--n_step', type=int, required=False, default=1, help='train worker nums')

	parser.add_argument('--env_name', type=str, required=False, default="MountainCarContinuous-v0", help='env env_name')

	# dataloader parameters #

	parser.add_argument('--savePath', type=str, required=False, default='./model/', help='save model path')

	parser.add_argument('--logPath', type=str, required=False, default='./', help='log path')

	args = parser.parse_args()

	return args




class test_woker(object):

	def __init__(self, idx, args):

		self.idx = idx 

		self.ddpg 			= DDPG(None, None, None, None, args, 0)

		self.env 			= gym.make(args.env_name)

		self.args 			= args

		print('%d test work start' % self.idx)

	def run(self):

		cc = 0

		while True:

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

					self.env.render()

					reward_epoch += reward 

					# next iteration #

					state = next_state

					counter += 1

					# finish one epoch #
					if done:

						break

				reward_list.append(reward_epoch)

			cc += 1


		print('%d test work finish' % self.idx)


if __name__=='__main__':

	args = parser_usage()

	test = test_woker(0, args)

	test.run()