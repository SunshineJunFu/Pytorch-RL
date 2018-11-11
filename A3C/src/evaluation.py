#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-10 21:05:40
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

from model import *

from optim import *

import gym

import argparse

from AsynchronousWorker import *

import torch.multiprocessing as mp

from  tensorboardX import SummaryWriter 


def parser_usage():

	# 1. create parser #
	
	parser = argparse.ArgumentParser('GA3C for OpenAI environment')

	# model parameters #

	parser.add_argument('--use_gpu', type=int, required=False, default= 1, help='enable use gpu')

	parser.add_argument('--resume', type=int, required=False, default= 1, help='use pretrained model')

	parser.add_argument('--pretrain_path', type=str, required=False, default= './model/pretrain_model.pkl', help='actor pretrained model')


	# training parameters #

	parser.add_argument('--gamma', type=float, required=False, default= 0.99, help='discount coefficient')

	parser.add_argument('--beta', type=float, required=False, default= 0.01, help='entropy regularization')

	parser.add_argument('--lr', type=float, required=False, default= 0.0001, help='learning rate')

	parser.add_argument('--T_max', type=int, required=False, default= 1000000, help='T_max')

	parser.add_argument('--episode_max_len', type=int, required=False, default= 1000, help='max episode len')

	parser.add_argument('--train_worker_nums', type=int, required=False, default=1, help='train worker nums')

	parser.add_argument('--test_epoch', type=int, required=False, default=10, help='test_epoch')

	parser.add_argument('--env_name', type=str, required=False, default="CartPole-v0", help='env env_name')

	# dataloader parameters #

	parser.add_argument('--savePath', type=str, required=False, default='./model/', help='save model path')

	parser.add_argument('--logPath', type=str, required=False, default='./', help='log path')



	args = parser.parse_args()

	return args


class test_worker(object):

	def __init__(self,  args):

		self.args = args

		# a copy of enviroment

		self.env = gym.make(self.args.env_name).unwrapped

		N_S = self.env.observation_space.shape[0]

		N_A = self.env.action_space.n

		self.agent = Net(N_S, N_A)

		if self.args.resume:

			self.agent.load_state_dict(torch.load(self.args.pretrain_path))

		if self.args.use_gpu and torch.cuda.is_available():

			self.agent = self.agent.cuda().eval()


	def run(self):

		while True:

			for i in range(self.args.test_epoch):

				state = self.env.reset()

				reward_all = 0

				while True:

					prob, v_s, a_t = self.agent.choose_action(state)

					next_state, reward, done, info = self.env.step(a_t)

					state = next_state

					self.env.render()

					reward_all  += reward

					if done:

						break

				print(reward_all)

			break


if __name__ == "__main__":

	args = parser_usage()

	test = test_worker(args)

	test.run()