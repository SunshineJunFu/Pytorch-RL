#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 19:54:42
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

	parser.add_argument('--resume', type=int, required=False, default= 0, help='use pretrained model')

	parser.add_argument('--pretrain_path', type=str, required=False, default= './model/actor.pkl', help='actor pretrained model')


	# training parameters #

	parser.add_argument('--gamma', type=float, required=False, default= 0.99, help='discount coefficient')

	parser.add_argument('--beta', type=float, required=False, default= 0.01, help='entropy regularization')

	parser.add_argument('--lr', type=float, required=False, default= 0.01, help='learning rate')

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



env = gym.make('CartPole-v0')

N_S = env.observation_space.shape[0]

N_A = env.action_space.n


if __name__ == '__main__':

	mp.set_start_method('spawn', force=True)

	args = parser_usage()

	writer = SummaryWriter(log_dir='%s/logs' % args.logPath )

	gnet = Net(N_S, N_A)

	gnet.share_memory()

	if args.use_gpu and torch.cuda.is_available():

		gnet = gnet.cuda().train()

	optimizer = SharedRMSprop(gnet.parameters(), lr=args.lr)

	optimizer.share_memory()

	T = mp.Value('i', 0)

	lock = mp.Lock()

	train_work_list = []

	train_queue = []

	for i in range(args.train_worker_nums):

		queue = mp.Queue(int(1e6))

		train_queue.append(queue)

		train_work_list.append(train_worker(gnet, optimizer, T, lock, queue, '%d' % i, args))

	test_queue = mp.Queue(int(1e6))

	test  = test_worker(gnet, T, lock, test_queue, '%d' % i, args)

	for w in train_work_list:
		
		w.start()

	test.start()

	while True:


		if T.value > args.T_max:

			break

		else:

			if test_queue.empty():

				pass

			else:

				data = test_queue.get()

				writer.add_scalar('test_reward', data[0], data[1])


	# wait all proces to stop

	for w in train_work_list:
		w.join()


	for i in range(args.train_worker_nums):

		train_work_list[i].join()

	test.join()