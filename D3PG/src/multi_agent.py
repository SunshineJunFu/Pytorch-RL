#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-06 20:09:13
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

from ddpg import *

from ReplayMemory import *

from optim import *

import argparse

from asynchronousWorker import *

import torch.multiprocessing as mp 

from tensorboardX import SummaryWriter

def parser_usage():

	# 1. create parser #
	
	parser = argparse.ArgumentParser('d3pg for OpenAI environment')


	# model parameters #
	parser.add_argument('--num_action', type=int, required=False, default=1, help='train worker nums')

	parser.add_argument('--num_feature', type=int, required=False, default=2, help='train worker nums')

	parser.add_argument('--use_gpu', type=int, required=False, default= 1, help='enable use gpu')

	parser.add_argument('--resume', type=int, required=False, default= 0, help='use pretrained model')

	parser.add_argument('--actor_pkl_path', type=str, required=False, default= './model/actor.pkl', help='actor pretrained model')

	parser.add_argument('--critic_pkl_path', type=str, required=False, default= './model/critic.pkl', help='critic pretrained model')

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


def main(args):

	# globale model #
	
	global_replayMemory 	= ReplayMemory(args.capacity)

	global_wolp_ddpg 		= DDPG(global_replayMemory, None, None, None, args, 1)

	actor_optimizer 		= SharedAdam(global_wolp_ddpg.actor.parameters(), lr=args.actor_lr)

	critic_optimzer			= SharedAdam(global_wolp_ddpg.critic.parameters(), lr=args.critic_lr)

	global_wolp_ddpg.share_memory()

	writer 					= SummaryWriter('%s/logs' % args.logPath)

	# create  train  and test worker #

	train_work_list = []

	train_finsh_list = []

	train_queue = []

	for i in range(args.train_worker_nums):

		queue = mp.Queue(int(1e6))

		train_finsh = mp.Value('i',0)

		train_queue.append(queue)

		train_finsh_list.append(train_finsh)

		train_work_list.append(train_woker(i, global_wolp_ddpg, global_replayMemory, queue, actor_optimizer, critic_optimzer, args, train_finsh))

	test_queue = mp.Queue(int(1e6))

	test_finish = mp.Value('i',0)

	test  = test_woker(args.train_worker_nums, global_wolp_ddpg, test_queue, args, train_finsh_list, test_finish)

	# start worker #
	
	for i in range(args.train_worker_nums):

		train_work_list[i].start()

	test.start()

	# visulize #
	
	while True:

		if test_finish.value ==1:

			break

		else:

			train_flags = 0

			for i in range(len(train_work_list)):

				train_flags |=train_queue[i].empty()

			if train_flags:

				pass

			else:

				reward_dict = {}

				mean = 0

				step = 0

				for i in range(len(train_work_list)):

					data = train_queue[i].get()

					reward_dict['work_%d'%data[0]] = data[1]

					step = data[2]

					mean += data[1]

				reward_dict['avg'] = mean / len(train_work_list)

				writer.add_scalars('reward', reward_dict, step)

			if test_queue.empty():

				pass

			else:

				data = test_queue.get()

				writer.add_scalar('test_work_%d_reward'% data[0], data[1], data[2])

	# join work stop #

	for i in range(args.train_worker_nums):

		train_work_list[i].join()

	test.join()


if __name__ == '__main__':

	mp.set_start_method('spawn', force=True)

	args = parser_usage()

	main(args)