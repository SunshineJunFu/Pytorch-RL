#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 17:00:31
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

import numpy as np

import random

from SumTree import *

class ReplayMemory(object):

	""" Prioritized memory buffer """

	def __init__(self, capacity, epilson=0.01, alpha=0.6, beta=1):

		self.epilson = epilson

		self.alpha = alpha

		self.beta = beta

		self.capacity = capacity

		self.tree = SumTree(self.capacity)

		# self.used_flag = mp.Value('i', 0)

		# self.lock = mp.Lock()

	def _getPriority(self, error):

		return (error + self.epilson) ** self.alpha

	def add(self, error, sample):

		p = self._getPriority(error)

		self.tree.add(p, sample)

	def sample(self, n):

		batch = []

		segment = self.tree.total() / n

		priority = []

		for i in range(n):

			a = segment * i
	
			b = segment * (i + 1)

			s = random.uniform(a, b)

			(idx, p, data) = self.tree.get(s)

			batch.append( (idx, data) )

			priority.append(p)

		sampling_p = np.array(priority) / self.tree.total()

		importance_w = np.power(self.tree.len*sampling_p, -self.beta)

		importance_w /= importance_w.max()

		return batch, importance_w	

	def update(self, idx, error):
		
		p = self._getPriority(error)

		self.tree.update(idx, p)

	def get_len(self):

		return self.tree.len


if __name__ == '__main__':

	memory = ReplayMemory(1000)

	s = [0]

	a = [0]

	r = [0]

	s_ = [0]

	done = [0]

	sample = (s, a, r, s_, done)

	for i in range(1000):

		error = 1 + random.uniform(1, 4)

		memory.add(error, sample)

	batch_data, _ = memory.sample(2)

	print(batch_data, _.shape)

	print(len(memory))

	a = []

	for i in range(10):

		a.append([1,1])

	a = np.array(a)

	print(a.shape)



