#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-22 17:40:57
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

import numpy as np
import itertools
import pyflann
from sklearn.neighbors import NearestNeighbors

class spaceBase(object):

	def __init__(self, action_min, action_max, point_each_dim, k, speed):
		"""[summary]
		
		[description]

		Args:
			action_min: [description] shape (1, x)
			action_max: [description]
			point_each_dim: [description]
		"""
		self.action_min = np.array(action_min)

		self.action_max = np.array(action_max)

		self.action_range = self.action_max -self.action_min

		self.action_dim = len(action_min)

		self.space = self.init_uniform_space([0]* self.action_dim, [1]*self.action_dim, point_each_dim)

		self.neigh = NearestNeighbors(n_neighbors=k, radius=1.5)

		self.neigh.fit(self.space)


		# self.flann = pyflann.FLANN()

		# if speed == 'Slow':

		# 	self.space_index = self.flann.build_index(self.space, algorithm='kmeans', branching=16, iterations=1000)

		# elif speed == 'Medium':

		# 	self.space_index = self.flann.build_index(self.space, algorithm='kdtree')

		# elif speed == 'Fast':

		# 	self.space_index = self.flann.build_index(self.space, algorithm='kdtree', trees=10)			

		self.k = k

	def quantization(self, point):

		return (point - self.action_min) / self.action_range

	def deQuantization(self, index):

		return self.action_min + index * self.action_range

	def search_point(self, point):
		"""[summary]
		
		[description]
		1. quantization
		2. search
		3. dequantization
		Args:
			point: [description]
		
		Returns:
			[description]
			[type]
		"""
		
		index = self.quantization(point)

		dists, search_res = self.neigh.kneighbors(index)

		knns = self.space[search_res]

		p_out = []

		for p in knns:

			p_out.append(self.deQuantization(p))

		p_out = np.array(p_out)

		if self.k == 1:

			p_out = p_out[:,np.newaxis,:]

		return p_out	

	def init_uniform_space(self, low, high, point_each_dim):
		
		axis = []

		for i in range(self.action_dim):

			axis.append(list(np.linspace(low[i], high[i], point_each_dim)))

		space = []

		for _ in itertools.product(*axis):

			space.append(list(_))

		return np.array(space)

class spaceContinous(spaceBase):

	def __init__(self, action_min, action_max, point_each_dim, k, speed):

		super(spaceContinous, self).__init__(action_min, action_max, point_each_dim, k, speed)

class spaceDiscrete(spaceBase):

	def __init__(self, action_min, action_max, point_each_dim, k, speed):

		super(spaceDiscrete, self).__init__(action_min, action_max, point_each_dim, k, speed)

	def deQuantization(self,index):

		action = super().deQuantization(index)

		return action.astype(int)
		

if __name__ =='__main__':

	sD = spaceDiscrete([0.0]*9, [2.0]*9, 3, 2, 0)

	# print(sD.space)

	p_out = sD.search_point([[3.0]*9,[2.0]*9])

	print(p_out.shape)

	print(p_out)


