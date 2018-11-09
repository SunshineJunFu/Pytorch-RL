#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-21 09:49:50
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

import torch
import torch.nn as nn

class actorNet(nn.Module):
	"""[summary]
	
	[description]
	

	actor network
	"""

	def __init__(self, num_action, num_feature):

		super(actorNet, self).__init__()

		self.fc1 = nn.Sequential(

				nn.Linear(num_feature, 1024),

				nn.LayerNorm(1024),

				nn.ReLU(),

				# nn.Dropout(0.5)
			)

		self.fc2 = nn.Sequential(

				nn.Linear(1024, 256),

				nn.LayerNorm(256),

				nn.ReLU(),

				# nn.Dropout(0.5)
			)

		self.out = nn.Sequential(
			nn.Linear(256, num_action),

			nn.Tanh()  # (-1.0, 1.0)
			)


	def forward(self, state):

		x = self.fc1(state)

		x = self.fc2(x)

		x = self.out(x)

		return x

class criticNet(nn.Module):
	"""[summary]
	
	[description]
	critic network
	"""

	def __init__(self, num_action, num_feature):

		super(criticNet, self).__init__()

		self.fc1 = nn.Sequential(

				nn.Linear(num_feature, 1024),

				nn.LayerNorm(1024),

				nn.ReLU(),

				# nn.Dropout(0.5)
			)

		self.fc2 = nn.Sequential(

				nn.Linear(1024 + num_action, 256),

				nn.LayerNorm(256),

				nn.ReLU(),

				# nn.Dropout(0.5)
			)

		self.out = nn.Linear(256, 1)


	def forward(self, state, action):

		action = action.cuda()

		x = self.fc1(state)

		x = torch.cat((x, action), dim=1)

		x = self.fc2(x)

		x = self.out(x)

		return x

if __name__ == '__main__':

	ac = actorNet(10,10)

	for name, param in ac.named_parameters():

		if 'weight' in name:
			
			print(name, param.size())

