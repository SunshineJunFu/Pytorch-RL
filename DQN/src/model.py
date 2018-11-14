#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-21 09:49:50
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$


import torch.nn as nn

class QNet(nn.Module):
	"""[summary]
	
	
	"""

	def __init__(self, num_action, num_feature):

		super(QNet, self).__init__()

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

			nn.Linear(256, num_action)

			)


	def forward(self, state):

		x = self.fc1(state)

		x = self.fc2(x)

		x = self.out(x)

		return x


if __name__ == '__main__':

	ac = QNet(10,10)

	print(ac)


