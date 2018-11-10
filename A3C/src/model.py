#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-24 13:41:58
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$


import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 100)
        self.pi2 = nn.Linear(100, a_dim)
        self.v1 = nn.Linear(s_dim, 100)
        self.v2 = nn.Linear(100, 1)
        utils.init_layers_params([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.categorical.Categorical

    def forward(self, x):
        pi1 = F.relu(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = F.relu(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        # self.eval()

        s = torch.from_numpy(s).cuda().float().unsqueeze(0)
        logits, v = self.forward(s)
        prob = F.softmax(logits, dim=1)
        m = self.distribution(prob.data)
        return prob, v, m.sample().cpu().numpy()[0]
        

if __name__ == '__main__':

	net = Net(10,2)

	print(Net)