#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-24 10:25:36
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

import os
import torch.nn as nn

def init_layers_params(layers):

	for layer in layers:

		nn.init.normal_(layer.weight, mean=0, std=0.1)

		nn.init.constant_(layer.bias, 0.1)



