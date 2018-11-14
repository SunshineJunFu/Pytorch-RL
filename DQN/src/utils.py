#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-23 11:40:18
# @Author  : Jun Fu (fujun@mail.ustc.edu.cn)
# @Version : $Id$

import torch

import copy

import numpy as np


def cvtTensor(data):
    """ summary the function of cvtTensor

    add batch size, convert data into torch and transfer to cfg.device

    Args:
            data: list or tuple or nd.array

    Returns:
            [description] a batch size of data shape(1,x,x,x)
            [type] torch.tensor
    """

    data = copy.deepcopy(data)

    data = np.array(data)[np.newaxis, :]

    data = torch.from_numpy(data)

    data = data.float().cuda()

    return data



