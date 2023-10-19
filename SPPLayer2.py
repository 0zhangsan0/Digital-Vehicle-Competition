"""
空间金字塔池化，不等长序列化为等长序列
"""

import math

from torch import nn
from torch.nn import functional as F

import torch


class SPPLayer(nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        batch, channel, height, weight = x.size()

        x = x.transpose(2, 3).reshape(-1, weight, height)
        x_flatten = torch.Tensor()
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = math.ceil(height / level)
            stride = math.ceil(height / level)
            pooling = math.floor((kernel_size * level - height + 1) / 2)
            # 选择池化方式
            if self.pool_type == 'max_pool ':
                tensor = F.max_pool1d(x, kernel_size=kernel_size, stride=stride, padding=pooling)
            else:
                tensor = F.avg_pool1d(x, kernel_size=kernel_size, stride=stride, padding=pooling)
            # 展开、拼接
            if i == 0:
                x_flatten = tensor
            else:
                x_flatten = torch.cat([x_flatten, tensor], dim=2)

        return x_flatten.transpose(1, 2).reshape(batch, channel, -1, weight)


if __name__ == '__main__':
    t = torch.ones(size=(5, 5, 5, 6))
    print(SPPLayer(num_levels=2)(t).shape)
