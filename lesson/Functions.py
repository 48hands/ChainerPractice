# coding: UTF-8

import chainer
import numpy as np
from chainer import Variable, Chain
import chainer.links as L
import chainer.functions as F

sample_array = np.array([[1, 2, 3]], dtype=np.float32)
x = Variable(sample_array)

y1 = F.sum(x)
print(y1.data)

y2 = F.average(x)
print(y2.data)

y3 = F.max(x)
print(y3.data)

# シグモイド関数
y4 = F.sigmoid(x)
print(y4.data)

# ReLu関数
y5 = F.relu(x)
print(y5.data)

