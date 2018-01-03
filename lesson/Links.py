# coding: UTF-8

import chainer
import numpy as np
from chainer import Variable
import chainer.links as L

# LinksのLinear Link
l = L.Linear(3, 2)

print(l.W.data)
print(l.b.data)

# オブジェクトlによりyを計算
# input_array = np.array([[1, 2, 3]], dtype=np.float32)
input_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
x = Variable(input_array)
y = l(x)
print(y.data)

# lの勾配をゼロに初期化(前回の計算の値が残ってしまうため)
l.cleargrads()

# y -> lと遡って微分の計算　
# y.grad = np.ones((1, 2), dtype=np.float32)
y.grad = np.ones((2, 2), dtype=np.float32)
y.backward()
print(l.W.grad)
print(l.b.grad)
