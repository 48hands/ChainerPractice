# coding: UTF-8

import chainer
from chainer import Variable, Chain, optimizers
import chainer.links as L
import chainer.functions as F
import numpy as np

import matplotlib.pyplot as plt

# 階段関数のデータ
x, t = [], []
for i in np.linspace(-1, 1, 100):
    x.append([i])
    if i < 0:
        t.append([0])
    else:
        t.append([1])

# plt.plot(np.array(x, dtype=np.float32).flatten(),
#          np.array(t, dtype=np.float32).flatten())
# plt.show()


class MyChain(Chain):
    """
    MyChain
    学習モデル
    """

    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(1, 10),
            l2=L.Linear(10, 1),
        )

    def __call__(self, x):
        # sigmoid関数
        h1 = F.sigmoid(self.l1(x))
        return self.l2(h1)


# Variableの記述
# x,tをVariableのオブジェクトとして上書き
x = Variable(np.array(x, dtype=np.float32))
t = Variable(np.array(t, dtype=np.float32))

model = MyChain()

# Optimizerの記述
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)  # モデルのセットアップ

# 学習
y = None
for i in range(100000):

    model.cleargrads()
    y = model(x)

    # 学習過程の表示
    if i % 10000 == 0:
        plt.plot(x.data.flatten(), y.data.flatten())
        plt.title("i = " + str(i))
        plt.show()

    # 損失関数による誤算の計算、平均二乗誤差を採用
    loss = F.mean_squared_error(y, t)
    loss.backward()

    # Optimizerによる重みの更新
    optimizer.update()

# 結果の表示
plt.plot(x.data.flatten(), y.data.flatten())
plt.show("Finish!")
plt.show()
