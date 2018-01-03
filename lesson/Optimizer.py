# coding: UTF-8

import chainer
import numpy as np

import chainer.links as L
import chainer.functions as F
from chainer import Variable, Chain, optimizers


# モデルをクラスで記述
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(1, 2),
            l2=L.Linear(2, 1),
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return self.l2(h)


if __name__ == '__main__':
    # Optimizerの記述
    model = MyChain()
    optimizer = optimizers.SGD()
    # optimizer = optimizers.Adam()
    optimizer.setup(model)

    # Optimizerの実行
    input_array = np.array([[1]], dtype=np.float32)
    answer_array = np.array([[1]], dtype=np.float32)
    x = Variable(input_array)
    t = Variable(answer_array)  # 教師データ

    # モデルの勾配をクリアする
    model.cleargrads()
    # モデルからyを求める
    y = model(x)

    # 二乗誤差を求める
    loss = F.mean_squared_error(y, t)
    # 誤差の逆伝播
    loss.backward()

    print(model.l1.W.data)

    # Optimizerで重みとバイアスの更新
    optimizer.update()

    print(model.l1.W.data)
