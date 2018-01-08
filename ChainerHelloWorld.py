# coding: UTF-8

import chainer
from chainer import Variable, Chain, optimizers, serializers
import chainer.links as L
import chainer.functions as F

from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import training, iterators

import matplotlib.pyplot as plt
import numpy as np


class StairsData:

    def __init__(self):
        x, t = [], []
        for i in np.linspace(-1, 1, 100):
            x.append([i])
            if i < 0:
                t.append([0])
            else:
                t.append([1])

        self.x = np.array(x, dtype=np.float32)
        self.t = np.array(t, dtype=np.float32)

    def train(self):
        train = tuple_dataset.TupleDataset(self.x, self.t)
        return train

    @staticmethod
    def data():
        """
        階段関数のデータ
        :return:
        """


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

    def predict(self, x):
        h1 = F.sigmoid(self.l1(x))  # 活性化関数はsigmoid関数
        h2 = self.l2(h1)
        return h2

    def __call__(self, x, t):
        return F.mean_squared_error(self.predict(x), t)


class HelloWorldClassification:

    def __init__(self):
        self.model = MyChain()
        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.model)
        self.data = StairsData()

    def train(self):
        # 一回の学習で20セットを利用する。ミニバッチ学習
        train_iter = iterators.SerialIterator(self.data.train(), 20)

        # updaterの生成
        updater = training.StandardUpdater(train_iter, self.optimizer)

        # 20000エポック学習する
        trainer = training.Trainer(updater, (20000, 'epoch'))

        # 学習状況をプログレスバーで出力
        trainer.extend(extensions.ProgressBar())

        # 学習の実行
        trainer.run()

    def result(self):
        y = self.model.predict(self.data.x)
        plt.plot(self.data.x.flatten(), y.data.flatten())
        plt.show("Finish!")
        plt.show()


if __name__ == '__main__':
    classification = HelloWorldClassification()
    classification.train()
    classification.result()
