# coding: UTF-8

import chainer
from chainer import Variable, Chain, optimizers, serializers, datasets
import chainer.links as L
import chainer.functions as F

from chainer.datasets import tuple_dataset
from chainer import training, iterators
from chainer.training import extensions

import numpy as np


class MnistData:
    def __init__(self):
        # MNISTデータの読み込み
        mnist_data = datasets.get_mnist(ndim=3)  # 手書き数字の画像を取得28*28の3次元データ
        self.train_data = mnist_data[0]  # 訓練用のデータ
        self.test_data = mnist_data[1]  # テスト用のデータ

    def train_data(self):
        return self.train_data

    def test_data(self):
        return self.test_data


class MnistChain(Chain):
    def __init__(self):
        super(MnistChain, self).__init__(
            # L.Convolution2D(チャンネル数,フィルタ数,フィルタのサイズ)
            # チャンネル数: RGBの場合 3、今回はモノクロ画像のため、チャンネル数は1
            # フィルタ数: 特徴抽出のためのフィルタ枚数
            # フィルタサイズ: フィルタのピクセル数 5 * 5

            # ここで畳み込み処理の定義。プーリング処理は、predictメソッド内に定義する
            cnn1=L.Convolution2D(1, 15, 5),  # 画像サイズ(1,28,28) → (15,24,24)になる。
            cnn2=L.Convolution2D(15, 40, 5),  # 画像サイズ(15,12,12) → (40,8,8)になる。

            # ここから全結合の定義
            l1=L.Linear(640, 400),  # 入力層は 40 * 4 * 4 = 640
            l2=L.Linear(400, 10)  # 出力は0~9なので、10
        )

    def predict(self, x):
        # プーリング処理の定義
        # F.max_pooling_2d(入力画像,領域のサイズ)

        h1 = F.max_pooling_2d(F.relu(self.cnn1(x)), 2)  # 画像サイズ(15,24,24) → (15,12,12)になる。
        h2 = F.max_pooling_2d(F.relu(self.cnn2(h1)), 2)  # 画像サイズ(40,8,8) → (40,4,4)になる。
        h3 = F.dropout(F.relu(self.l1(h2)))  # 中間層ではドロップアウトを実行する

        return self.l2(h3)

    def __call__(self, x, t):
        # 損失関数の計算結果を返却
        return F.softmax_cross_entropy(self.predict(x), t)  # 損失関数: 交差エントロピー


class MnistClasification:
    def __init__(self):
        self.model = MnistChain()
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.data = MnistData()

    def train(self):
        """
        学習の実行
        :return:
        """
        # 一回の学習で500セットのデータを使う。ミニバッチ
        iterator = iterators.SerialIterator(self.data.train_data, 500)

        # updaterの生成
        updater = training.StandardUpdater(iterator, self.optimizer)

        # 20エポック学習
        trainer = training.Trainer(updater, (20, 'epoch'))

        # プログレスバーで学習状況をプリントアウトする
        trainer.extend(extensions.ProgressBar())

        # 学習の実行
        trainer.run()

    def save_model(self):
        """
        モデルの永続化
        :return:
        """
        serializers.save_npz('mnist.npz', self.model)

    def load_model(self):
        """
        モデルの読み込み
        :return:
        """
        serializers.load_npz('mnist.npz', self.model)

    def correct_answer(self):
        """
        正解率を表示する
        :return:
        """
        correct = 0
        test_data = self.data.test_data

        for i in range(len(test_data)):
            x = Variable(np.array([test_data[i][0]], dtype=np.float32))
            t = test_data[i][1]
            y = self.model.predict(x)

            max_index = np.argmax(y.data)

            if max_index == t:
                correct += 1

        print("Correct:", correct,
              "Total:", len(test_data),
              "Accuracy:", correct / len(test_data) * 100, "%")


if __name__ == '__main__':
    classification = MnistClasification()

    # 学習の実行〜モデルの永続化
    classification.train()
    classification.save_model()

    # # モデルのロード〜テストデータの分類結果表示
    # classification.load_model()
    # classification.correct_answer()
