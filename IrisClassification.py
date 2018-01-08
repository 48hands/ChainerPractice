# coding: UTF-8

import chainer
from chainer import Variable, Chain, optimizers
import chainer.links as L
import chainer.functions as F

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Iris:
    """
    Irisデータクラス
    """

    def __init__(self):
        pass

    @staticmethod
    def load_iris():
        """
        Irisデータセットの読み込み
        :return:
        """
        iris_data = datasets.load_iris()
        x = iris_data.data.astype(np.float32)  # float32型に変換しておく
        t = iris_data.target
        n = t.size

        return x, t, n

    @staticmethod
    def preproc(x, t):
        """
        Irisデータの前処理
        :param x:
        :param t:
        :return:
        """

        # 訓練用とテスト用にでデータを分割しておく
        x_train, x_test, t_train, t_test = train_test_split(
            x, t, test_size=0.5, random_state=0
        )

        # 二重配列を作成する
        n = np.size(t_train)
        t_train_matrix = np.zeros(3 * n).reshape(n, 3).astype(np.float32)

        # 二重配列の正解値の位置を1.0にする。
        for i in range(n):
            t_train_matrix[i, t_train[i]] = 1.0

        t_train = t_train_matrix

        return x_train, x_test, t_train, t_test

    @staticmethod
    def convert_to_variable(x_train, x_test, t_train):
        """
        numpyの形式からVariableに変換するためのメソッド
        :param x_train:
        :param t_train:
        :param x_test:
        :return:
        """
        x_train_v = Variable(x_train)
        t_train_v = Variable(t_train)
        x_test_v = Variable(x_test)

        return x_train_v, x_test_v, t_train_v


class IrisChain(Chain):
    """
    モデルの定義クラス
    Chainを継承
    """

    def __init__(self):
        super(IrisChain, self).__init__(
            l1=L.Linear(4, 6),
            l2=L.Linear(6, 6),
            l3=L.Linear(6, 3),
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        h3 = self.l3(h2)
        return h3


class IrisClassification:
    """
    Irisデータセット分類用クラス
    """

    def __init__(self):
        self.model = IrisChain()
        self.optimizer = optimizers.Adam()  # 最適化アルゴリズムにAdam
        self.optimizer.setup(self.model)  # オプティマイザとモデルの紐付け

    def train(self, x_train_v, t_train_v):
        """
        学習用のメソッド
        :param x_train_v:
        :param t_train_v:
        :return:
        """
        for i in range(10000):
            self.model.cleargrads()  # モデルの勾配をクリア
            y_train_v = self.model(x_train_v)  # モデルから予測値を算出

            # 損失関数による誤差の計算
            loss = F.mean_squared_error(y_train_v, t_train_v)  # 損失関数: 二乗誤差
            # loss = F.softmax_cross_entropy(y_train_v,t_train_v) # 損失関数: 交差エントロピー

            # 誤差の逆伝播
            loss.backward()

            # Optimizerによる重みの更新
            self.optimizer.update()

    def test(self, x_test_v):
        """
        テスト用のメソッド
        :param x_test_v:
        :return:
        """
        self.model.cleargrads()  # モデルの勾配を削除

        y_test_v = self.model(x_test_v)  # テストデータに対して予測を実行

        # y_test_vはVariableオブジェクトのため、そこからデータを取り出す。
        # y_testはnumpyの形式
        y_test = y_test_v.data

        return y_test

    def count_correct(self, y_test, t_test):
        """
        正解数のカウントメソッド
        :param y_test:
        :param t_test:
        :return:
        """
        correct = 0

        # y_testの要素数
        row_count = y_test.shape[0]

        for i in range(row_count):
            max_index = np.argmax(y_test[i, :])  # np.argmax関数は最大の要素のインデックスを返す
            print(y_test[i, :], max_index)
            if max_index == t_test[i]:
                correct += 1
        # 正解率の表示
        print("Correct:", correct, "Total:", row_count,
              "Accuracy:", correct / row_count * 100, "%")


class Main:
    """
    メイン処理記述用のクラス
    """

    def nn_execute(self):
        """
        ニューラルネットワーク実行メソッド
        :return:
        """

        # Irisデータの読み込み
        x, t, n = Iris.load_iris()

        # Irisデータの前処理
        x_train, x_test, t_train, t_test = Iris.preproc(x, t)

        # データ形式をVariableに変換する
        x_train_v, x_test_v, t_train_v = \
            Iris.convert_to_variable(x_train, x_test, t_train)

        # 学習
        iris_classification = IrisClassification()
        iris_classification.train(x_train_v, t_train_v)

        # テスト
        y_test = iris_classification.test(x_test_v)

        # 正解数のカウント
        iris_classification.count_correct(y_test, t_test)


if __name__ == '__main__':
    Main().nn_execute()
