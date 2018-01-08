# coding: UTF-8

from chainer import Variable, Chain, optimizers, serializers
import chainer.links as L
import chainer.functions as F

from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import training, iterators

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Iris:
    """
    Irisデータクラス
    """

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

        x_test_v = Variable(x_test)

        train = tuple_dataset.TupleDataset(x_train, t_train)

        return train, x_test_v


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

    def predict(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        h3 = self.l3(h2)
        return h3

    # trainerを用いる場合、callメソッドに誤差を返さないといけない
    def __call__(self, x, t):
        return F.mean_squared_error(self.predict(x), t)


class IrisClassification:
    """
    Irisデータセット分類用クラス
    """

    def __init__(self):
        self.model = IrisChain()
        self.optimizer = optimizers.Adam()  # 最適化アルゴリズムにAdam
        self.optimizer.setup(self.model)  # オプティマイザとモデルの紐付け

    def train(self, train):
        """
        学習用のメソッド
        :param train: TupleDataset
        :return:
        """

        # 一回の学習で30セット使う。ミニバッチ方式
        train_iter = iterators.SerialIterator(train, 30)

        # updaterの生成
        updater = training.StandardUpdater(train_iter, self.optimizer)

        # 5000エポック学習する
        trainer = training.Trainer(updater, (5000, 'epoch'))

        # プログレスバーで学習の進行状況を表示する
        trainer.extend(extensions.ProgressBar())

        # 学習の実行
        trainer.run()

    def save_model(self):
        """
        学習モデルの保存
        :return:
        """
        serializers.save_npz("my_iris.npz", self.model)

    def load_model(self):
        """
        学習モデルのロード
        :return:
        """
        serializers.load_npz("my_iris.npz", self.model)

    def test(self, x_test_v):
        """
        テスト用のメソッド
        :param x_test_v:
        :return:
        """
        self.model.cleargrads()  # モデルの勾配を削除

        y_test_v = self.model.predict(x_test_v)  # テストデータに対して予測を実行

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

    def __init__(self):
        # Irisデータの読み込み
        x, t, n = Iris.load_iris()

        # Irisデータの前処理
        x_train, x_test, t_train, self.t_test = Iris.preproc(x, t)

        # データ形式をVariableに変換する
        self.train, self.x_test_v = \
            Iris.convert_to_variable(x_train, x_test, t_train)

    def test_execute(self):
        """
        テスト用データの検証
        :return:
        """
        iris_classification = IrisClassification()

        # 永続化されたモデルファイルの読み込み
        iris_classification.load_model()

        # テスト
        y_test = iris_classification.test(self.x_test_v)

        # 正解数のカウント
        iris_classification.count_correct(y_test, self.t_test)

    def train_execute(self):
        """
        訓練データによる学習実行
        :return:
        """

        # 学習
        iris_classification = IrisClassification()
        iris_classification.train(self.train)

        # 学習結果を保存
        iris_classification.save_model()


if __name__ == '__main__':
    Main().train_execute()
    Main().test_execute()
