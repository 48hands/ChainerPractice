# coding: UTF-8

import chainer
from chainer import Variable, Chain
from chainer import training, optimizers, iterators
from chainer.training import extensions
from chainer.datasets import tuple_dataset, split_dataset_random
import chainer.functions as F
import chainer.links as L

import re
import numpy as np

import input_data


class LSTMSentenceChain(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, out_size):
        """
        初期化処理
        :param vocab_size: 単語数
        :param embed_size: 埋め込みベクトルサイズ
        :param hidden_size: 隠れ層サイズ
        :param out_size: 出力サイズ
        """
        super(LSTMSentenceChain, self).__init__(
            # encode用のLink関数
            xe=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh=L.LSTM(embed_size, hidden_size),
            hh=L.Linear(hidden_size, hidden_size),
            # classifierの関数
            hy=L.Linear(hidden_size, out_size)
        )

    def __call__(self, x):
        """
        順伝播の計算を行うメソッド
        :param x: 入力値
        :return:
        """

        # エンコード
        x = F.transpose_sequence(x)
        self.eh.reset_state()

        for word in x:
            e = self.xe(word)
            h = self.eh(e)

        y = F.relu(self.hh(h))
        return self.hy(y)


class PreProc:
    """
    データ前処理クラス
    """

    @staticmethod
    def sentence2word(sentence):
        stopwords = ["i", "a", "an", "the", "and", "or", "if", "is", "are", "am", "it", "this", "that", "of", "from",
                     "in",
                     "on"]
        sentence = sentence.lower()
        sentence = sentence.replace('\n', '')
        sentence = re.sub(re.compile(r"[!-\/:-@[-`{-~]"), " ", sentence)
        sentence = sentence.split(" ")
        sentence_words = []

        for word in sentence:
            if (re.compile(r"^.*[0-9]+.*$").fullmatch(word) is not None):  # 数字が含まれるものは除外
                continue
            if word in stopwords:  # ストップワードに含まれるものは除外
                continue
            sentence_words.append(word)

        return sentence_words


if __name__ == '__main__':
    N = len(input_data.data)
    print(N)

    x, t = [], []
    for d in input_data.data:
        x.append(d[0])
        t.append(d[1])

    # 単語辞書
    words = {}
    for sentence in x:
        sentence_words = PreProc.sentence2word(sentence)
        print(sentence_words)

        for word in sentence_words:
            if word not in words:
                words[word] = len(words)
    print(words)

    # 文章を単語ID配列にする
    x_vec = []
    for sentence in x:
        sentence_words = PreProc.sentence2word(sentence)
        sentence_ids = []
        for word in sentence_words:
            sentence_ids.append(words[word])
        x_vec.append(sentence_ids)
    print(x_vec)

    # 文章の長さを揃えるために、-1パディングする（系列を覚えて起きやすくするために前パディングする)
    max_sentence_size = 0
    for sentence_vec in x_vec:
        if max_sentence_size < len(sentence_vec):
            max_sentence_size = len(sentence_vec)
    for sentence_ids in x_vec:
        while len(sentence_ids) < max_sentence_size:
            sentence_ids.insert(0, -1)  # 先頭に追加
    print(max_sentence_size)
    print(x_vec)

    # データセットの作成
    data_x = np.array(x_vec, dtype=np.int32)
    data_t = np.array(t, dtype=np.int32)
    dataset = tuple_dataset.TupleDataset(data_x, data_t)

    # 定数
    EPOCH_NUM = 100
    EMBED_SIZE = 200
    HIDDEN_SIZE = 100
    BATCH_SIZE = 5
    OUT_SIZE = 2

    # モデルの定義
    model = L.Classifier(LSTMSentenceChain(
        vocab_size=len(words),
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        out_size=OUT_SIZE
    ))
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train, test = split_dataset_random(dataset, N - 20)
    train_iter = iterators.SerialIterator(train, BATCH_SIZE)
    test_iter = iterators.SerialIterator(test, BATCH_SIZE, shuffle=False, repeat=False)

    updater = training.StandardUpdater(train_iter, optimizer)

    trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.LogReport(trigger=(1, "epoch"), log_name='TextClassificationRNN.log'))
    trainer.extend(extensions.PrintReport(
        ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy",
         "elapsed_time"]))  # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間

    trainer.run()
