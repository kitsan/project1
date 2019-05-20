import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer.optimizer_hooks import WeightDecay
from chainer.datasets import TupleDataset
from chainer.datasets import split_dataset_random
from chainer.iterators import SerialIterator
from chainer import Sequential
import matplotlib.pyplot as plt


class Net(chainer.Chain):

    def __init__(self, n_in=4, n_hidden=3, n_out=3):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hidden)
            self.l2 = L.Linear(n_hidden, n_hidden)
            self.l3 = L.Linear(n_hidden, n_out)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)

        return h


net = Net(n_hidden=6)

# Iris データセットの読み込み
x, t = load_iris(return_X_y=True)
x = x.astype('float32')
t = t.astype('int32')

dataset = TupleDataset(x, t)

train_val, test = split_dataset_random(dataset, int(len(dataset) * 0.7), seed=0)
train, valid = split_dataset_random(train_val, int(len(train_val) * 0.7), seed=0)

train_iter = SerialIterator(train, batch_size=4, repeat=True, shuffle=True)

optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
optimizer.setup(net)

for param in net.params():
    if param.name != 'b':  # バイアス以外だったら
        param.update_rule.add_hook(WeightDecay(0.0001))  # 重み減衰を適用

n_batch = 64  # バッチサイズ
n_epoch = 50  # エポック数

# ログ
results_train, results_valid = {}, {}
results_train['loss'], results_train['accuracy'] = [], []
results_valid['loss'], results_valid['accuracy'] = [], []

train_iter.reset()  # 上で一度 next() が呼ばれているため

count = 1

for epoch in range(n_epoch):

    while True:

        # ミニバッチの取得
        train_batch = train_iter.next()

        # x と t に分割
        # データを GPU に転送するために、concat_examples に gpu_id を渡す
        x_train, t_train = chainer.dataset.concat_examples(train_batch)

        # 予測値と目的関数の計算
        y_train = net(x_train)
        loss_train = F.softmax_cross_entropy(y_train, t_train)
        acc_train = F.accuracy(y_train, t_train)

        # 勾配の初期化と勾配の計算
        net.cleargrads()
        loss_train.backward()

        # パラメータの更新
        optimizer.update()

        # カウントアップ
        count += 1

        # 1エポック終えたら、valid データで評価する
        if train_iter.is_new_epoch:

            # 検証用データに対する結果の確認
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x_valid, t_valid = chainer.dataset.concat_examples(valid)
                y_valid = net(x_valid)
                loss_valid = F.softmax_cross_entropy(y_valid, t_valid)
                acc_valid = F.accuracy(y_valid, t_valid)

            # # 結果の表示
            # print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'
            #       'acc (train): {:.4f}, acc (valid): {:.4f}'.format(
            #     epoch, count, loss_train.array.mean(), loss_valid.array.mean(),
            #       acc_train.array.mean(), acc_valid.array.mean()))

            # 可視化用に保存
            results_train['loss'] .append(loss_train.array)
            results_train['accuracy'] .append(acc_train.array)
            results_valid['loss'].append(loss_valid.array)
            results_valid['accuracy'].append(acc_valid.array)

            break


# # 損失 (loss)
# plt.plot(results_train['loss'], label='train')  # label で凡例の設定
# plt.plot(results_valid['loss'], label='valid')  # label で凡例の設定
# plt.legend()  # 凡例の表示
# plt.show()
#
# # 精度 (accuracy)
# plt.plot(results_train['accuracy'], label='train')  # label で凡例の設定
# plt.plot(results_valid['accuracy'], label='valid')  # label で凡例の設定
# plt.legend()  # 凡例の表示
# plt.show()

# テストデータに対する損失と精度を計算
x_test, t_test = chainer.dataset.concat_examples(test,)
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_test = net(x_test)
    loss_test = F.softmax_cross_entropy(y_test, t_test)
    acc_test = F.accuracy(y_test, t_test)

print('test loss: {:.4f}'.format(loss_test.array))
print('test accuracy: {:.4f}'.format(acc_test.array))

# ネットワークの保存
chainer.serializers.save_npz('my_iris.net', net)
