import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import chainer
import chainer.links as L
import chainer.functions as F
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

minibatch = train_iter.next()
