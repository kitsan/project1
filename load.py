import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Sequential
import matplotlib.pyplot as plt

# Iris データセットの読み込み
x, t = load_iris(return_X_y=True)
x = x.astype('float32')
t = t.astype('int32')
x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.3, random_state=0)
x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=0)

# net としてインスタンス化
n_input = 4
n_hidden = 6
n_output = 3

loaded_net = Sequential(
    L.Linear(n_input, n_hidden), F.relu,
    L.Linear(n_hidden, n_hidden), F.relu,
    L.Linear(n_hidden, n_output)
)

chainer.serializers.load_npz('my_iris.net', loaded_net)

with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_test = loaded_net(x_test)

pred = []

for i in range(len(t_test)):
    pred.append(np.argmax(y_test[i, :].array))

cm = confusion_matrix(t_test, pred)
print(cm)
