# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
x = np.linspace(start=-1, stop=1, num=100).reshape(100, 1)
y = 3 * np.power(x, 2) + 2 + 0.2 * np.random.rand(x.size).reshape(100, 1)

plt.scatter(x, y)
plt.show()

w1 = np.random.rand(1, 1)
b1 = np.random.rand(1, 1)

lr = 0.001

y_pred = None
for i in range(10000):
    y_pred = np.power(x, 2) * w1 + b1
    loss = 0.5 * (y_pred - y) ** 2
    total_loss = loss.sum()

    grad_w = np.sum((y_pred - y) * np.power(x, 2))
    grad_b = np.sum(y_pred - y)

    w1 -= lr * grad_w
    b1 -= lr * grad_b

    plt.plot(x, y_pred, 'g-', label='predict')
    plt.scatter(x, y, color='black', marker='.', label='true')
    plt.xlim(-1, 1)
    plt.ylim(2, 6)
    plt.legend()
    plt.show()
    print(w1, b1)

# plt.plot(x, y_pred, 'g-', label='predict')
# plt.scatter(x, y, color='black', marker='.', label='true')
# plt.xlim(-1, 1)
# plt.ylim(2, 6)
# plt.legend()
# plt.show()
# print(w1, b1)
