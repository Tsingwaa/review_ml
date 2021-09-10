import numpy as np


def cross_entropy(y, y_hat):
    # n = 1e-6
    # return -np.sum(y * np.log(y_hat + n) + (1 - y) * np.log(1 - y_hat + n), axis=1)
    assert y.shape == y_hat.shape
    res = -np.sum(np.nan_to_num(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))
    return round(res, 3)


def softmax(y):
    y_shift = y - np.max(y, axis=1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_exp_sum = np.sum(y_exp, axis=1, keepdims=True)
    return y_exp / y_exp_sum


if __name__ == "__main__":
    y = np.array([1, 0, 0, 1]).reshape(-1, 1)
    y_hat = np.array([1, 0.4, 0.5, 0.1]).reshape(-1, 1)
    print(cross_entropy(y, y_hat))
    # y = np.array([[1,2,3,4],[1,3,4,5],[3,4,5,6]])
    # print(softmax(y))
