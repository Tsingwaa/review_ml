import torch
from matplotlib import pyplot as plt

torch.manual_seed(0)
dtype = torch.float

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 3 * pow(x, 2) + 2 + 0.2 * torch.rand(x.size())

plt.scatter(x, y)
plt.show()

w = torch.randn(1, 1, dtype=dtype, requires_grad=True)
b = torch.zeros(1, 1, dtype=dtype, requires_grad=True)
lr = 0.001

for i in range(80000):
    y_pred = x.pow(2).mm(w) + b

    loss = 0.5 * (y_pred - y) ** 2
    total_loss = loss.sum()

    total_loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

    plt.plot(x.numpy(), y_pred.detach().numpy(), 'g-', label='predict')
    plt.scatter(x.numpy(), y.numpy(), color='black', marker='.', label='true')

    plt.xlim(-1, 1)
    plt.ylim(2, 6)
    plt.legend()
    plt.show()
