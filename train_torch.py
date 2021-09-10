# Network layer -> Build NN -> forward -> backward -> Optimization

import torch
import torchvision
import sklearn

# Set hyperparameters
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
MOMENTUM = 0.5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Preprocessing
Transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5]),
])

train_datasets = torchvision.datasets.MNIST('./data/', train=True, transform=Transforms, download=True)
test_datasets = torchvision.datasets.MNIST('./data', train=False, transform=Transforms)

train_loader = torch.utils.data.Dataloader(train_datasets, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=TEST_BATCH_SIZE, shuffle=False)


# Visualize source data
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_target) = next(examples)
#
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground truth: {}".format(example_target[i]))
#     plt.xsticks([])
#     plt.ysticks([])


# Build Net
class Net(torch.nn.Module):  # input 1-dim vector
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, n_hidden_1),
            torch.nn.BatchNorm1d(n_hidden_1),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_1, n_hidden_2),
            torch.nn.BatchNorm1d(n_hidden_2),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_2, out_dim),
            torch.nn.BatchNorm1d(out_dim),
        )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout2d(0.2)  # consider which map to be thrown

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        out = self.relu(self.layer3(x))

        return out


model = Net(28 * 28, 100, 100, 10).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM).to(DEVICE)

for epoch in range(1, NUM_EPOCHS + 1):
    print('++++++++++++++++ EPOCH NO.{} +++++++++++++++++++'.format(epoch))
    train_predictions = []
    train_targets = []
    total_loss = 0.

    if epoch % 5 == 0:
        optimizer.para_group[0]['lr'] *= 0.1

    model.train()
    for data, label in train_loader:
        data = data.view(data.size(0), -1).to(DEVICE)
        label = label.to(DEVICE)

        out = model(data)

        optimizer.zero_grad()
        loss = criterion(out, label)  # loss is a scalar
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_predictions.extend(out.max(1)[1])  # max_value, max_index = output.max(1) 横向求最大值及其索引
        train_targets.extend(label)

    train_acc = sklearn.metrics.accuracy_score(train_predictions, train_targets)

    if epoch % 10 == 0:
        model.eval()
        test_predictions = []
        test_targets = []

        for data, label in test_loader:
            data = data.view(data.size(0), -1).to(DEVICE)
            label = label.to(DEVICE)

            out = model(data)

            test_predictions.extend(out.max(1)[1])
            test_targets.extend(label)

        test_acc = sklearn.metrics.accuracy_score(test_predictions, test_targets)
