from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 196)
        self.fc3 = nn.Linear(196, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                           download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

model = Net()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

n_epochs = 20

model.train()

for epoch in range(n_epochs):
    train_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)

        ones = torch.sparse.torch.eye(10)
        target_expand = ones.index_select(0, target)

        loss = criterion(output, target_expand)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        for i in range(128):
            if i >= len(target):
                break

            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    train_loss = train_loss / len(train_loader)

    print('Epoch: {} \tloss: {:.4f}, accuracy: {:.4f}'.format(
        epoch+1, train_loss, np.sum(class_correct) / np.sum(class_total)
    ))

test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()

for data, target in test_loader:
    output = model(data)

    ones = torch.sparse.torch.eye(10)
    target_expand = ones.index_select(0, target)

    loss = criterion(output, target_expand)

    test_loss += loss.item()
    _, pred = torch.max(output, 1)
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))

    for i in range(128):
        if i >= len(target):
            break

        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss / len(test_loader)
print('Test loss:', test_loss)
print('Test accuracy:', np.sum(class_correct) / np.sum(class_total))
