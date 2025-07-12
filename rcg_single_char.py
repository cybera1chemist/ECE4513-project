import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import custom libraries
from parameters_settings import *
from my_lib import *

# Set constants
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

# Load Data
train_set, class_names = load_dataset("./data/merged_train.pt")
test_set, class_names = load_dataset("./data/merged_test.pt")
train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=True)

# Start building the network!

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_dropout = nn.Dropout2d() # Dropout's purpose: prevent overfit
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu( F.max_pool2d(self.conv1(x), 2) )
        x = F.relu( F.max_pool2d(self.conv2_dropout(self.conv2(x)), 2))
        x = x.view(-1, 320) # change the shape of x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# Load network and optimizer
network = Net(len(class_names))
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
network_state_dict = torch.load('./model.pth')
network.load_state_dict(network_state_dict)
optim_state_dict = torch.load('./optimizer.pth')
optimizer.load_state_dict(optim_state_dict)

# Start training!!!

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs+1)]

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch {epoch} [ {batch_idx*len(data)} / {len(train_loader.dataset)}]\tLoss: {loss.item()}')
        train_losses.append(loss.item())
        train_counter.append( (batch_idx*64) + (epoch-1) * len(train_loader.dataset)) 
    torch.save(network.state_dict(), './model.pth')
    torch.save(optimizer.state_dict(), './optimizer.pth')
# train(1)

def test():
    network.eval()
    test_loss = 0
    num_correct = 0
    num_test = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            num_correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= num_test
    test_losses.append(test_loss)
    print(f"Test set: Avg. loss: {test_loss}, Accuracy: {num_correct} / {num_test}")
    return num_correct/num_test
# test()

if __name__ == 'main':
    accuracies = []
    for epoch in range(1, n_epochs+1):
        train(epoch)
        acc = test()
        accuracies.append(acc)

    plt.plot(accuracies)
    plt.show()
