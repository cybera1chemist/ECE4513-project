import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# import custom libraries
from parameters_settings import *
from my_lib import *



train_set, class_names = load_dataset("./datasets/merged_train.pt")
test_set, class_names = load_dataset("./datasets/merged_test.pt")

# network = Net()
# optimizer = optim.SGD(network.parameters())

# network_state_dict = torch.load('./model.pth')
# network.load_state_dict(network_state_dict)
# optim_state_dict = torch.load('./optimizer.pth')
# optimizer.load_state_dict(optim_state_dict)

if __name__ == 'main':
    pass