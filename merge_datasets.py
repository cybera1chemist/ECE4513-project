import torch
from torchvision import datasets
from torchvision import transforms as tf
from torch.utils.data import Dataset, ConcatDataset

# import custom libraries
from parameters_settings import *
from my_lib import *

transform = tf.Compose([
    tf.Grayscale(),
    tf.Resize((28, 28)),
    tf.ToTensor(),
    tf.Normalize( (0.1307, ), (0.3801, ) ) 
])

class CombinedDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.length = [len(d) for d in datasets]
        self.total_length = sum(self.length)
    def __len__(self):
        return self.total_length
    def __getitem__(self, idx):
        for i, len in enumerate(self.length):
            if idx < len:
                return self.datasets[i][idx]
            idx -= len
        raise IndexError

def save_dataset(combined, path):
    images = []
    labels = []
    for img, label in combined:
        images.append(img)
        labels.append(label)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    torch.save( {
        'images': images,
        'labels': labels, 
        'class_names': class_names
    }, path)
    print(f"Dataset saved at {path}.")

def merge_two_datasets(existing_data, new_data, train=True):
    combined_data = CombinedDataset(existing_data, new_data)
    if train:
        save_dataset(combined_data, "./data/merged_train.pt")
    else:
        save_dataset(combined_data, "./data/merged_test.pt")

# Load EMNIST data

from torchvision.datasets import EMNIST

def filter_letters(data):
    letter_indices = [i for i, (_,label) in enumerate(data) if label>=10]
    return torch.utils.data.Subset(data, letter_indices)

def load_EMNIST():
    pass
#     emnist_train = EMNIST(root='./data/emnist', split='letters', train=True, transform=transform)
#     emnist_test = EMNIST(root='./data/emnist', split='letters', train=False, transform=transform)
#     letter_train = filter_letters(emnist_train)
#     letter_test = filter_letters(emnist_test)
#     for dataset in [letter_train, letter_test]:
#         for i in range(len(dataset)):
#             img, label = dataset[i]
#             dataset.dataset.targets[dataset.indices[i]] = label-10
#     return letter_train, letter_test

letter_train, letter_test = load_EMNIST()

# Load Existing Data
existing_train = torch.load("./data/merged_train.pt")
existing_test = torch.load("./data/merged_test.pt")

if __name__ == 'main':
    merge_two_datasets(existing_train, letter_train)
    merge_two_datasets(existing_test, letter_test)