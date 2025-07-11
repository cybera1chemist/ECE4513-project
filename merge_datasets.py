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

train_symbols = UnifiedMathDataset('./datasets/unprocessed_data/handwritten_numbers_and_signs/train', transform=transform)
test_symbols = UnifiedMathDataset('./datasets/unprocessed_data/handwritten_numbers_and_signs/test', transform=transform)

train_mnist = datasets.MNIST('./datasets/unprocessed_data/', train=True, download=True, transform=transform)
test_mnist = datasets.MNIST('./datasets/unprocessed_data/', train=False, download=True, transform=transform)

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

train_combined = CombinedDataset(train_mnist, train_symbols)
test_combined = CombinedDataset(test_mnist, test_symbols)


    
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

save_dataset(train_combined, "./datasets/merged_train.pt")
save_dataset(test_combined, "./datasets/merged_test.pt")