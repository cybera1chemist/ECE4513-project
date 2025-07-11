import torch
from torch.utils.data import Dataset
import os
from PIL import Image

# import custom settings
from parameters_settings import *

class UnifiedMathDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            label = label_mapping.get(class_dir.lower(), -1)
            if label == -1:
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.samples.append( (img_path, label) )
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label
    
class PreprocessedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_dataset(path):
    data = torch.load(path)
    return PreprocessedDataset(
        data['images'], data['labels']),  data['class_names']