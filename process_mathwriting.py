import os
import torch
import numpy as np

from mathwriting_lib import read_inkml_file, render_ink

INKML_ROOT = './data/mathwriting-2024/mathwriting-2024/train'

images = []
labels = []

inkml_file_paths = []
for root, dirs, files in os.walk(INKML_ROOT):
    for file in files:
        if file.endswith('.inkml'):
            inkml_file_paths.append(os.path.join(root, file))

num_files = len(inkml_file_paths)
print(f"Found {num_files} InkML files.")

num_processed = 0
def process_inkml(inkml_path):
    try:
        ink = read_inkml_file(inkml_path)

        img = render_ink(ink)
        # img = img.resize((64, 64)).convert('L')
        img = img.convert('L')
        img_tensor = torch.tensor(np.array(img)).float()/255.0
        img_tensor = img_tensor.to(torch.float16)
        images.append(img_tensor)

        label = ink.annotations.get('normalizedLabel', ink.annotations.get('label'))
        labels.append(label)

        num_processed  += 1
        if num_processed % 500 == 0:
            print(f"{num_processed} files have been processed. Progress: {num_processed / num_files * 100}%. ")

        return True
    
    except Exception as e:
        print(f"Error processing {inkml_path}: {str(e)}")
        return False
    
for inkml_path in inkml_file_paths:
    process_inkml(inkml_path)

def save_dataset(imgs, lbls, path: str):
  images = torch.stack(imgs)
  torch.save({'images': images, 'labels': lbls}, path)

save_dataset(images, labels, "./data/train.pt")