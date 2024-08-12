import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CLEVRDataset(Dataset):
    def __init__(self, img_dir, json_file, obj_file, transform=None, mode='train', img_sz=64):
        """
        Args:
            img_dir (string): Directory with all the images.
            json_file (string): Path to the json file with annotations.
            obj_file (string): Path to the json file with object classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            mode (string): Whether the dataset is for training or testing.
            img_sz (int): Size of the image.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        self.img_sz = img_sz  # Add img_sz attribute
        
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        with open(obj_file, 'r') as f:
            self.objects = json.load(f)
        
        if mode == 'train':
            self.img_names = list(self.data.keys())  # Store all image filenames
        else:
            self.img_names = None  # In test mode, we don't have image names
        
        self.classes = self._get_classes()

    def _get_classes(self):
        # Build a dictionary of class labels from the dataset
        classes = set()
        if self.mode == 'train':
            for labels in self.data.values():
                classes.update(labels)
        else:
            for labels in self.data:
                classes.update(labels)
        return {label: idx for idx, label in enumerate(classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_name = self.img_names[idx]  # Get image filename by index
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            labels = self.data[img_name]  # Get labels by image filename

            # Convert string labels to one-hot encoding
            label_tensor = torch.zeros(len(self.objects))
            for label in labels:
                label_tensor[self.objects[label]] = 1

            if self.transform:
                image = self.transform(image)
            
            return image, label_tensor
        else:
            # For test mode, assume `self.data` is a list of label sets
            labels = self.data[idx]
            label_tensor = torch.zeros(len(self.objects))
            for label in labels:
                label_tensor[self.objects[label]] = 1
            return torch.zeros(3, self.img_sz, self.img_sz), label_tensor  # Return zero tensor as image placeholder
