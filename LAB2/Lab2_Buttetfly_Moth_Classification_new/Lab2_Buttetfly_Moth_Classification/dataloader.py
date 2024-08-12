import pandas as pd
from PIL import Image
from torch.utils import data
import os
import torch
from torchvision import transforms

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('./dataset/train.csv')
    elif mode == 'test':
        df = pd.read_csv('./dataset/test.csv')
    else:
        df = pd.read_csv('./dataset/valid.csv')
    
    path = df['filepaths'].tolist()
    labels_text = df['labels'].tolist()
    label_id = df['label_id'].tolist()
    
    return path, labels_text, label_id

class ButterflyMothLoader(data.Dataset):
    def __init__(self, root, mode, transform=None):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label_text, self.label_id = getData(mode)
        self.mode = mode
        self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))  
        
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        
        img_path = os.path.join(self.root, self.img_name[index])
        # PIL load the images
        img = Image.open(img_path).convert('RGB')
        
        label = self.label_id[index]
        
        img = self.transform(img)
        
        return img, label