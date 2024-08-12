
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
def get_key(fp):
    filename = os.path.basename(fp)
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)

    # original
    # filename = fp.split('/')[-1]
    # filename = filename.split('.')[0].replace('frame', '')
    # return int(filename)

class Dataset_Dance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, transform, mode='train', video_len=7, partial=1.0):
        # super().__init__()
        # assert mode in ['train', 'val'], "There is no such mode !!!"
        # if mode == 'train':
        #     self.img_folder     = sorted(glob(os.path.join(root, 'train/train_img/*.png')), key=get_key)
        #     self.prefix = 'train'
        # elif mode == 'val':
        #     self.img_folder     = sorted(glob(os.path.join(root, 'val/val_img/*.png')), key=get_key)
        #     self.prefix = 'val'
        # else:
        #     raise NotImplementedError
        
        # self.transform = transform
        # self.partial = partial
        # self.video_len = video_len
        
        super().__init__()
        assert mode in ['train', 'val'], "There is no such mode !!!"
        img_dir = os.path.join(root, mode, f"{mode}_img")
        label_dir = os.path.join(root, mode, f"{mode}_label")
        self.img_folder = sorted(glob(os.path.join(img_dir, '*.png')), key=get_key)
        self.label_folder = label_dir
        self.transform = transform
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        # path = self.img_folder[index]
        
        # imgs = []
        # labels = []
        # for i in range(self.video_len):
        #     label_list = self.img_folder[(index*self.video_len)+i].split('/')
        #     label_list[-2] = self.prefix + '_label'
            
        #     img_name    = self.img_folder[(index*self.video_len)+i]
        #     label_name = '/'.join(label_list)

        #     imgs.append(self.transform(imgloader(img_name)))
        #     labels.append(self.transform(imgloader(label_name)))
        # return stack(imgs), stack(labels)
        
        imgs = []
        labels = []
        for i in range(self.video_len):
            img_path = self.img_folder[index * self.video_len + i]
            img = self.transform(imgloader(img_path))
            imgs.append(img)
            
            label_name = os.path.basename(img_path)
            label_path = os.path.join(self.label_folder, label_name)
            label = self.transform(imgloader(label_path))
            labels.append(label)

        return stack(imgs), stack(labels)