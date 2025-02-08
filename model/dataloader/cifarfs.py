import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
IMAGE_PATH1 = osp.join(ROOT_PATH, 'data/CIFAR-FS/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/CIFAR-FS/split')
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')
split_map = {'train':IMAGE_PATH1, 'val':IMAGE_PATH1, 'test':IMAGE_PATH1}

def identity(x):
    return x

def get_transforms(size):
    
    normalization = transforms.Normalize(np.array([x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]),
                                            np.array([x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]))
    
    data_transforms = transforms.Compose([
                        transforms.RandomCrop(size, padding=4),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        transforms.RandomHorizontalFlip(),
                        # lambda x: np.asarray(x),
                        transforms.ToTensor(),
                        normalization])
    
    return data_transforms


class CIFARFS(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        self.data, self.label = self.parse_csv(csv_path, setname)
        self.num_class = len(set(self.label))

        image_size = 32
        self.transform = get_transforms(image_size)

    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(split_map[setname], name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append( path )
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        image = self.transform(Image.open(data).convert('RGB'))
        return image, label

