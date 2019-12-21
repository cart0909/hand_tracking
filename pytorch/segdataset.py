import os
import os.path as osp
import pickle
import torch
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
from torchvision import transforms

class SegDataset(Dataset):
    def __init__(self, root, is_train=True, output_size=None, transform=None):
        super().__init__()
        if is_train:
            self.set_name = 'training'
        else:
            self.set_name = 'evaluation'

        self.output_size = output_size
        self.transform = transform

        self.root = osp.join(root, self.set_name)

        with open(osp.join(self.root, 'anno_%s.pickle' % self.set_name), 'rb') as fi:
            self.anno_all = pickle.load(fi)
    
    def __len__(self):
        return len(self.anno_all)

    def __getitem__(self, i):
        img_filename = osp.join(self.root, 'color', '%.5d.png' % i)
        msdk_filename = osp.join(self.root, 'mask', '%.5d.png' % i)
        img = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(msdk_filename, cv2.IMREAD_UNCHANGED)
        new_mask = np.zeros(mask.shape)
        new_mask[mask > 1] = 1

        if self.output_size is not None:
            img = cv2.resize(img, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
            new_mask = cv2.resize(new_mask, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)

        if self.transform is not None:
            img = self.transform(img)
            new_mask = torch.from_numpy(new_mask).type(torch.long)

        return img, new_mask

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = SegDataset('.', output_size=256, transform=transform)
    img, mask = dataset[0]
    print(img.dtype)
    print(mask.dtype)