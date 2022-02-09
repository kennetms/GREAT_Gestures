import torch
import h5py
import numpy as np
import pandas as pd
import pickle
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor, Lambda

# device setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RGBDataset(torch.utils.data.Dataset):
    '''
    __getitem__ return each data, target
        
        data : torch.size([num_frames, RGB channels, image_width, image_height]) # example : torch.Size([20, 3, 224, 224])
        target : one hot vector(torch.tensor) which len is len(label_map)
    '''

    def __init__(self, train, file_path, transform=None, target_transform=None):
        super(RGBDataset, self).__init__()
        self.f = h5py.File(file_path, 'r')

        self.train = train

        self.transform = transform
        self.target_transform = target_transform

        self.train_root = self.f['data/train']
        self.test_root = self.f['data/test']

        self.train_data_list = []
        self.train_label_list = []

        self.test_data_list = []
        self.test_label_list = []

        if train:
            for i in self.train_root:
                self.train_label_list.extend(
                    list(self.train_root[f'{i}/labels']))
                for j in list(self.train_root[i])[:-1]:
                    self.train_data_list.append(f'data/train/{i}/{j}')
        else:
            for i in self.test_root:
                self.test_label_list.extend(
                    list(self.test_root[f'{i}/labels']))
                for j in list(self.test_root[i])[:-1]:
                    self.test_data_list.append(f'data/test/{i}/{j}')

    def __getitem__(self,  idx):

        data_path = self.train_data_list[idx]
        data = self.f[data_path]
        data = torch.tensor(data).permute(0, 3, 1, 2)
        if self.transform:
            data = self.transform(data)

        label = self.train_label_list[idx]
        label = torch.tensor(label, dtype=torch.int64)
        if self.target_transform:
            label = self.target_transform(label)

        return data.to(device), label.to(device)

    def __len__(self):
        return len(self.train_data_list)


def fill_zero(num, l):
    fill_num = str(num).zfill(l)
    return fill_num


def create_datasets(
        hdf5_file_path,
        im_size,  # I set 224
        transform=None,
        target_transform=None):

    with open('label_map.pickle', 'rb') as f:
        label_map = pickle.load(f)

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            # transforms.ToTensor(),#])
            # transforms.Normalize(mean,std)
        ])

    if target_transform is None:
        target_transform = Lambda(lambda y: torch.zeros(len(
            label_map), dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    train_ds = RGBDataset(hdf5_file_path, train=True,
                          transform=transform,
                          target_transform=target_transform,
                          )

    test_ds = RGBDataset(hdf5_file_path, train=False,
                         transform=transform,
                         target_transform=target_transform,
                         )

    return train_ds, test_ds


def create_dataloader(batch_size,
                      hdf5_file_path,
                      transform=None,
                      target_transform=None):

    train_d, test_d = create_datasets(
        hdf5_file_path=hdf5_file_path,
        batch_size=batch_size,
        transform=transform,
        target_transform=target_transform)

    train_dl = torch.utils.data.DataLoader(
        train_d, shuffle=True, batch_size=batch_size)
    test_dl = torch.utils.data.DataLoader(
        test_d, shuffle=False, batch_size=batch_size)

    return train_dl, test_dl
