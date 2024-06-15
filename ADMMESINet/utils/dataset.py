# encoding:utf-8
import torch.utils.data as data
import os
import os.path
from scipy.io import loadmat
import re
import numpy as np
import torch
import h5py


class EEGDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.eeg_files = [f for f in os.listdir(root_dir) if f.startswith('data_')]
        # print(self.eeg_files)
        self.eeg_files.sort()
        self.eeg_labels = [re.findall('\d+', f)[0] for f in self.eeg_files]
        # self.lead_field = loadmat(os.path.join(root_dir, 'lead_field.mat'))

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        eeg_file = self.eeg_files[idx]
        eeg_label = self.eeg_labels[idx]
        # eeg_data = loadmat(os.path.join(self.root_dir, eeg_file))
        eeg_data = loadmat(os.path.join(self.root_dir, eeg_file))
        source_data = eeg_data['s_real']
        # print(source_data.shape)
        eeg_data = eeg_data['B']
        # print(eeg_data.shape)

        source_data = torch.Tensor(source_data)
        source_data = torch.Tensor(source_data).unsqueeze(0)# [ n_sources, 1]   ( 6004, 100, 1)
        eeg_data = torch.Tensor(eeg_data)
        eeg_data = torch.Tensor(eeg_data).unsqueeze(0)
        return source_data, eeg_data
