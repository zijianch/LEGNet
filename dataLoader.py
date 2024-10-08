import torch
import numpy as np
import torch.nn.functional as F
import torch.nn
import os.path
import scipy
from scipy import io
import sys
import random
from datetime import datetime

def train_val_split(num_of_sub, fold, val_index):
    '''
    Args:
        num_of_sub (int): number of subjects
        fold (int): number of folds
        val_index (int): index of the validation set, ranging from 1 to fold (not starting from 0)
    Returns:
        train_subj_idx (list): list of indices of subjects in the training set
        val_subj_idx (list): list of indices of subjects in the validation set
    '''
    random.seed(0)
    all_subj_idx = list(range(num_of_sub))
    #random.shuffle(all_subj_idx)
    val_subj_idx = all_subj_idx[(val_index-1)*num_of_sub//fold:val_index*num_of_sub//fold]
    train_subj_idx = list(set(all_subj_idx) - set(val_subj_idx))
    return train_subj_idx, val_subj_idx

def augment_sample(sample):

    # creating artificial language deficits on HCP data based on the 'pos' matrix

    '''
    Args:
        sample (tuple): (input, label, position matrix)
    Returns:
        sample (tuple): (input, label, position matrix)
    '''

    connMat = sample[0]
    score = sample[1]
    pos = sample[2] # diagonal matrix, shape: (246, 246). (i,i)-element: percentage of spare gray matter in region i (range: 0~1)

    num_regions = pos.shape[0]

    #-----------------------------------------#
    # Augmentation 1: score 
    #-----------------------------------------#

    # compute total percentage of spare gray matter on the left hemisphere
    total_psg = (np.sum(np.diag(pos)) - num_regions/2) / (num_regions/2)
    # augment the score according to the total percentage of spare gray matter
    # the higher the psg, the higher the score
    score = total_psg ** 5 * score # we amplify the effect of psg on the score
    
    #-----------------------------------------#
    # Augmentation 2: connMat
    #-----------------------------------------#

    for i in range(num_regions):

        np.random.seed(i) # bind seed to region i for reproducibility
        psg = pos[i,i]

        if psg < 1:
            # connectivities between region i and other regions should be deminished
            connMat[0,i,:] = connMat[0,i,:] * psg
            connMat[0,:,i] = connMat[0,:,i] * psg

            # create base random noise (of size (1,246)) for one region
            noise = np.random.normal(0, 0.05, (1, num_regions))

            # add noise to the related regions
            connMat[0,i,:] = connMat[0,i,:] + noise
            connMat[0,:,i] = connMat[0,:,i] + noise

            # connectivity values should be within [0,1]
            connMat[0,i,:] = np.clip(connMat[0,i,:], 0, 1)
            connMat[0,:,i] = np.clip(connMat[0,:,i], 0, 1)

            # diagonal element should still be 1
            connMat[0,i,i] = 1

    return connMat, score, pos


class HCPdataset(torch.utils.data.Dataset):

    # working fine. date tested: 2024-04-13

    def __init__(self, 
                 transform=False, 
                 root_dir='/project', 
                 section = 'train', 
                 fold=10, 
                 val_index=1,
                 augment=True):
        
        '''
        File structure:
        root_dir
        ├── data
        │   ├── HCP
        │   │   ├── xAll.mat
        │   │   ├── yAll.mat
        │   │   └── posAll.mat
    
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
            root_dir (string): Project directory with all the data. Data folder should be located at root_dir/data
            section (string): 'train' or 'validation'
            fold (int): number of folds
            val_index (int): index of the validation set, ranging from 1 to fold (not starting from 0)
        Returns:
            self.X (torch.Tensor): input data, shape: torch.Size([num_of_sub, 1, 246, 246])
            self.Y (torch.Tensor): label, shape: torch.Size([num_of_sub])
            self.pos (torch.Tensor): position matrix, shape: torch.Size([num_of_sub, 246, 246])
        '''

        self.transform = transform
        self.augment = augment

        connMat_file = os.path.join(root_dir, 'data/HCP/xAll.mat')
        label_file = os.path.join(root_dir, 'data/HCP/yAll.mat')
        pos_file = os.path.join(root_dir, 'data/HCP/posAll.mat')

        x = scipy.io.loadmat(connMat_file)['x'] # x.shape: (705, 246, 246)
        y = scipy.io.loadmat(label_file)['y'] # y.shape: (705, 1)
        pos = scipy.io.loadmat(pos_file)['pos'] # pos.shape: (705, 246, 246)

        train_subj_idx, val_subj_idx = train_val_split(len(x), fold, val_index)

        if section == 'train':
            x = x[train_subj_idx]
            y = y[train_subj_idx]
            pos = pos[train_subj_idx]
        else:
            x = x[val_subj_idx]
            y = y[val_subj_idx]
            pos = pos[val_subj_idx]

        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32)) # self.X.shape: torch.Size([635, 1, 246, 246])
        self.pos = torch.FloatTensor(pos.astype(np.float32)) # self.pos.shape: torch.Size([635, 246, 246])
        self.Y = torch.tensor(y, dtype=torch.float32).squeeze() # self.Y.shape: torch.Size([635])

        print('HCP dataset loaded. Section: {}. Number of subjects: {}. Augmentation: {}. Fold: {}/{}.'.format(section, len(self.X), augment, val_index, fold))

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        sample = (self.X[idx], self.Y[idx], self.pos[idx])

        if self.augment:
            sample = augment_sample(sample)

        if self.transform:
            sample = self.transform(sample[0]), sample[1],  sample[2]
        return sample
    


class DS1dataset(torch.utils.data.Dataset):

    def __init__(self, 
                 transform=False, 
                 root_dir='/project', 
                 section = 'train', 
                 data_split = 1,
                 fold=10,
                 val_index=1):
        
        '''
        File structure:
        root_dir
        ├── data
        │   ├── DS1
        │   │   ├── x_1.mat
        │   │   ├── y1.mat
        │   │   ├── pos_1.mat
        │   │   ├── ...
        │   │   ├── x_10.mat
        │   │   ├── y10.mat
        │   │   └── pos_10.mat

        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
            root_dir (string): Project directory with all the data. Data folder should be located at root_dir/data
            section (string): 'train' or 'validation'
            fold (int): number of folds
            val_index (int): index of the validation set, ranging from 1 to fold (not starting from 0)
            data_split (int): 1,...,10. Each number corresponds to a different data assignment plan.
        Returns:
            self.X (torch.Tensor): input data, shape: torch.Size([num_of_sub, 1, 246, 246])
            self.Y (torch.Tensor): label, shape: torch.Size([num_of_sub])
            self.pos (torch.Tensor): position matrix, shape: torch.Size([num_of_sub, 1, 246, 246])

        Note: for DS1 data, we fixed the subject assignment in each fold (so they have similar WAB score distribution).
        '''
        
        self.transform = transform

        # read in all data
        x_allfolds = []
        y_allfolds = []
        pos_allfolds = []        
        for fold_idx in range(1, fold+1):
            # the path of the data files should be 'root_dir/data/DS1/data_split/xxxxx'
            connMat_file = os.path.join(root_dir, 'data/DS1/' + str(data_split) + '/x_' + str(fold_idx) + '.mat')
            label_file = os.path.join(root_dir, 'data/DS1/' + str(data_split) + '/y' + str(fold_idx) + '.mat')
            pos_file = os.path.join(root_dir, 'data/DS1/' + str(data_split) + '/pos_' + str(fold_idx) + '.mat')

            x = scipy.io.loadmat(connMat_file)['x']
            y = scipy.io.loadmat(label_file)['y']
            pos = scipy.io.loadmat(pos_file)['pos']

            x_allfolds.append(x)
            y_allfolds.append(y)
            pos_allfolds.append(pos)
        
        # split data based on "train" or "validation"
        if section == 'train':
            x = np.concatenate(x_allfolds[0:val_index-1] + x_allfolds[val_index:], axis=0)
            y = np.concatenate(y_allfolds[0:val_index-1] + y_allfolds[val_index:], axis=0)
            pos = np.concatenate(pos_allfolds[0:val_index-1] + pos_allfolds[val_index:], axis=0)
        else: 
            x = x_allfolds[val_index-1]
            y = y_allfolds[val_index-1]
            pos = pos_allfolds[val_index-1]

        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32))
        self.pos = torch.FloatTensor(np.expand_dims(pos, 1).astype(np.float32))
        self.Y = torch.tensor(y, dtype=torch.float32).squeeze()

        print('DS1 dataset loaded. Section: {}. Number of subjects: {}. Fold: {}/{}.'.format(section, len(self.X), val_index, fold))

        mod_time = os.path.getmtime(connMat_file)
        readable_time = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print('Last modified time of the data file: {}'.format(readable_time))
        print(f"Fetched data from {connMat_file}")

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx], self.Y[idx], self.pos[idx]
        if self.transform:
            sample = self.transform(sample[0]), sample[1],  sample[2]
        return sample



class DS2dataset(torch.utils.data.Dataset):

    def __init__(self, 
                 transform=False, 
                 root_dir='/project'):
        
        '''
        File structure:
        root_dir
        ├── data
        │   ├── DS2
        │   │   ├── x.mat
        │   │   ├── y.mat
        │   │   └── pos.mat

        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
            root_dir (string): Project directory with all the data. Data folder should be located at root_dir/data
        Returns:
            self.X (torch.Tensor): input data, shape: torch.Size([num_of_sub, 1, 246, 246])
            self.Y (torch.Tensor): label, shape: torch.Size([num_of_sub])
            self.pos (torch.Tensor): position matrix, shape: torch.Size([num_of_sub, 1, 246, 246])

        Note: DS2 data is purely for testing purposes, so we don't need to split the data .
        '''
        
        self.transform = transform

        connMat_file = os.path.join(root_dir, 'data/DS2/x.mat')
        label_file = os.path.join(root_dir, 'data/DS2/y.mat')
        pos_file = os.path.join(root_dir, 'data/DS2/pos.mat')

        x = scipy.io.loadmat(connMat_file)['x']
        y = scipy.io.loadmat(label_file)['y']
        pos = scipy.io.loadmat(pos_file)['pos']

        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32))
        self.pos = torch.FloatTensor(np.expand_dims(pos, 1).astype(np.float32))
        self.Y = torch.tensor(y, dtype=torch.float32).squeeze()

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx], self.Y[idx], self.pos[idx]
        if self.transform:
            sample = self.transform(sample[0]), sample[1],  sample[2]
        return sample


        
        

        
        
        

