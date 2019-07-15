import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
import os

class IHM_data(data.Dataset):

    def __init__(self, mode, args, sample_ind=None):
        """
            mode: 'train, 'val', 'test'
            data: a tuple consisting of a numpy array (X) and a list (y). 
                len(y)=X.shape[0]=size of data set/ number of patients/admissions
                X.shape[1:]=(48,76)=(T,d)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if mode=='train': data_raw =np.load(os.path.join(args['datadir'],'IHM_train.npz')) 
        if mode=='val': data_raw =np.load(os.path.join(args['datadir'],'IHM_val.npz'))
        if mode=='test': data_raw =np.load(os.path.join(args['datadir'],'IHM_test.npz'))
                
        if (mode=='train') & (args['N']!=0):
            if sample_ind is None:
                print('New subsampling')
                self.sample_ind = np.random.choice(len(data_raw['labels']), size=args['N'], replace=False)
            elif len(sample_ind)!=args['N']:
                print('ERROR: length of sample_ind != N')
            else: 
                self.sample_ind=sample_ind
                print('Prev subsampling applied')
            self.data = torch.from_numpy(data_raw['data'][self.sample_ind,:,:]).type(torch.FloatTensor).cuda()
            self.labels = torch.from_numpy(data_raw['labels'][self.sample_ind]).cuda()

        else:
            self.data = torch.from_numpy(data_raw['data']).type(torch.FloatTensor).cuda()
            self.labels = torch.from_numpy(data_raw['labels']).cuda()
            self.sample_ind = 'placeholder'
                    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        T = self.data.shape[1]
        return self.data[idx,:,:], self.labels[idx].repeat(T)
