import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as data
    
class real_data(data.Dataset):

    def __init__(self, mode, args, sample_ind=None):
        """
            mode: 'train, 'val', 'test'
            data: a tuple consisting of a numpy array (X) and a list (y). 
                len(y)=X.shape[0]=size of data set/ number of patients/admissions
                X.shape[1:]=(48,76)=(T,d)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Bring in Data
        savepath = args['datadir']
        if mode=='train': data_raw = np.load(savepath+'IHMtrain.npz') 
        if mode=='val': data_raw = np.load(savepath+'IHMval.npz')
        if mode=='test': data_raw = np.load(savepath+'IHMtest.npz')
        
        # Bring in ARF/Shock Labels
        if 'ARF' in args['mode']:
            if mode=='train': new_labels = pd.read_hdf(savepath+'train_ARF.h5','df')
            if mode=='val': new_labels = pd.read_hdf(savepath+'val_ARF.h5','df')
            if mode=='test': new_labels = pd.read_hdf(savepath+'test_ARF.h5','df')
        if 'Shock' in args['mode']:
            if mode=='train': new_labels = pd.read_hdf(savepath+'train_Shock.h5','df')
            if mode=='val': new_labels = pd.read_hdf(savepath+'val_Shock.h5','df')
            if mode=='test': new_labels = pd.read_hdf(savepath+'test_Shock.h5','df')

        # Remove rows without ARF/Shock Labels. Replace data_raw['labels']
        if ('ARF' in args['mode']) | ('Shock' in args['mode']):
            a = data_raw['data'][new_labels['Order'].values,:,:]
            b = new_labels['LABEL'].values
            data_raw = {'data':a, 'labels': b}

        # Subsample for smaller training set sizes (N)
        if (args['N'] > len(data_raw['labels'])) & (mode=='train'): 
            raise ValueError('Chosen N is larger than training set size')
        if (mode=='train') & (args['N']!=0):
            if sample_ind is None:
                print('--New Subsampling Occurring')
                self.sample_ind = np.random.choice(len(data_raw['labels']), size=args['N'], replace=False)
            elif len(sample_ind)!=args['N']:
                raise ValueError('Length of sample_ind != N')
            else: 
                print('--Subsampling (old) Occurring')
                self.sample_ind=sample_ind
            self.data = torch.from_numpy(data_raw['data'][self.sample_ind,:,:]).type(torch.FloatTensor).cuda()
            self.labels = torch.from_numpy(data_raw['labels'][self.sample_ind]).type(torch.LongTensor).cuda()

        else:
            self.data = torch.from_numpy(data_raw['data']).type(torch.FloatTensor).cuda()
            self.labels = torch.from_numpy(data_raw['labels']).type(torch.LongTensor).cuda()
            self.sample_ind = np.arange(len(data_raw['labels']))
                    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        T = self.data.shape[1]
        return self.data[idx,:,:], self.labels[idx].repeat(T)
    