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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics
import itertools
import random
from scipy.stats import entropy


class synth_data2(data.Dataset):

    def __init__(self, args,k_dist=None,d_dist=None):
        
        self.args=args
        self.orig_N=args['N']
        self.new_N=args['N']
        self.k_dist=k_dist
        self.d_dist=d_dist
        if args['T']<=args['k']: print('Uhoh: T<=k')
        
        """Gen X"""
        x_size = args['N']*args['T']*args['d']
        self.x=np.zeros(x_size)
        self.x[np.random.choice(x_size, size=int(x_size/10), replace=False)]=np.random.uniform(size=int(x_size/10))*100
        self.x=np.resize(self.x, (args['N'],args['T'],args['d']))

        """Gen y"""
        if (self.k_dist is None) or (self.d_dist is None): 
            self.k_dist = []
            self.d_dist = []
            for i in range(args['T']):
                # If i<k, we won't evaluate using that timestep therefore it doesn't matter
                if i<args['k']: 
                    self.k_dist.append(np.ones(args['k']))
                    self.d_dist.append(np.ones(args['d']))
                elif i==args['k']: 
                    self.k_dist.append(self.convert_distb(np.random.uniform(size=(args['k']))))
                    self.d_dist.append(self.convert_distb(np.random.uniform(size=(args['d']))))
                else: 
                    delta_t = np.random.uniform(-args['delta'],args['delta'],size=(args['k']))
                    delta_d = np.random.uniform(-args['delta'],args['delta'],size=(args['d']))
                    self.k_dist.append(self.convert_distb(self.k_dist[i-1]+delta_t))
                    self.d_dist.append(self.convert_distb(self.d_dist[i-1]+delta_d))
        
        self.y=np.ones((self.args['N'],self.args['T'],1))
        for i in range(args['T']):
            if i>=args['k']:
                self.y[:,i,0] = np.matmul(np.matmul(self.x[:,i-args['k']:i,:],self.d_dist[i]), self.k_dist[i])

        self.x = torch.from_numpy(self.x).type(torch.FloatTensor).cuda()
        self.y = torch.from_numpy(self.y).type(torch.FloatTensor).cuda()
        
    def convert_distb(self, a):
        # First normalize in case value is negative
        a_min = min(a)
        a_max = max(a)
        a = (a-a_min)/(a_max-a_min)
        # Create distribution
        a_sum = sum(a)
        a = a/a_sum
        return a
        
    def change_N(self, newN):
        self.new_N=newN
        
    def __len__(self):
        return self.new_N
    
    def __getitem__(self, idx):
        return self.x[idx,:,:], self.y[idx,:,:]


def synth_evaluate2(args):
    df = pd.DataFrame({})
    
    for run in range(args['synth_num']):
        eval_data = synth_data2(args)
        np.savez('/data1/jeeheh/'+args['modelname']+'_model'+str(run), k=args['k'], delta=args['delta'],\
                 k_dist=eval_data.k_dist, d_dist=eval_data.d_dist)
    print('synth_num: {}'.format(args['synth_num']))
    return
