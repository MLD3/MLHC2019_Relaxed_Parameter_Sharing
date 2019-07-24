import argparse
import numpy as np
import torch
import pickle
import os

import LSTM11q_datagen as datagen 
import LSTM11q_functions as functions
import LSTM11q_models as models 

'''Settings'''
parser=argparse.ArgumentParser()

parser.add_argument('--runname',type=str,required=True,help='identifier used to name all output files') 
parser.add_argument('--genrunname',type=str,default='DEFAULT',help='allows use of subsample used in run different from the current runname')
parser.add_argument('--model', type=str,required=True, nargs="+", help='models to train')
parser.add_argument('--synthstart',type=bool,default=False, help='generates data. set to True on first run or provide genrunnum with pre-existing data.')

parser.add_argument('--cuda',type=int,default=0, help='which GPU device to run code on')
parser.add_argument('--budget',type=int,default=40, help='number of random initializations during hyperparameter search')
parser.add_argument('--epochs',type=int,default=30, help='max number of epochs')
parser.add_argument('--batch_size',type=int,default=100)
parser.add_argument('--synth_num',type=int,default=5,help='Number of synthetic datasets to generate') 

parser.add_argument('--T',type=int,default=30, help='total number of time steps')
parser.add_argument('--d',type=int,default=3, help='dimension of input')
parser.add_argument('--N',type=int,default=1000, help='size of training/val/test set')
parser.add_argument('--output_size',type=int,default=1)
parser.add_argument('--delta',type=float,default=.05, help='controls the amount of change between temporally adjacent tasks.')
parser.add_argument('--l',type=int,default=10, help="number of previous time steps to use in synthetic task")

parser.add_argument('--te_size',type=int,default=16, help='hyperparameter for temporal embedding for LSTM+TE. must be even number.')
parser.add_argument('--te_base',type=float,default=10000.0,help='hyperparameter for temporal embedding for LSTM+TE')
parser.add_argument('--savedir',type=str,default='/data1/jeeheh/',help='directory where results will be saved')


args=parser.parse_args()
args=vars(args)
if args['genrunname']=='DEFAULT': args['genrunname']=args['runname']


'''Cuda'''
args['use_cuda'] = torch.cuda.is_available()
if args['use_cuda']:
    torch.cuda.set_device(args['cuda'])
    print('Run on cuda: {}'.format(torch.cuda.current_device()))

'''Create Synth Dataset'''
if args['synthstart']:
    datagen.synth_evaluate2(args)


'''Test Models on the Synthetic Datasets'''
m_list = []
for m in args['model']:
    if m=='LSTM': m_list.append((m,models.LSTM))
    elif 'nidLSTM' in m: m_list.append((m,models.nidLSTM))
    elif m=='LSTMTE': m_list.append((m,models.LSTMTE))
    elif m=='HyperLSTM': m_list.append((m,models.HyperLSTM))
    elif m=='LSTMT': m_list.append((m,models.LSTMT))
    elif m=='mow': m_list.append((m,models.mixLSTM))
    else: print('Model {} Not Recognized'.format(m))
        
functions.synth_data_search(args, datagen.synth_data2, m_list)

