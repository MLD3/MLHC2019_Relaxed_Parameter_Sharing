import argparse
import numpy as np
import torch, os
import pickle
import os

import LSTM11s_datagen as datagen 
import LSTM11s_functions as functions 
import LSTM11s_models as models 

'''Settings'''
parser=argparse.ArgumentParser()

parser.add_argument('--runname',type=str,required=True,help='identifier used to name all output files') 
parser.add_argument('--mode',type=str,required=True,help='choose ARF or Shock')
parser.add_argument('--model',type=str,required=True,help='model to train')

parser.add_argument('--genrunname',type=str,default='DEFAULT',help='allows use of subsample used in run different from the current runname')
parser.add_argument('--cuda',type=int,default=0, help='which GPU device to run code on')
parser.add_argument('--T',type=int,default=48, help='total number of time steps')
parser.add_argument('--d',type=int,default=76, help='dimension of input')
parser.add_argument('--num_classes',type=int,default=2, help='number of output classes')
parser.add_argument('--N',type=int,default=0,help='Size of training set. If not set to zero, indicates that training set should be subsampled.') 

parser.add_argument('--budget',type=int,default=20,help='number of random initializations during hyperparameter search') 
parser.add_argument('--epochs',type=int,default=30,help='max number of epochs')
parser.add_argument('--share_num',type=int,default=2,help="share number for moo and mow")
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--shiftLSTMk',type=int,default=0,help='number of LSTM cells learned')
parser.add_argument('--realstart',type=bool,default=False,help='if set to True, redraws the subsample instead of loading existing subsample from run genrunname')
parser.add_argument('--te_size',type=int,default=24,help='hyperparameter for temporal embedding for LSTM+TE') #must be even
parser.add_argument('--te_base',type=float,default=10000.0,help='hyperparameter for temporal embedding for LSTM+TE')

parser.add_argument('--savedir',type=str,default='save/',help='directory where results will be saved')
parser.add_argument('--datadir',type=str,default='/data1/jeeheh/mimic/mimic3models/in_hospital_mortality/joh data extraction/',help='location of data (the h5 files generated in savedata.py)')


args=parser.parse_args()
args=vars(args)

if args['genrunname']=='DEFAULT': args['genrunname']=args['runname']
if ('ARF' not in args['mode']) & ('Shock' not in args['mode']):
    raise ValueError('Real data task not set via mode.')

'''Cuda'''
args['use_cuda'] = torch.cuda.is_available()
if args['use_cuda']:
    torch.cuda.set_device(args['cuda'])
    print('Run on cuda: {}'.format(torch.cuda.current_device()))
    
'''Test Models on Datasets'''
if args['model']=='LSTM':
    functions.real_data_search3(args, datagen.real_data,models.LSTM)
if 'shiftLSTM' in args['model']: 
    functions.real_data_search3(args, datagen.real_data,models.nidLSTM)
if args['model']=='mow': functions.real_data_search3(args, datagen.real_data,models.mixLSTM)
if args['model']=='HyperLSTM': functions.real_data_search3(args, datagen.real_data, models.HyperLSTM)
if args['model']=='LSTMT': functions.real_data_search3(args, datagen.real_data,models.LSTMT)