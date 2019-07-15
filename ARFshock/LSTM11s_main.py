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

parser.add_argument('--modelname',type=str,required=True) 
parser.add_argument('--genmodelname',type=str,default='DEFAULT')
parser.add_argument('--mode',type=str,required=True)
parser.add_argument('--cuda',type=int,default=0)

parser.add_argument('--budget',type=int,default=20) #HP search budget
parser.add_argument('--epochs',type=int,default=30)

parser.add_argument('--T',type=int,default=48)
parser.add_argument('--d',type=int,default=76)
parser.add_argument('--num_classes',type=int,default=2)
parser.add_argument('--N',type=int,default=0) 
parser.add_argument('--synth_num',type=int,default=10) ## if change - gotta also change synthnum_batch
parser.add_argument('--synthnum_batch',nargs='+',type=int,default=[0,10]) 
parser.add_argument('--k',type=int,default=5)
parser.add_argument('--delta',type=float,default=.05)
parser.add_argument('--share_num',type=int,default=2,help="share number for moo and mow")

parser.add_argument('--hidden_size',type=int,default=20)
parser.add_argument('--hyp_hidden_size',type=int,default=20)
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--nidLSTM',type=int,default=0)
parser.add_argument('--realstart',type=bool,default=False)
parser.add_argument('--te_size',type=int,default=24) #must be even
parser.add_argument('--te_base',type=float,default=10000.0)
parser.add_argument('--verbose',type=bool,default=False)
parser.add_argument('--savedir',type=str,default='save/')
parser.add_argument('--datadir',type=str,default='/data1/jeeheh/mimic/mimic3models/in_hospital_mortality/joh data extraction/')


args=parser.parse_args()
args=vars(args)

if args['genmodelname']=='DEFAULT': args['genmodelname']=args['modelname']
if ('IHM' not in args['mode']) & ('ARF' not in args['mode']) & ('Shock' not in args['mode']):
    raise ValueError('Real data task not set via mode.')

'''Cuda'''
args['use_cuda'] = torch.cuda.is_available()
if args['use_cuda']:
    torch.cuda.set_device(args['cuda'])
    print('Run on cuda: {}'.format(torch.cuda.current_device()))
    
    
'''Test Models on Datasets'''
if args['mode']=='ARF1': functions.real_data_search3(args, datagen.real_data,'LSTM',models.LSTM)
if args['mode']=='ARF2': functions.real_data_search3(args, datagen.real_data,'nidLSTM 1',models.nidLSTM)
if args['mode']=='ARF3': functions.real_data_search3(args, datagen.real_data,'nidLSTM 24',models.nidLSTM)
if args['mode']=='ARF4': functions.real_data_search3(args, datagen.real_data,'nidLSTM 12',models.nidLSTM)
if args['mode']=='ARF5': functions.real_data_search3(args, datagen.real_data,'nidLSTM 6',models.nidLSTM)
if args['mode']=='ARF6': functions.real_data_search3(args, datagen.real_data,'nidLSTM 16',models.nidLSTM)

if args['mode']=='Shock1': functions.real_data_search3(args, datagen.real_data,'LSTM',models.LSTM)
if args['mode']=='Shock2': functions.real_data_search3(args, datagen.real_data,'nidLSTM 1',models.nidLSTM)
if args['mode']=='Shock3': functions.real_data_search3(args, datagen.real_data,'nidLSTM 24',models.nidLSTM)
if args['mode']=='Shock4': functions.real_data_search3(args, datagen.real_data,'nidLSTM 12',models.nidLSTM)
if args['mode']=='Shock5': functions.real_data_search3(args, datagen.real_data,'nidLSTM 6',models.nidLSTM)
if args['mode']=='Shock6': functions.real_data_search3(args, datagen.real_data,'nidLSTM 16',models.nidLSTM)
    
if args['mode']=='Shock_mow': functions.real_data_search3(args, datagen.real_data,'mow',models.mixLSTM)
if args['mode']=='ARF_mow': functions.real_data_search3(args, datagen.real_data,'mow',models.mixLSTM)