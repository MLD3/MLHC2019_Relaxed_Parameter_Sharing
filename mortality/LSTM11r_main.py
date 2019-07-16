import argparse
import numpy as np
import torch
import pickle
import os

import LSTM11r_datagen as datagen ## CHANGE with VERSION ##
import LSTM11r_functions as functions ## CHANGE with VERSION ##
import LSTM11r_models as models ## CHANGE with VERSION ##

'''Settings'''
parser=argparse.ArgumentParser()

parser.add_argument('--modelname',type=str,required=True) ## CHANGE with VERSION ##
parser.add_argument('--genmodelname',type=str,default='DEFAULT') ## CHANGE with VERSION ##
parser.add_argument('--mode',type=str,required=True)
parser.add_argument('--cuda',type=int,default=0)
parser.add_argument('--gatenames',type=str, nargs="+", default=[],
                    help="gates to change", choices=["input", "output", "cell", "forget"])

parser.add_argument('--budget',type=int,default=10) #HP search budget
parser.add_argument('--epochs',type=int,default=10)

parser.add_argument('--T',type=int,default=48)
parser.add_argument('--d',type=int,default=76)
parser.add_argument('--num_classes',type=int,default=2)
parser.add_argument('--N',type=int,default=0)
parser.add_argument('--share_num',type=int,default=2,help="share number for moo and mow")
parser.add_argument('--synth_num',type=int,default=10) ## if change - gotta also change synthnum_batch
parser.add_argument('--synthnum_batch',nargs='+',type=int,default=[0,10]) 

parser.add_argument('--hidden_size',type=int,default=20)
parser.add_argument('--hyp_hidden_size',type=int,default=20)
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--nidLSTM',type=int,default=0)
parser.add_argument('--realstart',type=bool,default=False)
parser.add_argument('--te_size',type=int,default=24) #must be even
parser.add_argument('--te_base',type=float,default=10000.0)

parser.add_argument('--verbose',type=bool,default=False)
parser.add_argument('--savedir',type=str,default='/data1/jeeheh/')
parser.add_argument('--datadir',type=str,default='/data1/jeeheh/mimic/mimic3models/in_hospital_mortality/joh data extraction/')


#args=parser.parse_args(['--modelname','STN13t','--cuda','4','--KLmatchlen','2','3'])
args=parser.parse_args()
args=vars(args)
if args['genmodelname']=='DEFAULT': args['genmodelname']=args['modelname']

'''Cuda'''
args['use_cuda'] = torch.cuda.is_available()
if args['use_cuda']:
    torch.cuda.set_device(args['cuda'])
    print('Run on cuda: {}'.format(torch.cuda.current_device()))

    
'''Test Models on Datasets'''
if args['mode']=='real1': functions.real_data_search3(args, datagen.IHM_data,'LSTM',models.LSTM)
if args['mode']=='real2': functions.real_data_search3(args, datagen.IHM_data,'nidLSTM 24',models.nidLSTM)
if args['mode']=='real3': functions.real_data_search3(args, datagen.IHM_data,'nidLSTM 1',models.nidLSTM)
if args['mode']=='real4': functions.real_data_search3(args, datagen.IHM_data,'nidLSTM 12',models.nidLSTM)
if args['mode']=='real5': functions.real_data_search3(args, datagen.IHM_data,'nidLSTM 6',models.nidLSTM)
if args['mode']=='real6': functions.real_data_search3(args, datagen.IHM_data,'nidLSTM 16',models.nidLSTM)

if args['mode']=='real7': functions.real_data_search3(args, datagen.IHM_data,'LSTMT',models.LSTMT)
if args['mode']=='real8': functions.real_data_search3(args, datagen.IHM_data,'rLSTM_learned2',models.rLSTM_learned2)
if args['mode']=='real9': functions.real_data_search3(args, datagen.IHM_data,'TCN',models.TCN)
if args['mode']=='real10': functions.real_data_search3(args, datagen.IHM_data,'SNAIL',models.SNAIL)
if args['mode']=='real11': functions.real_data_search3(args, datagen.IHM_data,'HyperLSTM',models.HyperLSTM)
if args['mode']=='real12': functions.real_data_search3(args, datagen.IHM_data,'LSTMTE',models.LSTMTE)
if args['mode']=='real_mow': functions.real_data_search3(args, datagen.IHM_data,'mow',models.mixLSTM)

    
