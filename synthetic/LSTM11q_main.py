import argparse
import numpy as np
import torch
import pickle
import os

import LSTM11q_datagen as datagen ## CHANGE with VERSION ##
import LSTM11q_functions as functions ## CHANGE with VERSION ##
import LSTM11q_models as models ## CHANGE with VERSION ##

'''Settings'''
parser=argparse.ArgumentParser()

parser.add_argument('--modelname',type=str,required=True) 
parser.add_argument('--genmodelname',type=str,default='DEFAULT') 
parser.add_argument('--mode',type=str,required=True)
parser.add_argument('--cuda',type=int,default=0)
parser.add_argument('--gatenames',type=str, nargs="+", default=[],
                    help="gates to change", choices=["input", "output", "cell", "forget"])

parser.add_argument('--budget',type=int,default=40) #HP search budget
parser.add_argument('--epochs',type=int,default=30)

parser.add_argument('--T',type=int,default=30)
parser.add_argument('--d',type=int,default=3)
parser.add_argument('--N',type=int,default=1000)
parser.add_argument('--k',type=int,default=10)
parser.add_argument('--delta',type=float,default=.05)
parser.add_argument('--output_size',type=int,default=1)
parser.add_argument('--synth3_duration',type=int,default=3)

parser.add_argument('--synth_num',type=int,default=5) ## if change - gotta also change synthnum_batch
parser.add_argument('--synthnum_batch',nargs='+',type=int,default=[0,5]) 

parser.add_argument('--batch_size',type=int,default=100)
parser.add_argument('--hidden_size',type=int,default=20)
parser.add_argument('--hyp_hidden_size',type=int,default=20)
parser.add_argument('--ratio',type=float,default=.5)
parser.add_argument('--sigma',type=float,default=1)
parser.add_argument('--verbose',type=bool,default=False)
parser.add_argument('--synthstart',type=bool,default=False)
parser.add_argument('--te_size',type=int,default=16) #must be even
parser.add_argument('--te_base',type=float,default=10000.0)
parser.add_argument('--kvdims',type=int,default=10)
parser.add_argument('--num_filters',type=int,default=10)

#args=parser.parse_args(['--modelname','STN13t','--cuda','4','--KLmatchlen','2','3'])
args=parser.parse_args()
args=vars(args)
if args['genmodelname']=='DEFAULT': args['genmodelname']=args['modelname']


'''Cuda'''
args['use_cuda'] = torch.cuda.is_available()
if args['use_cuda']:
    torch.cuda.set_device(args['cuda'])
    print('Run on cuda: {}'.format(torch.cuda.current_device()))

'''Create Synth Dataset'''
if args['synthstart']:
    datagen.synth_evaluate(args)

'''Test Models on the Synthetic Datasets'''

if args['mode']=='case1':
    if args['synthstart']:
        datagen.synth_evaluate2(args)
        
    functions.synth_data_search(args, datagen.synth_data2,\
                                [('LSTM',models.LSTM),('nidLSTM 1',models.nidLSTM)]) 
    
if args['mode']=='case2':
    if args['synthstart']:
        datagen.synth_evaluate2(args)
        
    functions.synth_data_search(args, datagen.synth_data2,[('LSTM',models.LSTM)])
    
if args['mode']=='case3':
    if args['synthstart']:
        datagen.synth_evaluate2(args)
        
    functions.synth_data_search(args, datagen.synth_data2,\
                                [('nidLSTMin 1',models.nidLSTMin),\
                                 ('nidLSTMcell 1',models.nidLSTMcell),('nidLSTMforget 1',models.nidLSTMforget),('nidLSTMout 1',models.nidLSTMout)])
    
if args['mode']=='case4':
    if args['synthstart']:
        datagen.synth_evaluate2(args)
        
    functions.synth_data_search(args, datagen.synth_data2,\
                                [('nidLSTMin 1',models.nidLSTMin),('LSTM',models.LSTM),('nidLSTM 1',models.nidLSTM), \
                                 ('nidLSTMcell 1',models.nidLSTMcell),('nidLSTMforget 1',models.nidLSTMforget),('nidLSTMout 1',models.nidLSTMout)])
    
if args['mode']=='case5':
    if args['synthstart']:
        datagen.synth_evaluate2(args)
        
    functions.synth_data_search(args, datagen.synth_data2,\
                                [('LSTMTE',models.LSTMTE),('rLSTM_learned2',models.rLSTM_learned2),('TCN',models.TCN),('SNAIL',models.SNAIL),\
                                ('HyperLSTM',models.HyperLSTM),('LSTMT',models.LSTMT),('LSTM',models.LSTM),('nidLSTM 1',models.nidLSTM)])

if args['mode']=='mow':
    if args['synthstart']:
        datagen.synth_evaluate2(args)
        
    functions.synth_data_search(args, datagen.synth_data2,
                                [('mowLSTM',models.mowLSTM)])

if args['mode']=='moo':
    if args['synthstart']:
        datagen.synth_evaluate2(args)
        
    functions.synth_data_search(args, datagen.synth_data2,
                                [('mooLSTM',models.mooLSTM)])

if args['mode']=='moe':
    if args['synthstart']:
        datagen.synth_evaluate2(args)
        
    functions.synth_data_search(args, datagen.synth_data2,
                                [('moe',models.moeLSTM)])
    
    
if args['mode']=='changegate_mow':
    if args['synthstart']:
        datagen.synth_evaluate2(args)
        
    functions.synth_data_search(args, datagen.synth_data2,
                                [('changegate_mow',models.ChangeGate_mow)])

if args['mode']=='changegate_moo':
    if args['synthstart']:
        datagen.synth_evaluate2(args)
        
    functions.synth_data_search(args, datagen.synth_data2,
                                [('changegate_moo',models.ChangeGate_moo)])
if args['mode']=='lstm':
    if args['synthstart']:
        datagen.synth_evaluate2(args)
        
    functions.synth_data_search(args, datagen.synth_data2,
                                [('LSTM',models.pytorchLSTM_jiaxuan)])

