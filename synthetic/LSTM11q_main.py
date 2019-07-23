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
parser.add_argument('--model', type=str,required=True, nargs="+", help='model to train')

## what if we just make it so that it runs separately.? or can we just
## not have choices option?
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
parser.add_argument('--verbose',type=bool,default=False)
parser.add_argument('--synthstart',type=bool,default=False)
parser.add_argument('--te_size',type=int,default=16) #must be even
parser.add_argument('--te_base',type=float,default=10000.0)
parser.add_argument('--savedir',type=str,default='/data1/jeeheh/')



#ADDING
# parser.add_argument('--cuda',type=int,default=0, help='which GPU device to run code on')

# parser.add_argument('--T',type=int,default=48, help='total number of time steps')
# parser.add_argument('--d',type=int,default=76, help='dimension of input')
# parser.add_argument('--num_classes',type=int,default=2, help='number of output classes')
# parser.add_argument('--N',type=int,default=0,help='Size of training set. If not set to zero, indicates that training set should be subsampled.') 


# parser.add_argument('--budget',type=int,default=10,help='number of random initializations during hyperparameter search') 
# parser.add_argument('--epochs',type=int,default=10,help='max number of epochs')
# parser.add_argument('--share_num',type=int,default=2,help="share number for moo and mow")

# parser.add_argument('--batch_size',type=int,default=8)
# parser.add_argument('--shiftLSTMk',type=int,default=0,help='number of LSTM cells learned')
# parser.add_argument('--realstart',type=bool,default=False,help='if set to True, redraws the subsample instead of loading existing subsample from run genrunname. set to True on fist run regardless of if you will subsample or not.')
# parser.add_argument('--te_size',type=int,default=24,help='hyperparameter for temporal embedding for LSTM+TE. must be even number.') 
# parser.add_argument('--te_base',type=float,default=10000.0,help='hyperparameter for temporal embedding for LSTM+TE')

# parser.add_argument('--savedir',type=str,default='save/',help='directory where results will be saved')
# parser.add_argument('--datadir',type=str,default='/data1/jeeheh/mimic/mimic3models/in_hospital_mortality/joh data extraction/',help='location of data (the h5 files generated in savedata.py)')
### for reference only




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



# functions.synth_data_search(args, datagen.synth_data2,\
#                 [('LSTM',models.LSTM),('nidLSTM 1',models.nidLSTM)]) 
    
#     functions.synth_data_search(args, datagen.synth_data2,[('LSTM',models.LSTM)])

    
#     functions.synth_data_search(args, datagen.synth_data2,\
#                                 [('nidLSTMin 1',models.nidLSTMin),('LSTM',models.LSTM),('nidLSTM 1',models.nidLSTM), \
#                                  ('nidLSTMcell 1',models.nidLSTMcell),('nidLSTMforget 1',models.nidLSTMforget),('nidLSTMout 1',models.nidLSTMout)])
    
        
#     functions.synth_data_search(args, datagen.synth_data2,\
#                                 [('LSTMTE',models.LSTMTE),
#                                 ('HyperLSTM',models.HyperLSTM),('LSTMT',models.LSTMT),('LSTM',models.LSTM),('nidLSTM 1',models.nidLSTM)])

        
#     functions.synth_data_search(args, datagen.synth_data2,
#                                 [('mowLSTM',models.mixLSTM)])

