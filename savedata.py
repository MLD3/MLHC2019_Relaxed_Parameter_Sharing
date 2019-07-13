#run using python 2

import numpy as np
import argparse
import time
import os
import imp
import re
import sys


# Bring in location of MIMIC data
parser=argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--savedir',type=str, required=True)
usrargs=parser.parse_args()
usrargs=vars(usrargs)
#eg: python savedata.py --datadir /data1/jeeheh/mimic/
#datadir = location of mimic data


# Change directory and add to python path
sys.path.insert(0,usrargs['datadir'])
os.chdir(usrargs['datadir'])


from mimic3models import metrics
from mimic3models import common_utils
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer


# Set settings
parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
args = parser.parse_args(['--network', 'mimic3models/keras_models/lstm.py'
                          , '--dim', '16'
                          ,'--timestep', '1.0'
                          , '--depth', '2'
                          , '--dropout', '0.3'
                          , "--mode", 'train'
                          ,'--batch_size', '8'])
target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')


# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir='data/in-hospital-mortality/train/',
                                         listfile='data/in-hospital-mortality/train_listfile.csv',
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir='data/in-hospital-mortality/train/',
                                       listfile='data/in-hospital-mortality/val_listfile.csv',
                                       period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here onlycont vs all
normalizer.load_params('mimic3models/in_hospital_mortality/ihm_ts%s.input_str:%s.start_time:zero.normalizer' % (args.timestep, args.imputation))


args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl


# Read data
train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part, return_names=True)
val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part, return_names=True)


test_reader = InHospitalMortalityReader(dataset_dir='data/in-hospital-mortality/test/',
                                        listfile='data/in-hospital-mortality/test_listfile.csv',
                                        period_length=48.0)

ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part, return_names=True)


# Save data
np.savez(os.path.join(usrargs['savedir'],"IHMtrain"), data=train_raw['data'][0], labels=train_raw['data'][1], ids=train_raw['names'])
np.savez(os.path.join(usrargs['savedir'],"IHMval"), data=val_raw['data'][0], labels=val_raw['data'][1], ids=val_raw['names'])
np.savez(os.path.join(usrargs['savedir'],"IHMtest"), data=ret['data'][0], labels=ret['data'][1], ids=ret['names'])