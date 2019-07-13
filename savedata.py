import numpy as np
import argparse
import time
import os
import imp
import re

#run using python2

parser=argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
usrargs=parser.parse_args()
usrargs=vars(usrargs)

os.chdir(os.path.join(usrargs['datadir'],'mimic')

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import common_utils

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
args = parser.parse_args(['--network', 'mimic3models/keras_models/lstm.py', '--dim', '16', 
                          '--timestep', '1.0', '--depth', '2', '--dropout', '0.3', "--mode", 'train', 
                          '--batch_size', '8'])
print('here')
print(args.small_part)
if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')
