#run using python 2

import numpy as np
import pandas as pd
import argparse
import time
import os
import imp
import re
import sys
import yaml


# Functions
def reformat(IDS):
    subject_id = [re.sub("_episode.*_timeseries.csv","",str(x)) for x in IDS]
    subject_id = [re.sub("b","",x) for x in subject_id]
    subject_id = [x.replace('\'','') for x in subject_id]
    subject_id = [int(x)for x in subject_id]
    
#     episode = [re.sub("b\'.*_episode","",str(x)) for x in IDS]
#     episode = [re.sub("_timeseries.csv\'","",str(x)) for x in episode]
    episode = [x.split('episode')[1] for x in IDS]
    episode = [re.sub("_timeseries.csv","",str(x)) for x in episode]
    episode = [int(x) for x in episode]
    
    IDS = pd.DataFrame({'SUBJECT_ID':subject_id,'Order':np.arange(len(IDS)),'Episode':episode,'Original':IDS})
    return IDS

def do_mapping(IDS, MAP, ON, NEWLABELS1, NEWLABELS2, ON2, keep=False):
    IDS = IDS.merge(MAP, how='left', on=ON, indicator=True)
#     print(IDS['_merge'].value_counts())
    IDS.drop('_merge',axis=1, inplace=True)
    
    IDS = IDS.merge(NEWLABELS1, how='left', on=ON2, indicator=True)
#     print(IDS['_merge'].value_counts())
    if keep==False: IDS = IDS.loc[IDS['_merge']=='both',:]
    IDS.drop('_merge',axis=1, inplace=True)

    IDS = IDS.merge(NEWLABELS2, how='left', on=ON2, indicator=True)
#     print(IDS['_merge'].value_counts())
    if keep==False: IDS = IDS.loc[IDS['_merge']=='both',:]
    IDS.drop('_merge',axis=1, inplace=True)

    return IDS

def transform_label(df, keep_feat):
    df_ARF = df.loc[(df['ARF_ONSET_HOUR'] > 48) | (df['ARF_LABEL']==0),:].sort_values('Order')
    df_ARF['LABEL'] = df_ARF['ARF_LABEL']
    df_Shock = df.loc[(df['Shock_ONSET_HOUR'] > 48) | (df['Shock_LABEL']==0),:].sort_values('Order')
    df_Shock['LABEL'] = df_Shock['Shock_LABEL']
    return df_ARF.loc[:,keep_feat+['LABEL']], df_Shock.loc[:,keep_feat+['LABEL']]


# Bring in location of MIMIC data and ARF/shock labels
parser=argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--savedir',type=str, required=True)
usrargs=parser.parse_args()
usrargs=vars(usrargs)
#eg: python savedata.py --datadir /data1/jeeheh/mimic/
#datadir = location of mimic data
data_path = yaml.full_load(open(os.path.join(os.getcwd(),'ARFshock_label','config.yaml')))['data_path']


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


# Bring in ARF and Shock Labels
ARF = pd.read_csv(os.path.join(data_path,'labels','ARF.csv'))
shock = pd.read_csv(os.path.join(data_path,'labels','Shock.csv'))

# Bring in Subject ID & Episode for MIMIC III Benchmark Data
train = train_raw['names']
val = val_raw['names']
test = ret['names']

# Clean Subject ID & Episode. Create Order feature.
train = reformat(train)
val = reformat (val)
test = reformat (test)

# Create Mapping from Subject ID & Episode -> ICUSTAY_ID
mapping = pd.read_csv(os.path.join(usrargs['datadir'],'raw_csvs','ICUSTAYS.csv'))
mapping.INTIME = pd.to_datetime(mapping.INTIME)
mapping.OUTTIME = pd.to_datetime(mapping.OUTTIME)
mapping['ones'] = np.ones(mapping.shape[0])
mapping['Episode'] = mapping.sort_values(by=['SUBJECT_ID','INTIME', 'OUTTIME']).groupby('SUBJECT_ID')['ones'].apply(lambda x: x.cumsum())


# Map (Subject ID & Episode) to ICUSTAY_ID. Merge ICUSTAY_ID to ARF Shock Labels.
# Only keep patients that have labels.
train = do_mapping(train, mapping, ['SUBJECT_ID','Episode'], ARF, shock, 'ICUSTAY_ID')
val = do_mapping(val, mapping, ['SUBJECT_ID','Episode'], ARF, shock, 'ICUSTAY_ID')
test = do_mapping(test, mapping, ['SUBJECT_ID','Episode'], ARF, shock, 'ICUSTAY_ID')


# Transform ARF/Shock labels to reflect benchmark formulation of the task.
# Only consider patients who did get ARF/Shock within the first 48 hours.
# User Order to order df correctly.
keep_feat = ['SUBJECT_ID','Order','Episode','HADM_ID','ICUSTAY_ID']
train_ARF, train_Shock = transform_label(train, keep_feat)
val_ARF, val_Shock = transform_label(val, keep_feat)
test_ARF, test_Shock = transform_label(test, keep_feat)

# Save
for zdf, zname in [(train_ARF,'train_ARF'), (train_Shock,'train_Shock'), (val_ARF,'val_ARF'), 
          (val_Shock,'val_Shock'), (test_ARF,'test_ARF'), (test_Shock,'test_Shock')]:
    zdf.to_hdf(os.path.join(usrargs['savedir'],zname+'.h5'), key='df', mode='w')