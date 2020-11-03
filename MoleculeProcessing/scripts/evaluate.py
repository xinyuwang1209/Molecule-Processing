import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import os
import sys
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import MACCSkeys
from moses import metrics
from rdkit import Chem
def test_valid(mol):
    try:
        mol = Chem.MolFromSmiles(mol)
    except TypeError:
        return None
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return mol




from SMILE_Generation.utils.utils        import *
from SMILE_Generation.utils.utils_sample  import *
from SMILE_Generation.config.config      import *





import configparser
config_data = configparser.ConfigParser()
config_data.read('config_data.txt')
dir_data = os.path.expandvars(config_data['DATA']['dir_data'])
size_block = int(config_data['DATA']['block_size'])

# Load config
import argparse
parser = argparse.ArgumentParser(description='Run experiment')
parser.add_argument('--path_sample', type=str, default='dir_save/bla',
                        help='dir_data')
parser.add_argument('--size_batch', type=int, default=2048, metavar='N',
                    help='size_batch')
parser.add_argument('--n_sample', type=int, default=30000,
                    help='device')
parser.add_argument('--temperature', type=float, default=1.,
                    help='device')
parser.add_argument('--overwrite', type=int, default=0,
                    help='device')
parser.add_argument('--idx', type=int, default=-1,
                    help='unique_k')
parser.add_argument('--old_vocab', type=int, default=0,
                    help='unique_k')




config = parser.parse_known_args()[0]


if config.path_sample.split(os.sep)[0] != DIR_SAVE:
    config.path_sample = os.path.join(DIR_SAVE,config.path_sample)

print('Evaluating:',config.path_sample)


dir_sample = os.sep.join(config.path_sample.split(os.sep)[:-1])
filename_sample = config.path_sample.split(os.sep)[-1]
filename_sample = 'metrics_' + '.'.join('_'.join(filename_sample.split('_')[1:]).split('.')[:-1]) + '.pt'


path_metrics = os.path.join(dir_sample,filename_sample)
if os.path.exists(path_metrics) and not config.overwrite:
    print('exists, quit')
    quit()


df_sample = pd.read_csv(config.path_sample)
df_sample = df_sample.loc[~df_sample[SMILES].isnull()]
print('Samples Loaded')
n_sample = df_sample.shape[0]


stats_sample = dict()
vocab,_,_,df_train_raw,df_test_raw = load_data(dir_data,load_tokenized=False,load_raw=True)
df_sample[NOVALTY] = ~df_sample[SMILES].isin(df_train_raw[SMILES])
rate_novalty = df_sample[NOVALTY].sum()/df_sample.shape[0]
df_sample[VALIDITY] = df_sample[SMILES].apply(test_valid)
df_sample[LOGP] = df_sample[VALIDITY].apply(lambda x: Chem.Crippen.MolLogP(x) if x is not None else None)
df_sample[WEIGHT] = df_sample[VALIDITY].apply(lambda x: Chem.Descriptors.MolWt(x) if x is not None else None)
df_sample[QED] = df_sample[VALIDITY].apply(lambda x: Chem.QED.qed(x) if x is not None else None)

df_test2            = df_test_raw.sample(n=n_sample,random_state=42)
df_test2[VALIDITY]  = df_test2[SMILES].apply(test_valid)
df_test2[LOGP]      = df_test2[VALIDITY].apply(lambda x: Chem.Crippen.MolLogP(x) if x is not None else None)
df_test2[WEIGHT]    = df_test2[VALIDITY].apply(lambda x: Chem.Descriptors.MolWt(x) if x is not None else None)
df_test2[QED]       = df_test2[VALIDITY].apply(lambda x: Chem.QED.qed(x) if x is not None else None)
path_test_dist = os.path.join(dir_sample,FILENAME_TEST_DIST)
if not os.path.exists(path_test_dist):
    torch.save(df_test2[[LOGP,WEIGHT,QED]],path_test_dist)



df_train2           = df_train_raw.sample(n=n_sample,random_state=42)
df_train2[VALIDITY] = df_train2[SMILES].apply(test_valid)
df_train2[LOGP]     = df_train2[VALIDITY].apply(lambda x: Chem.Crippen.MolLogP(x) if x is not None else None)
df_train2[WEIGHT]   = df_train2[VALIDITY].apply(lambda x: Chem.Descriptors.MolWt(x) if x is not None else None)
df_train2[QED]      = df_train2[VALIDITY].apply(lambda x: Chem.QED.qed(x) if x is not None else None)
path_train_dist = os.path.join(dir_sample,FILENAME_TRAIN_DIST)
if not os.path.exists(path_train_dist):
    torch.save(df_train2[[LOGP,WEIGHT,QED]],path_train_dist)


list_test = df_test2[SMILES].tolist()
list_sample = df_sample[SMILES].tolist()
device=get_device()
metrics_sample = metrics.get_all_metrics(test=list_test,gen=list_sample,device=device)
# metrics_sample[NOVALTY]=n_novalty/n_sample
torch.save(metrics_sample,path_metrics)
# [print(a,b) for a,b in metrics_sample.items()]
print('Computing metrics')
metrics_sample['Model'] = config.path_sample.split(os.sep)[-2]
idx = config.path_sample.split(os.sep)[-1].split('_')[1]
model = torch.load(os.path.join(dir_sample,'model_'+idx+'.pt'))
metrics_sample['Model_Size'] = sum(p.numel() for p in model.parameters())
metrics_sample['Epoches'] = idx
metrics_sample[NOVALTY] = rate_novalty
print(metrics_sample['Model'])
metrics_sample['Sheet'] = ','.join([str(metrics_sample['Model_Size']),
                str(metrics_sample['Epoches']),
                str(metrics_sample['valid']),
                str(metrics_sample[NOVALTY]),
                str(metrics_sample['unique@1000']),
                str(metrics_sample['unique@10000']),
                str(metrics_sample['FCD/Test']),
                str(metrics_sample['SNN/Test']),
                str(metrics_sample['Frag/Test']),
                str(metrics_sample['Scaf/Test']),
                str(metrics_sample['IntDiv']),
                str(metrics_sample['IntDiv2']),
                str(metrics_sample['Filters']),
                str(metrics_sample['logP']),
                str(metrics_sample['SA']),
                str(metrics_sample['QED']),
                ])
print(metrics_sample['Sheet'])
torch.save(metrics_sample,path_metrics)
