# os
import os,sys
import time
import pickle as pkl

# data process
import numpy as np
import pandas as pd

# torch
import torch

# log
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

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
parser.add_argument('--dir_data', type=str, default='/scratch/xiw14035/data/Data_Real_76M/',
                        help='dir_data')
parser.add_argument('--dir_model', type=str, default='dir_save/bla',
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
# Assign config parameters
print('Sampling:',config.dir_model)
if config.dir_model.split(os.sep)[0] != DIR_SAVE:
    config.dir_model = os.path.join(DIR_SAVE,config.dir_model)
print('Path:',config.dir_model)
device = get_device()

# idx = len([a for a in os.listdir(dir_model) if 'model' in a])-1
if config.idx == -1:
    config.idx=float('inf')
idx = get_idx_if_not_reached(config.dir_model,config.idx)
print('idx:',idx)
if idx==-1:
    print('No model found, quit')
    quit()

path_sample = os.path.join(config.dir_model,'sample_'+str(idx)+'_'+str(config.temperature)+'.csv')
if os.path.exists(path_sample) and not config.overwrite:
    print('exists, quit')
    quit()



vocab,_,_,_,_ = load_data(dir_data,load_tokenized=False)
if config.old_vocab:
    vocab = torch.load(os.path.join(dir_data,'vocab_save.pt'))
inverted_vocab = get_inverted_vocab(vocab)
print('Vocab Loaded')
model = torch.load(os.path.join(config.dir_model,'model_'+str(idx)+'.pt'))
print('Model Loaded')
# Load Model
if config.dir_model[-1] == '/':
    config.dir_model = config.dir_model[:-1]
config.model_type = config.dir_model.split(os.sep)[-1].split('_')[0]
if config.model_type == 'LSTM':
    from SMILE_Generation.models.lstm.sampler import sample
    # Load Model
    parameters_sample = dict(model              = model,
                             size_block         = size_block,
                             size_batch         = config.size_batch,
                             temperature        = config.temperature,
                             vocab_bos          = vocab['<bos>'],
                             )
elif config.model_type == 'AMG':
    from SMILE_Generation.models.amg.sampler import sample
    parameters_sample = dict(model              = model,
                             x                  = torch.tensor([[vocab['<bos>']]*config.size_batch], dtype=torch.long,device=device).transpose(0,1).contiguous(),
                             steps              = size_block,
                             temperature        = config.temperature,
                             sample             = True,
                             top_k              = None,
                             )


parameters_sample_n = dict(n_to_sample        = config.n_sample,
                           inverted_vocab       = inverted_vocab,
                           size_batch           = config.size_batch,
                           parameters_sample    = parameters_sample,
                           sample               = sample,
                           )
array_sampled = sample_n(**parameters_sample_n)
array_sampled = [[b for b in a if (b !='<unk>') and (b!='<pad>')] for a in array_sampled]

print('Sample Completed')

list_sample = [''.join(torch_array).split('<eos>')[0] for torch_array in array_sampled]
df_sample = pd.DataFrame(list_sample)
df_sample.columns=[SMILE]
df_sample.to_csv(path_sample,index=False)


















def get_fingerprints(df,morgan__r=2,morgan__n=1024):
    df['fingerprint_maccs']=df[VALIDITY].apply(convert2fgs)
    length = 1
    for index_test_valid_unique in range(df.shape[0]):
        fp = df.iloc[index_test_valid_unique]['fingerprint_maccs']
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    df['fingerprint_maccs']=df['fingerprint_maccs'].apply(lambda fp: fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :])
    return np.vstack(df['fingerprint_maccs'])



















def process_df(df):
    df[VALIDITY] = df[SMILES].apply(test_valid)
    df[IS_VALID] = ~df[VALIDITY].isnull()
    df[IS_NOVAL] = ~df[SMILES].isin(df_train_raw[SMILES])
    df_valid = df.loc[df[IS_VALID]][[SMILE,VALIDITY]]
    df_valid['CANONIC'] = df_valid[VALIDITY].apply(lambda x: Chem.MolToSmiles(x))
    df_valid['IS_UNIQUE'] = ~df_valid['CANONIC'].duplicated()
    n_novalty = df[IS_NOVAL].sum()
    n_valid = df[IS_VALID].sum()
    n_unique_k = df_valid['IS_UNIQUE'].iloc[:unique_k].sum()
    return df,df_valid,n_novalty,n_valid,n_unique_k
