import torch
import os
import pickle as pkl
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import MACCSkeys
from moses import metrics
from rdkit import Chem



def load_data(dir_data,load_tokenized=True,load_raw=False):
    vocab,df_train,df_test,df_train_raw,df_test_raw = [None]*5
    vocab = torch.load(os.path.join(dir_data,'vocab.pt'))
    if load_tokenized:
        df_train    = torch.load(os.path.join(dir_data,'train.pt'))
        df_test     = torch.load(os.path.join(dir_data,'test.pt'))
    if load_raw:
        df_train_raw= torch.load(os.path.join(dir_data,'train_raw.pt'))
        df_test_raw = torch.load(os.path.join(dir_data,'test_raw.pt'))
    return vocab,df_train,df_test,df_train_raw,df_test_raw


def get_model_size(model):
    return sum(p.numel() for p in model.parameters())


def add_before_extension(string_origin,string_add):
    list_origin = string_origin.split('.')
    return '.'.join(list_origin[:-1]) \
           + '_' \
           + string_add \
           + '.' \
           + list_origin[-1]
def get_inverted_vocab(vocab):
    return dict([[v,k] for k,v in vocab.items()])
# def load_model(dir_model,which='latest'):
#     if which == 'latest':
#         idx = len([a for a in os.listdir(dir_model) if 'model' in a])-1
#     else:
#         idx = int(which)
#     path_model = os.path.join(dir_model,'model_'+str(idx)+'.pt')
#     try:
#         model = torch.load(path_model,map_location=torch.device(device))
#     except:
#         model = None
#     return model,idx
def load_model(dir_model,idx):
    path_model = os.path.join(dir_model,'model_'+str(idx)+'.pt')
    model = torch.load(path_model,map_location=torch.device('cpu'))
    return model
def load_optimizer(dir_optimizer,idx):
    path_optimizer = os.path.join(dir_optimizer,'optimizer_'+str(idx)+'.pt')
    optimizer = torch.load(path_optimizer)
    return optimizer


def load_scheduler(dir_scheduler,idx):
    path_scheduler = os.path.join(dir_scheduler,'scheduler_'+str(idx)+'.pt')
    scheduler = torch.load(path_scheduler)
    return scheduler


def status_log_init(epoch):
    current_status = dict()
    current_status['epoch'] = epoch
    current_status['current_loss'] = 0.
    current_status['train_loss'] = 0.
    current_status['valid_loss'] = 0.
    current_status['valid_gen'] = 0.
    current_status['elapsed'] = 0.
    current_status['lr'] = 0.
    return current_status


def gen_checkpoint_name(config):
    dir_checkpoint = [config.model_type]
    if config.model_type == 'LSTM':
        dir_checkpoint += [config.hidden,
                           config.num_layers,
                           ]
    elif config.model_type =='AMG' or model_type=='myGPT':
        dir_checkpoint += [config.dim_attention,
                           config.n_heads,
                           config.dim_feedforward,
                           config.n_layers,
                           ]
    dir_checkpoint = [str(a) for a in dir_checkpoint]
    return '_'.join(dir_checkpoint)

def write_string_to_file(path,command):
    pd.DataFrame([command]).to_csv(path,index=False,)
    return

def save_experiment_command(config,sys_argv):
    path_experiment_command = os.path.join(config.path_checkpoint,
                                           'experiment_command.txt')
    string_experiment_command = ' '.join(sys_argv)
    write_string_to_file(path   = path_experiment_command,
                         command= string_experiment_command
                         )
    return

UTIL_IO_PT  = dict(
                   extension        = '.pt',
                   save_function    = torch.save,
                   load_function    = torch.load,
                   io_util          = lambda x:x,
                   )
UTIL_IO_PKL = dict(
                   extension        = '.pkl',
                   save_function    = pkl.dump,
                   load_function    = pkl.load,
                   io_util          = lambda x: open(x,'wb'),
                   )
UTIL_IO_TSV = dict(
                   extension        = '.tsv',
                   save_function    = pd.read_csv,
                   load_function    = lambda x: x.to_csv,
                   io_util          = lambda x: open(x,'wb'),
                   )

UTILS_IO    = dict(
                   pt   = UTIL_IO_PT,
                   pkl  = UTIL_IO_PKL,
                   tsv  = UTIL_IO_TSV,
                   )


class CharDataset(Dataset):
    def __init__(self, data, vocab,block_size):
        self.vocab = vocab
        self.data = data
        data_size, vocab_size = data.shape, len(vocab)
        self.stoi = vocab
        self.itos = dict([[v,k] for k,v in vocab.items()])
        self.block_size = block_size
        self.vocab_size = vocab_size
    def shuffle(self,random_state):
        self.data = self.data.sample(frac=1,random_state=random_state).reset_index(drop=True)
        return
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        diex = self.data.iloc[idx,0]
        diex += [self.vocab['<pad>']]*(self.block_size-len(diex))
        diex = diex[:self.block_size]
        x = torch.tensor(diex[:-1], dtype=torch.long)
        y = torch.tensor(diex[1:], dtype=torch.long)
        return x, y

def load_model_by_index(dir_model,idx):
    path_model = os.path.join(dir_model,'model_'+str(idx)+'.pt')
    model = torch.load(path_model,map_location=torch.device('cpu'))
    return model

def get_device():
    device='cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    return device

def load_to_device(model):
    device = get_device()
    model = model.to(device)
    return model,device

def get_idx_if_not_reached(dir_model,idx=float('inf')):
    max_idx = -1
    current_idx = -1
    files = os.listdir(dir_model)
    for file in files:
        if 'model' in file:
            current_idx = file.split('.')[0].split('_')[-1]
            if current_idx.isdigit():
                current_idx = int(current_idx)
                if current_idx > max_idx:
                    max_idx = current_idx
    return min(max_idx,idx)

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
