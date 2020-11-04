# os
import os,sys
import time
import argparse,configparser
import pickle as pkl

# data process
import numpy as np
import pandas as pd

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


# log
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

from MoleculeProcessing.utils.utils        import *
from MoleculeProcessing.utils.utils_train  import *
from MoleculeProcessing.config.config      import *
# model

#
config_data = configparser.ConfigParser()
config_data.read('config_data.txt')
dir_data = os.path.expandvars(config_data['DATA']['dir_data'])
size_block = int(config_data['DATA']['block_size'])
# Load config
parser = argparse.ArgumentParser(description='Run experiment')
model_arg = parser.add_argument_group('Model')
# Transformer
model_arg.add_argument('--model_type', type=str, default='AMG',
                    help='print')
model_arg.add_argument('--dim_attention', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 20)')
model_arg.add_argument('--n_heads', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 20)')
model_arg.add_argument('--dim_feedforward', type=int, default=0, metavar='N',
                        help='input batch size for training (default: 20)')
model_arg.add_argument('--n_layers', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 20)')
model_arg.add_argument('--dropout', type=float, default=0.2, metavar='N',
                        help='input batch size for training (default: 20)')
# LSTM
model_arg.add_argument('--hidden', type=int, default=768, metavar='N',
                        help='input batch size for training (default: 20)')
model_arg.add_argument('--num_layers', type=int, default=3, metavar='N',
                        help='input batch size for training (default: 20)')

train_arg = parser.add_argument_group('Train')
train_arg.add_argument('--epochs', type=int, default=10,
                    help='epochs')
train_arg.add_argument('--size_batch', type=int, default=64, metavar='N',
                    help='size_batch')
train_arg.add_argument('--learningrate', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
train_arg.add_argument('--dir_checkpoint', type=str, default='default',
                    help='For loading the model')
train_arg.add_argument('--lock_lr', type=int, default=0,
                    help='lock_lr')
train_arg.add_argument('--print_log', type=int, default=0,
                    help='print')
train_arg.add_argument('--transposed_embedding', type=int, default=0,
                    help='print')
train_arg.add_argument('--mask_pading', type=int, default=1,
                    help='print')
train_arg.add_argument('--test', type=int, default=0,
                    help='print')
train_arg.add_argument('--print_train_input', type=int, default=0,
                    help='print')
train_arg.add_argument('--loss_weight_mode', type=str, default='normal',
                    help='print')
train_arg.add_argument('--type_encoding', type=int, default=0,
                    help='print')
train_arg.add_argument('--screen_width', type=int, default=141,
                    help='print')
train_arg.add_argument('--load_pretrain', type=int, default=0,
                    help='print')



config = parser.parse_args()
if config.dim_feedforward == 0:
    config.dim_feedforward = config.dim_attention*4
# Config device
try:
    config.screen_width = os.get_terminal_size()[0]
except:
    pass
if config.dir_checkpoint == 'default':
    config.dir_checkpoint = gen_checkpoint_name(config)
config.path_checkpoint = os.path.join(os.path.expandvars(DIR_SAVE),
                                      config.dir_checkpoint)
if not os.path.exists(config.path_checkpoint):
    os.makedirs(config.path_checkpoint)
# Save Experiment Command
save_experiment_command(config,sys.argv)
# fh = logging.FileHandler(os.path.join(config.path_checkpoint,'train.log'))
# logger.addHandler(fh)
# Load data
# dir_data = config.data_dir
vocab,df_train,df_test,_,_ = load_data(dir_data)
# if os.path.exists(os.path.join(config.path_checkpoint,'vocab.pt')):
torch.save(vocab,os.path.join(config.path_checkpoint,'vocab.pt'))
vocab_size = len(vocab)
# block_size = df_train[0].apply(len).max()
train_dataset = CharDataset(df_train,vocab,size_block)
test_dataset  = CharDataset(df_test,vocab,size_block)
del df_train
del df_test


# Load Model
if config.model_type == 'LSTM':
    from MoleculeProcessing.models.lstm.char_rnn import CharRNN
    from MoleculeProcessing.models.lstm.trainer import Trainer, TrainerConfig
    # Load Model
    model = CharRNN(vocab,config)
elif config.model_type == 'AMG':
    from MoleculeProcessing.models.amg.model import AMG, AMGConfig
    from MoleculeProcessing.models.amg.trainer import Trainer, TrainerConfig
    # Load Model
    mconf = AMGConfig(vocab_size,
                      size_block,
                      n_layer=config.n_layers,
                      n_head=config.n_heads,
                      n_embd=config.dim_attention)
    model = AMG(mconf)
    # Load Trainer
#
if config.load_pretrain:
    model = torch.load(os.path.join(config.path_checkpoint,MODEL_PRETRAIN))
    print('pretrain loaded')
print(config)
print(get_model_size(model))
tconf = TrainerConfig(max_epochs=config.epochs,
                      ckpt_path='checkpoints',
                      config=config)
trainer = Trainer(model, train_dataset, test_dataset, tconf)
del train_dataset
del test_dataset
trainer.train()

#
# for epoch in range(epochs):
#     if config.randomize_train:
#         df_train_epoch = df_train.sample(frac=1,random_state=epoch).reset_index(drop=True)
#     else:
#         df_train_epoch = df_train
#     path_model_epoch = add_before_extension(path_model_save,str(epoch))
#     # path_model_state_save_epoch = add_before_extension(path_model_state_save,str(epoch))
#     path_log_epoch = add_before_extension(path_train_log,str(epoch))
#     if os.path.exists(path_model_epoch):
#         model = torch.load(path_model_epoch)
#         log_save = pd.read_csv(path_train_log)
#         if not config.lock_lr:
#             scheduler.step()
#         continue
#     current_status = status_log_init(epoch)
#     # Train
#     if config.reload_data:
#         df_train = pkl.load(open(path_df_train,'rb'))
#         df_test  = pkl.load(open(path_df_test ,'rb'))
#     model.train()
#     current_status = run_epoch(data_iter=df_train_epoch,
#                                model=model,
#                                criterion=criterion,
#                                optimizer=optimizer,
#                                scheduler=scheduler,
#                                epoch=epoch,
#                                config=config,
#                                vocab=vocab,
#                                current_status=current_status,
#                                batch_first=config.batch_first,)
#     current_status['lr'] = scheduler.get_lr()[0]
#     # Valid
#     model.eval()
#     current_status = run_epoch(data_iter=df_test,
#                                model=model,
#                                criterion=criterion,
#                                optimizer=None,
#                                scheduler=None,
#                                epoch=epoch,
#                                config=config,
#                                vocab=vocab,
#                                current_status=current_status,
#                                batch_first=config.batch_first,)
#     # Valid from sampling
#     # _, rate = sample(model,vocab)
#     current_status['valid_gen'] = 0
#     if not config.lock_lr:
#         scheduler.step()
#     if config.print_log:
#         print_string = 'Epoch: ' + str(epoch+1).zfill(len(str(epoch))) + '/' + str(epochs) \
#                      + ', train_loss: ' + '{:1.5f}'.format(current_status['train_loss']) \
#                      + ', valid_loss: ' + '{:1.5f}'.format(current_status['valid_loss']) \
#                      + ', valid_gen: '  + '{:1.3f}'.format(current_status['valid_gen']) \
#                      + ', LR: ' + '{:.3e}'.format(scheduler.get_lr()[0]) \
#                      + ', elapsed: ' + str(int(current_status['elapsed'])//60) + ':' + str(int(current_status['elapsed'])%60).zfill(2)
#         print_string = print_string.ljust(config.screen_width)
#         print(print_string,sep='')
#     torch.save(model,path_model_epoch)
#     # torch.save(model.state_dict(),path_model_state_save_epoch)
#     pkl.dump(current_status,open(path_log_epoch,'wb'))
#     log_save.loc[epoch] = current_status
#     log_save.to_csv(path_train_log,index=False)
