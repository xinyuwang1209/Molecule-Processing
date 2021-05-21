# Required path

import torch
import pickle as pkl

import numpy as np
import pandas as pd

SMILES = 'SMILES'
IS_NOVEL = 'IS_NOVAL'
NOVALTY = 'Novalty'
# VALIDITY = 'Validity'
IS_VALID = 'IS_VALID'
IS_NOVAL = 'IS_NOVAL'
DIR_SAVE = 'dir_save'
MODEL_LATEST = 'model.pt'
LOG_TRAIN_LATEST = 'train_log.csv'
OPTIMIZER_LATEST = 'optimizer.pt'
SCHEDULER_LATEST = 'scheduler.pt'
EPOCH           = 'epoch'
TRAIN_LOSS      = 'train_loss'
TEST_LOSS       = 'test_loss'
TIME_ELAPSED    = 'time_elapsed'
LR              = 'lr'
TOKENS          = 'tokens'

LOGP = 'logP'
WEIGHT = 'weight'
QED = 'QED'
VALIDITY = 'SMILES_VALID'
FILENAME_TRAIN_DIST = 'train_dist.pt'
FILENAME_TEST_DIST = 'test_dist.pt'
MODEL_PRETRAIN = 'model_pretrained.pt'

PYFILE_SAMPLER = "sampler.py"
PYFILE_TRAINER = "trainer.py"
PYFILE_DATALOADER = "dataloader.py"
# PYFILE_SAMPLER = "sampler.py"
