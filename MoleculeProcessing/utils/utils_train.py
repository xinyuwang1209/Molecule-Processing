import os
# def get_optimizer
# # Init criterion, optimizer, scheduler
# if config.loss_weight_mode == 'normal':
#     criterion = nn.CrossEntropyLoss()
# elif config.loss_weight_mode == 'reduce_CO':
#     weights = [0.8]*len(vocab.keys())
#     weights[vocab['C']] = 0.5
#     weights[vocab['O']] = 0.8
#     weights[vocab['<bos>']] = 1.
#     weights[vocab['<eos>']] = 1.
#     class_weights = torch.FloatTensor(weights).to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
# elif config.loss_weight_mode == 'increase_parentheses':
#     weights = [1.]*len(vocab.keys())
#     weights[vocab['(']] = 10.
#     weights[vocab[')']] = 10.
#     class_weights = torch.FloatTensor(weights).to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
# elif config.lospd.Datas_weight_mode == 'dataset_distribution':
#     weights = pd.read_csv('vocab_distribution').values
#     weights = 1/(weights/weights.sum())
#     class_weights = torch.FloatTensor(weights).to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10.0,gamma=0.5)
#
# from ..config.config import *
# path_train_log  = os.path.join(config.path_checkpoint,
#                                MODEL_LATEST)
# path_model_save = os.path.join(config.path_checkpoint,
#                                LOG_TRAIN_LATEST)

from SMILE_Generation.utils.utils        import *
from SMILE_Generation.utils.utils_train  import *
from SMILE_Generation.config.config      import *

import numpy as np
import pandas as pd
def init_train_log():
    return pd.DataFrame([],columns=[EPOCH,
                                    TRAIN_LOSS,
                                    TEST_LOSS,
                                    TIME_ELAPSED,
                                    LR,
                                    TOKENS])
