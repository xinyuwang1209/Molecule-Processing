import math
import logging
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from SMILE_Generation.utils.utils_train import *
logger = logging.getLogger(__name__)
from SMILE_Generation.utils.utils        import *
from SMILE_Generation.utils.utils_train  import *
from SMILE_Generation.config.config      import *

class TrainerConfig:
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 0
    config = None
    epoch = 0
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.train_log = init_train_log()
        if os.path.exists(os.path.join(self.config.config.path_checkpoint,LOG_TRAIN_LATEST)):
            self.train_log = pd.read_csv(os.path.join(self.config.config.path_checkpoint,LOG_TRAIN_LATEST))
        self.config.epoch = get_idx_if_not_reached(self.config.config.path_checkpoint)+1
        if self.config.epoch>0:
            self.model      = load_model(    self.config.config.path_checkpoint,self.config.epoch-1)
            self.optimizer  = load_optimizer(self.config.config.path_checkpoint,self.config.epoch-1)
            self.tokens = self.train_log.loc[self.config.epoch-1,TOKENS]
        else:
            self.tokens = 0 # counter used for learning rate decay
            self.optimizer = model.configure_optimizers(config)
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        path_checkpoint = self.config.config.path_checkpoint
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", path_checkpoint)
        path_model_epoch = add_before_extension(os.path.join(path_checkpoint,
                MODEL_LATEST),
            str(self.config.epoch))
        torch.save(raw_model, path_model_epoch)
        path_optimizer_epoch = \
            add_before_extension(
                os.path.join(
                    path_checkpoint,
                    OPTIMIZER_LATEST
                    ),
                    str(self.config.epoch)
                    )
        torch.save(
            self.optimizer,
            path_optimizer_epoch
            )
        self.train_log.to_csv(
            os.path.join(
                path_checkpoint,
                LOG_TRAIN_LATEST
                )
            ,index=False
            )
        path_train_log_epoch = \
            add_before_extension(
                os.path.join(
                    path_checkpoint,
                    LOG_TRAIN_LATEST
                    ),
                str(self.config.epoch)
                )
        self.train_log.to_csv(
            path_train_log_epoch,
            index=False)
    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = self.optimizer
        while self.config.epoch < config.config.epochs and self.config.epoch != config.config.epochs:
            current_status = dict([[a,None] for a in self.train_log.columns])
            current_status[EPOCH] = self.config.epoch
            time_start = time.time()
            current_status = self.run_epoch('train',current_status)
            current_status[TIME_ELAPSED] = int(time.time()-time_start)
            current_status[TOKENS] = self.tokens
            if self.test_dataset is not None:
                current_status = self.run_epoch('test',current_status)
            self.train_log.loc[self.config.epoch] = current_status
            self.save_checkpoint()
            self.config.epoch += 1

    def run_epoch(self,split,current_status):
        model = self.model
        is_train = split == 'train'
        model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset
        data.shuffle(random_state=self.config.epoch)
        loader = DataLoader(data, shuffle=False, pin_memory=True,
                            batch_size=self.config.config.size_batch,
                            num_workers=self.config.num_workers)
        del data
        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        for it, (x, y,boundary) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.set_grad_enabled(is_train):
                logits, loss = model.forward(x, y,boundary=boundary)
                loss = loss.mean()
                losses.append(loss.item())

            if is_train:
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()
                if self.config.lr_decay:
                    self.tokens += (y >= 0).sum()
                    if self.tokens < self.config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, self.config.warmup_tokens))
                    else:
                        progress = float(self.tokens - self.config.warmup_tokens) / float(max(1, self.config.final_tokens - self.config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = self.config.learning_rate * lr_mult
                    current_status[LR] = lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.config.learning_rate
                    current_status[LR] = lr
                pbar.set_description(f"epoch {self.config.epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
        current_status[split+'_loss'] = float(np.mean(losses))
        if not is_train:
            test_loss = float(np.mean(losses))
            logger.info("test loss: %f", test_loss)

        return current_status
