import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse
import time

from .dataset import DittoDataset
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from torch.cuda import amp

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)


    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc) # .squeeze() # .sigmoid()


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()
    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        p = metrics.precision_score(all_y, pred)
        r = metrics.recall_score(all_y, pred)
        return f1, p, r
    else:
        best_th = 0.5
        f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def train_step(train_iter, model, optimizer, scheduler, hp, scaler=None):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        if hp.fp16:
            with amp.autocast():
                if len(batch) == 2:
                    x, y = batch
                    prediction = model(x)
                else:
                    x1, x2, y = batch
                    prediction = model(x1, x2)


                loss = criterion(prediction, y.to(model.device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if len(batch) == 2:
                x, y = batch
                prediction = model(x)
            else:
                x1, x2, y = batch
                prediction = model(x1, x2)

            loss = criterion(prediction, y.to(model.device))
            loss.backward()
            optimizer.step()
            scheduler.step()

        #if hp.fp16:
        #    with amp.scale_loss(loss, optimizer) as scaled_loss:
        #        scaled_loss.backward()
        #else:
        #    loss.backward()
        #optimizer.step()
        #scheduler.step()
        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


def train(trainset, validset, testset, run_tag, hp):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    if type(validset)!= type(None):
        valid_iter = data.DataLoader(dataset=validset,
                                     batch_size=hp.batch_size*16,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=padder)
    if type(testset) != type(None):
        test_iter = data.DataLoader(dataset=testset,
                                     batch_size=hp.batch_size*16,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=padder)
    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug)
    if device == 'cuda':
        model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    if hp.fp16:
        scaler = amp.GradScaler()
        #model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    else:
        scaler=None
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)

    best_dev_f1 = best_test_f1 = 0.0
    results = []
    best_epoch = []
    t_start = time.process_time()
    for epoch in range(1, hp.n_epochs+1):
        t_epoch = time.process_time()
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp, scaler)

        # eval
        model.eval()
        t_train = time.process_time()
        if type(validset)!= type(None):
            dev_f1, th = evaluate(model, valid_iter)
        else:
            dev_f1 = 0.0
            th = 0.5
        t_valid = time.process_time()
        if type(testset) != type(None):
            test_f1, test_p, test_r = evaluate(model, test_iter, threshold=th)
        else:
            test_f1, test_p, test_r = 0.0, 0.0, 0.0

        t_test = time.process_time()
        curr_results = [epoch, test_f1, test_p, test_r, t_train-t_epoch, t_valid-t_train, t_test-t_valid]
        results += [curr_results]
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            best_epoch = curr_results
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

        if (t_test - t_start) + (t_test - t_epoch) > 8*60*60: #if running the next epoch would lead to a total runtime
                                                            #higher than 8 hours, break.
            break

    writer.close()
    results += [best_epoch]
    return model, th, results
