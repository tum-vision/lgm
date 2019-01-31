#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# This file is part of LGM.
#
# Copyright 2018 Yuesong Shen
# Copyright 2018,2019 Technical University of Munich
#
# Developed by Yuesong Shen <yuesong dot shen at tum dot de>.
#
# If you use this file for your research, please cite the following paper:
#
# "Probabilistic Discriminative Learning with Layered Graphical Models" by
# Yuesong Shen, Tao Wu, Csaba Domokos and Daniel Cremers
#
# LGM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LGM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LGM. If not, see <http://www.gnu.org/licenses/>.
###############################################################################
"""
Conventional NNs as references
"""
import os
import argparse
from typing import Tuple, Optional, List, Callable
import torch
from torch import Tensor
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as nnf
from lgm.utils.common import (get_timestamp, Log, check_cuda, mkdirp,
                              display_param_stats, rm, EarlyStopper,
                              Conv2dLocal)
from lgm.utils.data.mnist import get_MNIST_dataloaders
from lgm.utils.train import train_epoch, eval_epoch, test


IN_SIZE = 28 * 28
OUT_SIZE = 10
LYR_DIM_0 = 28
LYR_DIM_1 = 13


class MLP(nn.Module):
    def __init__(self, layers: List[int], nlf: Callable[[Tensor], Tensor]
                 ) -> None:
        super(MLP, self).__init__()
        self.layers = [IN_SIZE] + layers
        self.nlf = nlf
        self.fcs = nn.ModuleList()
        for i in range(len(self.layers)-1):
            self.fcs.append(nn.Linear(self.layers[i], self.layers[i+1]))
        self.fc_out = nn.Linear(self.layers[-1], OUT_SIZE)

    def forward(self, x: List[Tensor]) -> Tensor:
        for i in range(len(self.layers) - 1):
            x = self.nlf(self.fcs[i](x))
        return nnf.log_softmax(self.fc_out(x), -1)


class LocalMininet(nn.Module):
    def __init__(self, nlf: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.nlf = nlf
        self.layers = None  # hacky way of saying it is LocalMininet
        self.local_l1 = Conv2dLocal(
            LYR_DIM_0, LYR_DIM_0, 1, 6, (5, 5), stride=(2, 2), padding=(1, 1))
        self.local_l2 = Conv2dLocal(
            LYR_DIM_1, LYR_DIM_1, 6, 16, (5, 5), stride=(2, 2))
        self.fc_h = nn.Linear(400, 100)
        self.fc_out = nn.Linear(100, OUT_SIZE)

    def forward(self, x: List[Tensor]) -> Tensor:
        x = self.nlf(self.local_l1(x))
        x = self.nlf(self.local_l2(x))
        x = x.view(x.size(0), -1)
        x = self.nlf(self.fc_h(x))
        return nnf.log_softmax(self.fc_out(x), -1)


def training_backup(model: nn.Module, optimizer: optim.Optimizer, path: str,
                    optim_kwargs=None) -> None:
    layers = model.layers
    if optim_kwargs is None:
        optim_kwargs = {}
    if not isinstance(layers, bool):
        layers = layers[1:]
    dic = {'state_dict': model.state_dict(),
           'layers': layers,
           'nlf': model.nlf,
           'optim_type': optimizer.__class__.__name__,
           'optim_state_dict': optimizer.state_dict(),
           'optim_kwargs': optim_kwargs}
    torch.save(dic, path)


def training_resume(path: str, use_cuda: bool
                    ) -> Tuple[nn.Module, optim.Optimizer]:
    dic = torch.load(path)
    nlf = dic['nlf']
    layers = dic['layers']
    if layers is None:
        model = LocalMininet[layers](nlf)
    else:
        model = MLP(layers, nlf)
    model.load_state_dict(dic['state_dict'])
    if use_cuda:
        model.cuda()
    else:
        model.cpu()
    optimizer = optim.__dict__[dic['optim_type']](
            model.parameters(), **dic['optim_kwargs'])
    optimizer.load_state_dict(dic['optim_state_dict'])
    return model, optimizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        choices=('dense', 'local'),
                        default='local',
                        help='Specify the NN model to run. Can be either dense'
                             ' or local. By default local.')
    parser.add_argument('-d', '--dataset',
                        choices=('MNIST', 'FashionMNIST'),
                        default='MNIST',
                        help='Specify the dataset to run. Can be either MNIST '
                             'or FashionMNIST. By default MNIST.')
    parser.add_argument('-a', '--activation',
                        choices=('relu', 'sigmoid'),
                        default='sigmoid',
                        help='Specify the activation function. Can be either '
                             'relu or sigmoid. By default sigmoid.')
    parser.add_argument('-g', '--gpu',
                        action='store_const',
                        const=True,
                        default=False,
                        help='Enable cuda usage. By default cpu only.')
    parser.add_argument('-b', '--batch',
                        type=int,
                        default=20,
                        help='Specify the batchsize. By default 20.')
    parser.add_argument('-e', '--epoch',
                        type=int,
                        help='Specify the number of epochs to run. Use early'
                             ' stopping by default.')
    parser.add_argument('-p', '--patience',
                        type=int,
                        default=5,
                        help='Specify the number of epochs to wait before '
                             'early stopping. By dafault 5. Will be ignored if'
                             ' --epoch is specified.')
    return parser.parse_args()


if __name__ == '__main__':

    # global parameters

    args = get_args()
    use_cuda = args.gpu

    epochs = args.epoch  # int or None: set to None to enjoy early stopping
    patience = args.patience

    hidden_layers = {'dense': [100],
                     'local': None}[args.model]  # int list: MLP / None: local
    activ_func = torch.__dict__[args.activation]  # torch.relu / torch.sigmoid
    dataset_flavor = args.dataset  # 'MNIST' / 'FashionMNIST'

    optim_type = 'Adam'
    optim_kwargs = {}

    train_val_split = 0.8
    train_batch = args.batch
    val_batch = args.batch
    test_batch = args.batch

    data_dir = {'MNIST': 'data/MNIST/',
                'FashionMNIST': 'data/FashionMNIST/'}[dataset_flavor]
    base_dir = __file__[:-3] + '/'
    save_dir = base_dir + 'model/'
    log_dir = base_dir + 'log/'
    resume_from = None  # save_dir + 'xxx.pickle'

    # create dirs if not there already

    mkdirp(data_dir)
    mkdirp(save_dir)
    mkdirp(log_dir)

    # prepare proper loss function

    lossfunc = nnf.nll_loss

    # start logger

    log_file = '{0}_{1}_{2}_{3}.log'.format(
        "LocalMininet" if hidden_layers is None else '-'.join(
            [str(i) for i in hidden_layers]),
        '-'.join([str(i) for i in (train_batch, val_batch, test_batch,
                                   activ_func.__name__)]),
        dataset_flavor, get_timestamp())
    log_title = log_file[:-4]
    logger = Log(log_dir + log_file)
    logger.start(log_title)
    logger.start_intercept()

    # check cuda availablility when needed

    if use_cuda:
        check_cuda()

    # set up dataset

    if dataset_flavor == 'MNIST':
        ((train_loader, val_loader, test_loader),
         (nb_train, nb_val, nb_test)) = get_MNIST_dataloaders(
            data_dir, train_batch, val_batch, test_batch, train_val_split,
            use_cuda, 'MNIST', keep_shape=(hidden_layers is None))
    elif dataset_flavor == 'FashionMNIST':
        ((train_loader, val_loader, test_loader),
         (nb_train, nb_val, nb_test)) = get_MNIST_dataloaders(
            data_dir, train_batch, val_batch, test_batch, train_val_split,
            use_cuda, 'FashionMNIST', keep_shape=(hidden_layers is None))
    else:
        raise Exception('Unknown dataset: {}'.format(dataset_flavor))
    print('dataset: {}, location: {}'.format(dataset_flavor, data_dir))
    print('sample / batch number for training:  ',
          nb_train, len(train_loader))
    print('sample / batch number for validation:',
          nb_val, len(val_loader))
    print('sample / batch number for testing:   ',
          nb_test, len(test_loader))
    print('train / val / test batchsizes: {} / {} / {}'.format(
        train_batch, val_batch, test_batch))
    print('non-linearity: {}'.format(
        activ_func.__name__))

    # load the model and optimizer

    if resume_from is None or not os.path.exists(resume_from):
        print('initializing model ...')
        # set up model
        if hidden_layers is None:
            ref_model = LocalMininet(activ_func)
        else:
            ref_model = MLP(hidden_layers, activ_func)
        if use_cuda:
            ref_model.cuda()
        # set up optimizer
        optimizer = optim.__dict__[optim_type](ref_model.parameters(),
                                               **optim_kwargs)
    else:
        print('Resume training from {0} ...'.format(resume_from))
        ref_model, optimizer = training_resume(resume_from, use_cuda)
    display_param_stats(ref_model)

    # training part

    def update_backup(backup: Optional[str], i: int, time_stamp: str) -> str:
        tmp = save_dir + '{0}_{1}_{2}.pickle'.format(log_title, i, time_stamp)
        training_backup(ref_model, optimizer, tmp, optim_kwargs)
        if backup is not None:
            if not rm(backup):
                print('Failed to delete {0}'.format(backup))
        return tmp


    def do_train_epoch(i: int) -> Tuple[float, str]:
        train_epoch(ref_model, optimizer, train_loader, i, use_cuda,
                    loss_func=lossfunc, log_interval=100)
        time_stamp = get_timestamp()
        avg_loss, _ = eval_epoch(ref_model, val_loader, i, use_cuda,
                                 loss_func=lossfunc)
        return avg_loss, time_stamp


    backup = None
    if epochs is None:  # use early stopping and backup only the best one
        i = 1
        earlystop = EarlyStopper(patience=patience, should_decrease=True)
        while earlystop.passes():
            avg_loss, time_stamp = do_train_epoch(i)
            isbest = earlystop.update(avg_loss)
            if isbest:
                backup = update_backup(backup, i, time_stamp)
            i += 1
        # revert to the best one for testing
        ref_model, _ = training_resume(backup, use_cuda)
    else:  # learning with fixed epochs and backup weights for each epoch
        for i in range(1, epochs + 1):
            _, time_stamp = do_train_epoch(i)
            backup = update_backup(None, i, time_stamp)

    # testing part

    test(ref_model, test_loader, use_cuda, loss_func=lossfunc)

    # stop logger

    logger.close()
