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
MNIST / FashionMNIST related utilities
"""
from typing import Tuple
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


def get_MNIST_dataloaders(data_dir: str,
                          train_batch: int,
                          val_batch: int,
                          test_batch: int,
                          train_val_split: float,
                          use_cuda: bool,
                          flavor: str = 'MNIST',
                          keep_shape: bool = False,
                          pin_memory: bool = True
                          ) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader],
                                     Tuple[int, int, int]]:
    assert flavor in ['MNIST', 'FashionMNIST']
    dataset = datasets.__dict__[flavor]
    kwargs = {'num_workers': 1, 'pin_memory': pin_memory} if use_cuda else {}
    tf_totensor = \
        transforms.ToTensor() if keep_shape \
        else transforms.Compose([transforms.ToTensor(),
                                 transforms.Lambda(lambda d: d.view(-1))])
    train_data = dataset(data_dir, train=True, download=True,
                         transform=tf_totensor)
    test_data = dataset(data_dir, train=False, download=True,
                        transform=tf_totensor)
    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))
    train_loader = DataLoader(train_data, batch_size=train_batch,
                              sampler=SubsetRandomSampler(train_indices),
                              **kwargs)
    val_loader = DataLoader(train_data, batch_size=val_batch,
                            sampler=SubsetRandomSampler(val_indices), **kwargs)
    test_loader = DataLoader(test_data, batch_size=test_batch, **kwargs)
    return ((train_loader, val_loader, test_loader),
            (nb_train, len(train_data) - nb_train, len(test_data)))
