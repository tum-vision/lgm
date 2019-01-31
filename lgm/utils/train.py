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
training related utilities
"""
import sys
from typing import Tuple, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from .common import get_timestamp, ProgressBar


def step(model: nn.Module, data: Tensor, target: Tensor, use_cuda: bool,
         correct: int, total: int, tot_loss: float,
         optimizer: Optional[optim.Optimizer] = None,
         loss_func: Callable[[Tensor, Tensor], Tensor] = nnf.cross_entropy,
         input_ops: Optional[Callable[[Tensor], Tensor]] = None,
         output_ops: Optional[Callable[..., Tensor]] = None
         ) -> Tuple[float, int, int, float]:
    is_train = optimizer is not None
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    if input_ops is not None:
        data = input_ops(data)
    if is_train:
        optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        output = model(data)
    if output_ops is not None:
        output = output_ops(output)
    pred = output.data.max(1)[1]
    correct += int(pred.eq(target.data.long()).cpu().long().sum())
    total += int(target.data.size()[0])
    loss = loss_func(output, target)
    tot_loss += loss.data.item()
    if is_train:
        loss.backward()
        optimizer.step()
    return loss.data.item(), correct, total, tot_loss


def train_epoch(model: nn.Module,
                optimizer: optim.Optimizer,
                train_loader: DataLoader,
                epoch: int,
                use_cuda: bool,
                loss_func: Callable[[Tensor, Tensor],
                                    Tensor] = nnf.cross_entropy,
                input_ops: Callable[[Tensor], Tensor] = None,
                output_ops: Callable[..., Tensor] = None,
                log_interval=100) -> Tuple[float, float]:
    model.train()
    tot_loss = 0.
    correct = 0
    total = 0
    print()
    for batch_idx, (data, target) in enumerate(train_loader):
        (loss, correct,
         total, tot_loss) = step(model, data, target, use_cuda, correct, total,
                                 tot_loss, optimizer, loss_func, input_ops,
                                 output_ops)
        if batch_idx % log_interval == 0:
            print('Train epoch {0}: [{1}/{2} ({3:.2f}%)]\tLoss: {4:f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), tot_loss / total))
            sys.stdout.flush()
    avg_loss = tot_loss / total
    accuracy = correct / total
    print('\n{0}: Train epoch {1}: mean loss = {2:.6f}, accuracy = {3:f}'
          .format(get_timestamp(), epoch, avg_loss, accuracy))
    sys.stdout.flush()
    return avg_loss, accuracy


def eval_epoch(model: nn.Module,
               val_loader: DataLoader,
               epoch: int,
               use_cuda: bool,
               loss_func: Callable[[Tensor, Tensor],
                                   Tensor] = nnf.cross_entropy,
               input_ops: Callable[[Tensor], Tensor] = None,
               output_ops: Callable[..., Tensor] = None
               ) -> Tuple[float, float]:
    model.eval()
    tot_loss = 0.
    correct = 0
    total = 0
    print('\nEval epoch {0} started: '.format(epoch))
    sys.stdout.flush()
    pb = ProgressBar()
    for batch_idx, (data, target) in enumerate(val_loader):
        (loss, correct,
         total, tot_loss) = step(model, data, target, use_cuda, correct, total,
                                 tot_loss, None, loss_func, input_ops,
                                 output_ops)
        pb.progress(batch_idx / len(val_loader))
    pb.complete()
    avg_loss = tot_loss / total
    accuracy = correct / total
    print('\n{0}: Eval epoch {1}: mean loss = {2:f}, accuracy = {3:f}'
          .format(get_timestamp(), epoch, avg_loss, accuracy))
    sys.stdout.flush()
    return avg_loss, accuracy


def test(model: nn.Module,
         test_loader: DataLoader,
         use_cuda: bool,
         loss_func: Callable[[Tensor, Tensor], Tensor] = nnf.cross_entropy,
         input_ops: Callable[[Tensor], Tensor] = None,
         output_ops: Callable[..., Tensor] = None
         ) -> Tuple[float, float]:
    model.eval()
    tot_loss = 0.
    correct = 0
    total = 0
    print('\nTest started:')
    sys.stdout.flush()
    pb = ProgressBar()
    for batch_idx, (data, target) in enumerate(test_loader):
        (loss, correct,
         total, tot_loss) = step(model, data, target, use_cuda, correct, total,
                                 tot_loss, None, loss_func, input_ops,
                                 output_ops)
        pb.progress(batch_idx / len(test_loader))
    pb.complete()
    avg_loss = tot_loss / total
    accuracy = correct / total
    print('\n{0}: Test result: mean loss = {1:f}, accuracy = {2:f}'
          .format(get_timestamp(), avg_loss, accuracy))
    sys.stdout.flush()
    return avg_loss, accuracy
