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
common utilities
"""
import abc
import math
import sys
import os
import shutil
import time
import itertools
import collections
from typing import (Iterable, Any, Iterator, TypeVar, Tuple, Optional,
                    Dict, Hashable, Callable, Sequence)
from functools import reduce
import pathlib
import numpy as np

import torch
from torch import Tensor
from torch import optim
import torch.nn as nn
import torch.nn.functional as nnf

# TODO: to be replaced with better type definition
T = TypeVar('T')
Picklable = Any


###############################################################################
# pytorch independent part
###############################################################################


class AlwaysGet(Sequence[T]):
    def __init__(self, val: T) -> None:
        self.val = val

    def __getitem__(self, item: Any) -> T:
        return self.val

    def __iter__(self) -> Iterator[T]:
        return itertools.repeat(self.val)

    def __len__(self) -> int:
        return 1


def full_class_name(cls: type) -> str:
    return '.'.join([cls.__module__, cls.__name__])


def shape2size(shape: Sequence[int]) -> int:
    return reduce(lambda x, y: x * y, shape)


def rm(filename: str) -> bool:
    try:
        os.remove(filename)
        return True
    except OSError:
        return False


def mkdir(dirpath: str, parents: bool = False, exist_ok: bool = False) -> None:
    pathlib.Path(dirpath).mkdir(parents=parents, exist_ok=exist_ok)


def mkdirp(dirpath: str) -> None:
    mkdir(dirpath, parents=True, exist_ok=True)


def cp(source: str, destination: str) -> None:
    shutil.copy(source, destination)


def get_timestamp() -> str:
    """generate time stamp of current time"""
    return time.strftime('%y%m%d%H%M%S')


class Tee(object):
    """imitation of the tee command"""
    def __init__(self, source, destination) -> None:
        self.source = source
        self.destination = destination
        self.flush()

    def write(self, msg: str) -> None:
        self.destination.write(msg)
        self.source.write(msg)

    def flush(self) -> None:
        self.destination.flush()
        self.source.flush()


class Log(object):
    """an easy implementation of logger"""
    _stdout = sys.stdout
    _stderr = sys.stderr

    def __init__(self, path: str) -> None:
        self.path = path
        self.logfile = None

    def start(self, title: str, overwrite: bool = False) -> None:
        self.logfile = open(self.path, 'w' if overwrite else 'a')
        self.logfile.write('\n{0}: starting log entry {1}\n\n'.format(
                get_timestamp(), title))
        self.logfile.flush()

    def write(self, content: str, end='\n') -> str:
        if self.logfile is None:
            raise Exception('Unable to write to closed log file.')
        timestamp = get_timestamp()
        self.logfile.write('{0}: {1}{2}'.format(
                timestamp, content, end))
        self.logfile.flush()
        return timestamp

    def start_intercept(self,
                        target_stdout: bool = True,
                        target_stderr: bool = True,
                        mute_stdout: bool = False,
                        mute_stderr: bool = False) -> None:
        """start logging content from stdout or stderr"""
        if not target_stdout and not target_stderr:
            return
        if self.logfile is None:
            raise Exception('Unable to start with closed log file.')
        if target_stdout:
            sys.stdout = self.logfile if mute_stdout else Tee(sys.stdout,
                                                              self.logfile)
        if target_stderr:
            sys.stderr = self.logfile if mute_stderr else Tee(sys.stderr,
                                                              self.logfile)

    @staticmethod
    def stop_intercept() -> None:
        sys.stdout = Log._stdout
        sys.stderr = Log._stderr

    def close(self) -> None:
        self.stop_intercept()
        if self.logfile is not None:
            self.logfile.close()
            self.logfile = None


class ProgressBar(object):
    """an easy implementation indicating the progress in console"""
    pattern = '....1....2....3....4....5....6....7....8....9....O'
    ptn_len = len(pattern)

    def __init__(self) -> None:
        self.pt = 0
        print('Pattern: [' + ProgressBar.pattern+']')
        print('Progress: ', end='')

    def progress(self, ratio: float) -> None:
        if ratio > 1.0 or ratio < 0.:
            raise ValueError('ratio should be between 0 and 1')
        newpt = int(ProgressBar.ptn_len * ratio)
        if newpt > self.pt:
            print(ProgressBar.pattern[self.pt:newpt], end='')
            sys.stdout.flush()
            self.pt = newpt

    def complete(self) -> None:
        self.progress(1.)
        print()


class EarlyStopper(object):
    def __init__(self, patience: int = 3, should_decrease: bool = True
                 ) -> None:
        self.patience = patience
        self.should_decrease = should_decrease
        self.current = float('Inf') if should_decrease else -float('Inf')
        self.strike = 0

    def update(self, value: float) -> bool:
        if (self.current > value) == self.should_decrease:
            self.strike = 0
            self.current = value
            print('EarlyStopper: Received better result.')
            return True
        else:
            self.strike += 1
            if self.strike > self.patience:
                print('EarlyStopper: Should stop now.')
            else:
                print('EarlyStopper: Strike {} / {}.'.format(self.strike,
                                                             self.patience))
            return False

    def passes(self) -> bool:
        return self.strike <= self.patience


###############################################################################
# pytorch dependent part
###############################################################################


# Code from https://github.com/pytorch/pytorch/pull/1583
#
# ==== LICENSE from PyTorch ====
#
# From PyTorch:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

def conv2d_local(input, weight, bias=None, padding=0, stride=1, dilation=1):
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors "
                                  "supported (got {}D)".format(input.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors "
                                  "supported (got {}D)".format(weight.dim()))

    outH, outW, outC, inC, kH, kW = weight.size()
    kernel_size = (kH, kW)

    # N x [inC * kH * kW] x [outH * outW]
    cols = nnf.unfold(
        input, kernel_size, dilation=dilation, padding=padding, stride=stride)
    cols = cols.view(
        cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)

    out = torch.matmul(
        cols, weight.view(outH * outW, outC, inC * kH * kW).permute(0, 2, 1))
    out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)

    if bias is not None:
        out = out + bias.expand_as(out)
    return out


def _pair(x):
    if isinstance(x, collections.Iterable):
        return x
    return tuple(itertools.repeat(x, 2))


class Conv2dLocal(nn.Module):
    def __init__(self, in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2dLocal, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] -
             self.dilation[0] * (self.kernel_size[0] - 1) - 1) /
            self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] *
             (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.weight = nn.Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input):
        return conv2d_local(input, self.weight, self.bias, stride=self.stride,
                            padding=self.padding, dilation=self.dilation)

# End of code from https://github.com/pytorch/pytorch/pull/1583


def as_numpy_dtype(dt: torch.dtype):
    return np.dtype(str(dt)[6:])


# constants that depend on PyTorch default precision settings
_DEFAULT_DTYPE = as_numpy_dtype(torch.get_default_dtype())
_DEFAULT_FINFO = np.finfo(_DEFAULT_DTYPE)
EPSILON = np.asscalar(_DEFAULT_FINFO.tiny)
MAXFLOAT = np.asscalar(_DEFAULT_FINFO.max)
MINFLOAT = np.asscalar(_DEFAULT_FINFO.min)
MAXPREEXP = np.asscalar(_DEFAULT_FINFO.maxexp * np.log(2))
MINPREEXP = np.asscalar(_DEFAULT_FINFO.minexp * np.log(2))


def check_cuda() -> None:
    # check availability
    if not torch.cuda.is_available():
        raise Exception('No CUDA device available')
    # show all available GPUs
    cuda_count = torch.cuda.device_count()
    print('{0} CUDA device(s) available:'.format(cuda_count))
    for i in range(cuda_count):
        print('- {0}: {1} ({2})'.format(i, torch.cuda.get_device_name(i),
                                        torch.cuda.get_device_capability(i)))
    # showing current cuda device
    curr_idx = torch.cuda.current_device()
    print('Currently using device {0}'.format(curr_idx))


def apply_batch(batch, fn):
    if isinstance(batch, Tensor):
        return fn(batch)
    elif isinstance(batch, collections.Mapping):
        return {k: apply_batch(sample, fn) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [apply_batch(sample, fn) for sample in batch]
    else:
        return batch


def flip_local(var: Tensor, inverse_shape: Iterable[int], default_val: float,
               indices_src, indices_tgt) -> Tensor:
    ret = var.new_full(inverse_shape, default_val, requires_grad=False)
    for sidx, tidx in zip(indices_src, indices_tgt):
        ret[tidx] = torch.transpose(var[sidx], 1, 2)
    return ret


def get_local_links(params: Any,
                    direct: bool,  # if output depict direct local connection
                    ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    doubledelta = params.doubleoffset
    stride = params.stride
    dilation = params.dilation
    lshape = params.lshape
    rshape = params.rshape
    kshape = params.kshape
    indices_src = []
    indices_tgt = []
    head_inds = [slice(None)] * 3  # for: batch, channel1, channel2
    if direct:
        for kindices in itertools.product(*[range(K) for K in kshape]):
            offsets = [-nd * k - (nd * (1 - K) + d2) // 2
                       for nd, k, K, d2 in zip(dilation, kindices,
                                               kshape, doubledelta)]
            rstarts = [max(0, (of + ns - 1) // ns)
                       for of, ns in zip(offsets, stride)]
            mstarts = [ns * s_ - of
                       for s_, ns, of in zip(rstarts, stride, offsets)]
            rends = [min(sq - 1, (sp - 1 + of) // ns) + 1
                     for sp, sq, of, ns in zip(lshape, rshape, offsets,
                                               stride)]
            mends = [ns * (s_ - 1) - of + 1
                     for s_, ns, of in zip(rends, stride, offsets)]
            rslices = [slice(s, e) for s, e in zip(rstarts, rends)]
            mslices = [slice(s, e, j)
                       for s, e, j in zip(mstarts, mends, stride)]
            k_indices = [K - k - 1 for k, K in zip(kindices, kshape)]
            indices_tgt.append(tuple(head_inds + [*kindices] + rslices))
            indices_src.append(tuple(head_inds + [*k_indices] + mslices))
    else:
        for k_indices in itertools.product(*[range(K) for K in kshape]):
            offsets = [-nd * k_ - (nd * (1 - K) - d2) // 2
                       for nd, k_, K, d2 in zip(dilation, k_indices,
                                                kshape, doubledelta)]
            rstarts = [max(of % ns, of)
                       for of, ns in zip(offsets, stride)]
            mstarts = [(s - of) // ns
                       for s, ns, of in zip(rstarts, stride, offsets)]
            rends = [min(sp - 1, ns * (sq - 1) + of) + 1
                     for sp, sq, of, ns in zip(lshape, rshape, offsets,
                                               stride)]
            mends = [(s - 1 - of) // ns + 1
                     for s, ns, of in zip(rends, stride, offsets)]
            rslices = [slice(s, e, j) for s, e, j in zip(rstarts, rends,
                                                         stride)]
            mslices = [slice(s, e) for s, e in zip(mstarts, mends)]
            kindices = [K - k_ - 1 for k_, K in zip(k_indices, kshape)]
            indices_tgt.append(tuple(head_inds + [*k_indices] + rslices))
            indices_src.append(tuple(head_inds + [*kindices] + mslices))
    return tuple(indices_src), tuple(indices_tgt)


def unzip_one(t: Tensor,
              dim: int = -1) -> Tensor:
    """unzips normalized entities with completion"""
    lastpart = 1. - torch.sum(t, dim, keepdim=True)
    lastpart[lastpart < 0] = 0
    return torch.cat((t, lastpart), dim)


class SavableModel(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def save_model(self, path: str,
                   addons: Optional[Dict[Hashable, Picklable]]=None) -> None:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load_model(cls, path: str
                   ) -> ('SavableModel', Dict[Hashable, Picklable]):
        raise NotImplementedError


def training_backup(model: SavableModel, optimizer: optim.Optimizer, path: str,
                    **kwargs) -> None:
    addon_dict = {'optim_type': optimizer.__class__.__name__,
                  'optim_state_dict': optimizer.state_dict(), **kwargs}
    if 'optim_kwargs' not in addon_dict:
        addon_dict['optim_kwargs'] = {}
    model.save_model(path, addon_dict)


def training_resume(path: str, use_cuda: bool, load_func: Callable[
                        [str], Tuple[SavableModel, Dict[Hashable, Picklable]]]
                    ) -> Tuple[SavableModel, optim.Optimizer]:
    model, addon_dict = load_func(path)
    if use_cuda:
        model.cuda()
    else:
        model.cpu()
    optimizer = optim.__dict__[addon_dict['optim_type']](
            model.parameters(), **addon_dict['optim_kwargs'])
    optimizer.load_state_dict(addon_dict['optim_state_dict'])
    return model, optimizer


def display_param_stats(model: nn.Module) -> None:
    tot = 0
    print('\nParam stats:')
    for n, p in model.named_parameters():
        print(n, 'size:', p.numel(), 'shape:', tuple(p.size()))
        tot += p.numel()
    print('Total params:', tot)
