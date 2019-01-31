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
import torch
from . import custom_ops_cpp
from torch.autograd import Function


# torch.autograd.Function definitions based on cpp modules


class _Logep(Function):
    @staticmethod
    def forward(ctx, t):
        t_out = custom_ops_cpp.logep_forward(t)
        ctx.save_for_backward(t)
        return t_out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = custom_ops_cpp.logep_backward(grad_out, *ctx.saved_tensors)
        return grad_in


class _Unzip0(Function):
    @staticmethod
    def forward(ctx, t, dim):
        t_out = custom_ops_cpp.unzip0_forward(t, dim)
        ctx.saved_nontensors = (dim,)
        return t_out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = custom_ops_cpp.unzip0_backward(grad_out,
                                                 *ctx.saved_nontensors)
        return grad_in, None


class _Softmax0fo(Function):
    @staticmethod
    def forward(ctx, t, dim):
        t_out = custom_ops_cpp.softmax0fo_forward(t, dim)
        ctx.save_for_backward(t_out, t)
        ctx.saved_nontensors = (dim,)
        return t_out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = custom_ops_cpp.softmax0fo_backward(
                        grad_out, *ctx.saved_tensors, *ctx.saved_nontensors)
        return grad_in, None


class _Logsoftmax0fo(Function):
    @staticmethod
    def forward(ctx, t, dim):
        t_out = custom_ops_cpp.logsoftmax0fo_forward(t, dim)
        ctx.save_for_backward(t_out, t)
        ctx.saved_nontensors = (dim,)
        return t_out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = custom_ops_cpp.logsoftmax0fo_backward(
                        grad_out, *ctx.saved_tensors, *ctx.saved_nontensors)
        return grad_in, None


class _Logsumexp0(Function):
    @staticmethod
    def forward(ctx, t, dim, keepdim):
        t_out = custom_ops_cpp.logsumexp0_forward(t, dim, keepdim)
        ctx.save_for_backward(t, t_out)
        ctx.saved_nontensors = (dim, keepdim)
        return t_out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = custom_ops_cpp.logsumexp0_backward(
                        grad_out, *ctx.saved_tensors, *ctx.saved_nontensors)
        return grad_in, None, None


# convenience function wrapper with type annotations


def logep(t: torch.Tensor) -> torch.Tensor:
    return _Logep.apply(t)


def unzip0(t: torch.Tensor, dim: int)-> torch.Tensor:
    return _Unzip0.apply(t, dim)


def softmax0fo(t: torch.Tensor, dim: int)-> torch.Tensor:
    return _Softmax0fo.apply(t, dim)


def logsoftmax0fo(t: torch.Tensor, dim: int)-> torch.Tensor:
    return _Logsoftmax0fo.apply(t, dim)


def logsumexp0(t: torch.Tensor, dim: int, keepdim: bool = False
               )-> torch.Tensor:
    return _Logsumexp0.apply(t, dim, keepdim)
