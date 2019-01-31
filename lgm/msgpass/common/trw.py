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
common parts for tree-reweighted message passing algorithm
"""
import abc
import numpy as np
import torch
from ...utils.common import flip_local
from . import common as cc


class DenseNeighborWrapper(cc.NeighborWrapper, metaclass=abc.ABCMeta):
    def __init__(self, source: cc.LayerWrapper, target: cc.LayerWrapper,
                 connection: cc.ConnectionWrapper) -> None:
        super().__init__(source, target, connection)
        self.register_buffer('_edge_proba', torch.tensor(
                                                1. / connection.right.size))

    def get_energy(self) -> torch.Tensor:
        return super().get_energy() / self._edge_proba

    def get_upd_contrib(self) -> torch.Tensor:
        return super().get_upd_contrib() * self._edge_proba


class LocalNeighborWrapper(cc.NeighborWrapper, metaclass=abc.ABCMeta):
    def __init__(self, source: cc.LayerWrapper, target: cc.LayerWrapper,
                 connection: cc.ConnectionWrapper) -> None:
        super().__init__(source, target, connection)
        self.links = self.connection.connection.links
        self.register_buffer('_edge_proba', self._inverse_edge_proba())

    def _inverse_edge_proba(self) -> torch.Tensor:
        params = self.params
        shapes = (
            (1, params.lchannel, params.rchannel,)+params.kshape+params.rshape,
            (1, params.rchannel, params.lchannel,)+params.kshape+params.lshape)
        ep = self._nodewise_reach()[(None, None, slice(None)) +
                                    ((None,) * params.dim) + (Ellipsis,)]
        ep = ep.expand(shapes[self.direct])
        return flip_local(ep, shapes[1-self.direct], 0.,
                          *self.links(self.direct)).squeeze(0)

    def _nodewise_reach(self) -> torch.Tensor:
        params = self.params
        kshape = np.array(params.kshape)
        conv_dim = len(kshape)
        dilation = np.array(params.dilation)
        doffset = np.array(params.doubleoffset)
        upperoffset = (dilation * (kshape - 1) + doffset) / 2
        loweroffset = (dilation * (1 - kshape) + doffset) / 2
        stride = np.array(np.array(params.stride)).reshape([-1]+[1]*conv_dim)
        upperoffset = np.array(upperoffset).reshape([-1]+[1]*conv_dim)
        loweroffset = np.array(loweroffset).reshape([-1]+[1]*conv_dim)
        lshape = params.lshape
        rshape = params.rshape
        if self.direct:
            channel = params.lchannel
            indices = np.stack(np.meshgrid(*[np.arange(i) for i in lshape]))
            ubound = np.array(rshape).reshape([-1]+[1]*conv_dim) - 1
            retu = np.prod(np.floor(np.minimum(
                                  ubound, (indices - upperoffset) / stride))
                           - np.ceil(np.maximum(
                                  0, (indices - loweroffset) / stride)) + 1, 0)
        else:
            channel = params.rchannel
            indices = np.stack(np.meshgrid(*[np.arange(i) for i in rshape]))
            ubound = np.array(lshape).reshape([-1]+[1]*conv_dim) - 1
            retu = np.prod(np.minimum(ubound, indices * stride + upperoffset)
                           - np.maximum(0, indices * stride + loweroffset) + 1,
                           0)
        return torch.from_numpy(
            1. / np.repeat(np.expand_dims(retu, 0), channel, axis=0)).type(
                                                    torch.get_default_dtype())

    def get_energy(self) -> torch.Tensor:
        return super().get_energy() / self._edge_proba

    def get_upd_contrib(self) -> torch.Tensor:
        msg = self.message
        conv_dim = self.params.dim
        sz = msg.size()
        final_size = (sz[0], -1, sz[-1])
        edge_proba = self._edge_proba[None, ..., None]
        return torch.sum((msg * edge_proba).view(
            *sz[0:3], -1, *sz[3+conv_dim:]), dim=(1, 3)).view(*final_size)
