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
from typing import cast
import numpy as np
import torch
from ...utils.common import flip_local
from ...proto import ConnectionType as CType, LocalParams
from ...model import Connection
from . import common as cc


# here we assume that the mapping is from left to right for each connection


class ConnectionWrapper(cc.ConnectionWrapper):
    def __init__(self, connection: Connection, left: cc.LayerWrapper,
                 right: cc.LayerWrapper) -> None:
        super(ConnectionWrapper, self).__init__(connection, left, right)
        self.register_buffer('edge_proba', self._inverse_edge_proba())

    def _inverse_edge_proba(self) -> torch.Tensor:
        method = self.method
        if method is CType.DENSE:
            return torch.tensor(1. / self.right.size).type(self.binary.dtype)
        elif method is CType.LOCAL or method is CType.CONV:
            return torch.from_numpy(1. / self._local_left_nodewise_reach()
                                    ).type(self.binary.dtype)
        else:
            raise RuntimeError('Unrecognized connection type: {}'.format(
                method))

    def _local_left_nodewise_reach(self) -> np.ndarray:
        params = cast(LocalParams, self.params)
        kshape = np.array(params.kshape)
        conv_dim = params.dim
        dilation = np.array(params.dilation)
        doffset = np.array(params.doubleoffset)
        upperoffset = (dilation * (kshape - 1) + doffset) / 2
        loweroffset = (dilation * (1 - kshape) + doffset) / 2
        stride = np.array(params.stride).reshape([-1] + [1] * conv_dim)
        upperoffset = upperoffset.reshape([-1] + [1] * conv_dim)
        loweroffset = loweroffset.reshape([-1] + [1] * conv_dim)
        lshape = params.lshape
        rshape = params.rshape
        rchannel = params.rchannel
        indices = np.stack(np.meshgrid(*[np.arange(i) for i in lshape]))
        ubound = np.array(rshape).reshape([-1] + [1] * conv_dim) - 1
        return np.prod(
            np.floor(np.minimum(ubound, (indices - loweroffset) / stride)) -
            np.ceil(np.maximum(0, (indices - upperoffset) / stride)) + 1, 0
        ) * rchannel


class NeighborWrapper(cc.NeighborWrapper, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def edge_proba(self) -> torch.Tensor:
        raise NotImplementedError


class DenseNeighborWrapper(NeighborWrapper, metaclass=abc.ABCMeta):
    @property
    def edge_proba(self) -> torch.Tensor:
        return self.connection.edge_proba

    def get_energy(self) -> torch.Tensor:
        return super().get_energy() / self.edge_proba

    def get_upd_contrib(self) -> torch.Tensor:
        return super().get_upd_contrib() * self.edge_proba


class LocalNeighborWrapper(NeighborWrapper, metaclass=abc.ABCMeta):
    def __init__(self, source: cc.LayerWrapper, target: cc.LayerWrapper,
                 connection: cc.ConnectionWrapper) -> None:
        super().__init__(source, target, connection)
        self.params = cast(LocalParams, self.params)

    @property
    def edge_proba(self):
        params = self.params
        dim, lshape, kshape = params.dim, params.lshape, params.kshape
        lchannel, rchannel = params.lchannel, params.rchannel
        rlshape = (1, rchannel, lchannel) + kshape + lshape
        ep = self.connection.edge_proba[(None,) * (dim + 3) + (Ellipsis,)]
        ep = ep.expand(rlshape)
        if self.direct:
            lrshape = (1, lchannel, rchannel) + kshape + params.rshape
            ep = flip_local(ep, lrshape, 0., *self.links(self.direct))
        return ep.squeeze(0)

    def get_energy(self) -> torch.Tensor:
        return super().get_energy() / self.edge_proba[..., None, None]

    def get_upd_contrib(self) -> torch.Tensor:
        msg = self.message
        dim = self.params.dim
        sz = msg.size()
        final_size = (sz[0], -1, sz[-1])
        edge_proba = self.edge_proba[None, ..., None]
        return torch.sum((msg * edge_proba).view(
            *sz[0:3], -1, *sz[3 + dim:]), dim=(1, 3)).view(*final_size)
