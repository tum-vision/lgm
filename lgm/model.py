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
layered graphical model instantiation from prototype declaration
"""
import abc
from typing import Dict, Tuple, Optional, Hashable
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import proto
from .utils.common import Picklable, flip_local, get_local_links, SavableModel


###############################################################################
# layer part
###############################################################################


class VariableLayer(nn.Module):
    """wrapper of VariableLayer"""
    def __init__(self, layer: proto.VariableLayer) -> None:
        super(VariableLayer, self).__init__()
        self.layer = layer
        self.name = layer.name
        self.size = layer.size
        self.nb_label = layer.nb_label
        self.layer_type = layer.layer_type
        self.neighbors = nn.Module()
        self.unary = None if self.layer_type == proto.LayerType.CND \
            else self._initialize_unary_energy()

    def get_neighbor(self, layername: str) -> 'Neighbor':
        return getattr(self.neighbors, layername)

    def get_all_neighbors(self) -> Dict[str, 'Neighbor']:
        return {n: m for n, m in self.neighbors.named_children()}

    def get_energy(self) -> Optional[torch.Tensor]:
        return self.unary

    def to_repr(self) -> str:
        return 'Layer {0}: {1}, range {2}, size {3}'.format(
                self.name, self.layer_type, self.nb_label, self.size)

    def _initialize_unary_energy(self) -> nn.Parameter:
        return nn.Parameter(torch.randn((self.size, self.nb_label-1)),
                            requires_grad=True)


###############################################################################
# connection part
###############################################################################


class Connection(nn.Module, metaclass=abc.ABCMeta):
    """wrapper of Connection"""
    def __init__(self,
                 connection: proto.Connection,
                 left: VariableLayer,
                 right: VariableLayer) -> None:
        super(Connection, self).__init__()
        self._detached_modules = {'left': left, 'right': right}
        self.connection = connection
        self.name = connection.name
        self.method = connection.method
        self.params = connection.params
        self.binary = self._initialize_binary_energy()

    # abstract methods

    @abc.abstractmethod
    def _initialize_binary_energy(self) -> nn.Parameter:
        raise NotImplementedError

    @abc.abstractmethod
    def flip_energy(self, energy: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # property definitions to break the submodule loop problem for pytorch

    @property
    def left(self) -> VariableLayer:
        return self._detached_modules['left']

    @left.setter
    def left(self, val: VariableLayer) -> None:
        self._detached_modules['left'] = val

    @property
    def right(self) -> VariableLayer:
        return self._detached_modules['right']

    @right.setter
    def right(self, val: VariableLayer) -> None:
        self._detached_modules['right'] = val

    # common methods

    def get_energy(self, flip: bool = False) -> torch.Tensor:
        binary = self.binary
        if flip:
            binary = self.flip_energy(binary)
        return binary

    def to_repr(self) -> str:
        return '{0} connection {1}'.format(self.method, self.name)


class DenseConnection(Connection):
    """Implementation for dense connection"""
    def _initialize_binary_energy(self) -> nn.Parameter:
        return nn.Parameter(torch.randn((self.left.size, self.right.size,
                                         self.left.nb_label-1,
                                         self.right.nb_label-1)),
                            requires_grad=True)

    def flip_energy(self, energy: torch.Tensor) -> torch.Tensor:
        return energy.transpose(0, 1).transpose(2, 3)


class ConvConnection(Connection):
    """Implemntation for convolution connection"""

    def __init__(self,
                 connection: proto.Connection,
                 left: VariableLayer,
                 right: VariableLayer) -> None:
        super().__init__(connection, left, right)
        self._links = (get_local_links(self.params, False),
                       get_local_links(self.params, True))

    def links(self, direct: bool):
        return self._links[direct]

    def _initialize_binary_energy(self) -> nn.Parameter:
        params = self.params
        return nn.Parameter(torch.randn((params.lchannel, params.rchannel,
                                         *params.kshape,
                                         self.left.nb_label-1,
                                         self.right.nb_label-1)),
                            requires_grad=True)

    def flip_energy(self, energy: torch.Tensor) -> torch.Tensor:
        sz = energy.size()
        return torch.flip(energy.view(*sz[0:2], -1, *sz[-2:]), [2]).view(
                    sz).transpose(0, 1).transpose(-2, -1)


class LocalConnection(Connection):
    """Implementation for local connection"""
    def __init__(self,
                 connection: proto.Connection,
                 left: VariableLayer,
                 right: VariableLayer) -> None:
        Connection.__init__(self, connection, left, right)
        params = connection.params
        self.inv_binary_shape = (params.rchannel, params.lchannel,
                                 *params.kshape, *params.lshape,
                                 self.right.nb_label-1, self.left.nb_label-1)
        self._links = (get_local_links(params, False),
                       get_local_links(params, True))

    def links(self, direct: bool):
        return self._links[direct]

    def _initialize_binary_energy(self) -> nn.Parameter:
        params = self.params
        return nn.Parameter(torch.randn((params.lchannel, params.rchannel,
                                         *params.kshape, *params.rshape,
                                         self.left.nb_label-1,
                                         self.right.nb_label-1)),
                            requires_grad=True)

    def flip_energy(self, energy: torch.Tensor) -> torch.Tensor:
        return flip_local(energy.unsqueeze(0).transpose(-2, -1),
                          (1,) + self.inv_binary_shape, 0., *self.links(False)
                          ).squeeze(0)


def connectionBuilder(connection: proto.Connection,
                      left: VariableLayer,
                      right: VariableLayer) -> Connection:
    """factory function to construct appropriate Connection objects"""
    cnnttype = proto.ConnectionType
    method = connection.method
    if method is cnnttype.DENSE:
        return DenseConnection(connection, left, right)
    elif method is cnnttype.CONV:
        return ConvConnection(connection, left, right)
    elif method is cnnttype.LOCAL:
        return LocalConnection(connection, left, right)
    else:
        raise ValueError('Unsupported connection method: {0}.'.format(method))


###############################################################################
# neighbor part
###############################################################################


class Neighbor(nn.Module, metaclass=abc.ABCMeta):
    """abstract class for neighbors of a variable layer"""
    def __init__(self,
                 source: VariableLayer,
                 target: VariableLayer,
                 connection: Connection
                 ) -> None:
        super(Neighbor, self).__init__()
        self._detached_modules = {'source': source, 'target': target,
                                  'connection': connection}
        # if the message is in the same direction as the connection
        self.direct = (self.connection.left == self.source)

    # property definitions to break the submodule loop problem for pytorch

    @property
    def source(self) -> VariableLayer:
        return self._detached_modules['source']

    @source.setter
    def source(self, val: VariableLayer) -> None:
        self._detached_modules['source'] = val

    @property
    def target(self) -> VariableLayer:
        return self._detached_modules['target']

    @target.setter
    def target(self, val: VariableLayer) -> None:
        self._detached_modules['target'] = val

    @property
    def connection(self) -> Connection:
        return self._detached_modules['connection']

    @connection.setter
    def connection(self, val: Connection) -> None:
        self._detached_modules['connection'] = val

    def get_energy(self) -> Variable:
        return self.connection.get_energy(not self.direct)

    def get_inverse_neighbor(self) -> 'Neighbor':
        return self.target.get_neighbor(self.source.name)

    def to_repr(self) -> str:
        return 'Neighbor to layer {0}, with {1}'.format(
                 self.target.name, self.connection.to_repr())


###############################################################################
# model part
###############################################################################


class LayerModel(nn.Module, SavableModel):
    """instantiation of LayerModel prototype"""
    def __init__(self, proto: proto.LayerModel) -> None:
        super(LayerModel, self).__init__()
        # define parameters and sanity check
        self.proto = proto
        self._check_model()
        self.nb_conditioned = len(proto.conditioned_layers)
        self.nb_observable = len(proto.observable_layers)
        self.nb_hidden = len(proto.hidden_layers)
        self.name = proto.name
        # fill up models with necessary parameters
        self.layers = self.build_layers()
        (self.connections,
         self.layers2connection) = self.build_connections()
        self.build_neighbors()

    def get_layer(self, name: str) -> VariableLayer:
        return getattr(self.layers, name)

    def get_connection(self, name: str) -> Connection:
        return getattr(self.connections, name)

    def get_connection_between(self, *layernames: str) -> Connection:
        return getattr(self.connections,
                       self.layers2connection[tuple(sorted(layernames))])

    def get_connection_name_between(self, *layernames: str) -> str:
        return self.layers2connection[tuple(sorted(layernames))]

    def get_neighbor(self, source: str, target: str) -> Neighbor:
        return getattr(self.layers, source).get_neighbor(target)

    def show_summary(self) -> None:
        print('LayerModel{0} with {1} layer(s):'
              ''.format(' (' + self.name + ')' if self.name else '',
                        self.nb_conditioned + self.nb_observable +
                        self.nb_hidden))
        for l in self.layers.children():
            neighbors = l.get_all_neighbors()
            print('  {0} with {1} neighbor(s):'.format(l.to_repr(),
                                                       len(neighbors)))
            for n in neighbors.values():
                print('    {0}'.format(n.to_repr()))

    def _check_model(self) -> None:
        model = self.proto
        closed_neighbor, all_connected = proto.check_model_connectivity(model)
        if not (closed_neighbor and all_connected):
            raise ValueError('Model check failed:' +
                             ('' if closed_neighbor else
                              '\n - neighbor outside of the model.') +
                             ('' if all_connected else
                              '\n - Multiple components.'))

    def build_layers(self) -> nn.Module:
        layers = nn.Module()
        for vlayer in self.proto.layers:
            vl = VariableLayer(vlayer)
            layers.add_module(vl.name, vl)
        return layers

    def build_connections(self) -> Tuple[nn.Module,
                                         Dict[Tuple[str, str], str]]:
        connections = nn.Module()
        lyrs2cnnt = dict()  # type: Dict[Tuple[str, str], str]
        layers = self.layers
        for c in self.proto.named_connections.values():
            cname = c.name
            lname, rname = [l.name for l in c.layers]  # type: str, str
            cnnt = connectionBuilder(c, getattr(layers, lname),
                                     getattr(layers, rname))
            connections.add_module(cname, cnnt)
            lyrs2cnnt[tuple(sorted([lname, rname]))] = cname
        return connections, lyrs2cnnt

    def build_neighbors(self) -> None:
        layers = self.layers
        for layer in layers.children():
            sname = layer.name
            for vlayer, connection in layer.layer.neighbors:
                nname = vlayer.name
                n = Neighbor(getattr(layers, sname), getattr(layers, nname),
                             self.get_connection(
                               self.get_connection_name_between(sname, nname)))
                layer.neighbors.add_module(nname, n)

    def save_model(self, path: str,
                   addons: Optional[Dict[Hashable, Picklable]] = None) -> None:
        if addons is None:
            addons = {}
        dic = {'proto': self.proto,
               'state_dict': self.state_dict(),
               **addons}
        torch.save(dic, path)

    @classmethod
    def load_model(cls, path: str
                   ) -> ('LayerModel', Dict[Hashable, Picklable]):
        dic = torch.load(path)
        protocol = dic['proto']
        state_dict = dic['state_dict']
        model = cls(protocol)
        if state_dict:
            model.load_state_dict(state_dict)
        del dic['proto']
        del dic['state_dict']
        return model, dic
