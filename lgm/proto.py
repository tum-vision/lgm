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
Prototype declaration for layered graphical model
"""
from queue import Queue
from enum import Enum
from typing import Tuple, Any, Dict, Union, Optional, Sequence, List
from .utils.common import shape2size


Vlayers = Sequence['VariableLayer']


class LayerType(str, Enum):
    CND = 'conditioned'
    OBS = 'observable'
    HID = 'hidden'


class ConnectionType(str, Enum):
    DENSE = 'dense'
    LOCAL = 'local'


class LayerParams(object):
    def __init__(self, size: Union[int, Sequence], nb_label: int,
                 layer_type: LayerType = LayerType.OBS) -> None:
        if isinstance(size, int):
            self.size = size  # type: int
            self.shape = (size,)  # type: Tuple[int, ...]
        else:
            self.size = shape2size(size)  # type: int
            self.shape = tuple(size)  # type: Tuple[int, ...]
        self.layer_type = layer_type  # type: LayerType
        self.nb_label = nb_label  # type: int


class ConnectionParams(object):
    """default empty connection parameter class"""
    def __init__(self, method: ConnectionType, left_params: LayerParams,
                 right_params: LayerParams, *args, **kwargs) -> None:
        self.method = method
        self.left_params = left_params
        self.right_params = right_params

    @staticmethod
    def parse(method: ConnectionType, left_params: LayerParams,
              right_params: LayerParams, *args: Any, **kwargs: Any
              ) -> 'ConnectionParams':
        if method is ConnectionType.DENSE:
            return DenseParams(left_params, right_params)
        elif method is ConnectionType.LOCAL:
            return LocalParams(left_params, right_params, *args, **kwargs)
        else:
            return ConnectionParams(
                method, left_params, right_params, *args, **kwargs)


class DenseParams(ConnectionParams):
    def __init__(self, left_params: LayerParams, right_params: LayerParams
                 ) -> None:
        ConnectionParams.__init__(
            self, ConnectionType.DENSE, left_params, right_params)


class LocalParams(ConnectionParams):
    """local connection parameter class"""
    def __init__(self,
                 left_params: LayerParams,
                 right_params: LayerParams,
                 kernel_shape: Sequence[int],
                 doubleoffset: Optional[Union[int, Sequence[int]]] = None,
                 stride: Union[int, Sequence[int]] = 1,
                 dilation: Union[int, Sequence[int]] = 1) -> None:
        ConnectionParams.__init__(
            self, ConnectionType.LOCAL, left_params, right_params)
        lshape, rshape = left_params.shape, right_params.shape
        # check dimension correspondence
        if len(lshape) != len(rshape):
            raise ValueError('Connection needs layers of the same dimension, '
                             'received {0} and {1}'.format(lshape, rshape))
        if len(lshape) < 2:
            raise ValueError('need at least 2 dimensions to have channel '
                             'size and convolution shape')
        self.lchannel = lshape[0]  # channel size of left layer
        self.lshape = tuple(lshape[1:])  # shape of left layer
        self.rchannel = rshape[0]  # channel size of right layer
        self.rshape = tuple(rshape[1:])  # shape of right layer
        # check kernel dimension
        dim = len(lshape) - 1
        self.dim = dim
        if len(kernel_shape) != dim:
            raise ValueError('Kernel should be of dimension {0}, found {1}'
                             .format(dim, len(kernel_shape)))
        self.kshape = tuple(kernel_shape)  # shape of kernel
        # parse and check stride
        if isinstance(stride, int):
            stride = [stride] * dim
        elif len(stride) != dim:
            raise ValueError('Stride should be of dimension {0}, found {1}'
                             .format(dim, len(stride)))
        self.stride = tuple(stride)
        # parse and check doubleoffset (= 2 * offset to always be integer)
        if doubleoffset is None:  # we center the 2 layers by default
            doubleoffset = [(lshape[i]-stride[i-1]*(rshape[i]-1)-1)
                            for i in range(1, len(lshape))]
        elif isinstance(doubleoffset, int):
            doubleoffset = [doubleoffset] * dim
        elif len(doubleoffset) != dim:
            raise ValueError(('Double offset should be of dimension {0}, '
                             'found {1}').format(dim, len(doubleoffset)))
        self.doubleoffset = tuple(doubleoffset)
        # parse and check dilation
        if isinstance(dilation, int):
            dilation = [dilation] * dim
        elif len(dilation) != dim:
            raise ValueError('Dilation should be of dimension {0}, found {1}'
                             .format(dim, len(dilation)))
        self.dilation = tuple(dilation)
        # Check for dangling nodes at the border and half shifts
        for i in range(dim):
            if (dilation[i]*(1-kernel_shape[i])+doubleoffset[i]) % 2 != 0:
                raise ValueError('Non-integer correspondence in dimension '
                                 '{0}'.format(i))
            double_bound = dilation[i] * (kernel_shape[i] - 1)
            lower_offset = doubleoffset[i]
            upper_offset = lower_offset + 2 * (stride[i] * (rshape[i+1]-1)
                                               + 1 - lshape[i+1])
            if lower_offset < - double_bound or \
                    lower_offset > double_bound or \
                    upper_offset < - double_bound or \
                    upper_offset > double_bound:
                raise ValueError('Dangling nodes in dimension {0}'.format(i))


class VariableLayer(object):
    """variable layer class for the LayerModel"""
    def __init__(self, name: str, layer_type: LayerType,
                 size: Union[int, Sequence[int]], nb_label: int
                 ) -> None:
        self.name = name
        self.layer_type = layer_type
        self.nb_label = nb_label
        self.params = LayerParams(size, nb_label, layer_type)
        self.size = self.params.size  # type: int
        self.shape = self.params.shape  # type: Tuple[int, ...]
        self.neighbors = []  # type: List[Tuple['VariableLayer', 'Connection']]

    @property
    def dim(self) -> int:
        return len(self.shape) - 1

    def to_repr(self) -> str:
        return 'Layer {0}: {1}, range {2}, size {3}'.format(
                self.name, self.layer_type, self.nb_label, self.size)


class Connection(object):
    """layer connection class for LayerModel"""
    def __init__(self,
                 name: str,
                 layers: Vlayers,  # should contain 2 different VariableLayers
                 method: ConnectionType, *args: Any, **kwargs: Any
                 ) -> None:
        if method not in ConnectionType:
            raise ValueError('Invalid method: {0}.'.format(method))
        self.name = str(name)
        self.method = method
        self.layers = tuple(layers)
        self.params = self._parse_args(args, kwargs)

    def to_repr(self) -> str:
        return "{0} connection {1}".format(self.method, self.name)

    def _parse_args(self,
                    args: Sequence[Any],
                    kwargs: Dict[str, Any]
                    ) -> ConnectionParams:
        llayer, rlayer = self.layers
        ret = ConnectionParams.parse(self.method, llayer.params, rlayer.params,
                                     *args, **kwargs)
        return ret


class LayerModel(object):
    def __init__(self, name: str) -> None:
        self.name = str(name)
        self.named_layers = dict()  # type: Dict[str, VariableLayer]
        self.conditioned_layers = []  # type: List[VariableLayer]
        self.observable_layers = []  # type: List[VariableLayer]
        self.hidden_layers = []  # type: List[VariableLayer]
        self.named_connections = dict()  # type: Dict[str, Connection]

    @property
    def layers(self):
        return self.conditioned_layers + self.observable_layers + \
               self.hidden_layers

    def add_layer(self, size: Union[int, Sequence[int]], nb_label: int,
                  layer_type: LayerType, name: str) -> None:
        if name in self.named_layers:
            raise ValueError('layer with name {0} already exists'.format(name))
        layer = VariableLayer(name, layer_type, size, nb_label)
        self.named_layers[name] = layer
        if layer_type is LayerType.OBS:
            self.observable_layers.append(layer)
        elif layer_type is LayerType.CND:
            self.conditioned_layers.append(layer)
        elif layer_type is LayerType.HID:
            self.hidden_layers.append(layer)
        else:
            raise ValueError('Invalid layer type: {}'.format(layer_type))

    def connect_layers(self, layers: Sequence[str], method: ConnectionType,
                       name: str, *args: Any, **kwargs: Any) -> None:
        if name in self.named_connections:
            raise ValueError('connection with name {0} already exists.'
                             .format(name))
        l1n, l2n = layers
        if l1n == l2n:
            raise ValueError('Self loop detected.')
        if {l1n, l2n}.issubset(set(self.conditioned_layers)):
            raise ValueError('Trying to create connection in the conditioned '
                             'part.')
        l1, l2 = self.named_layers[l1n], self.named_layers[l2n]
        connection = Connection(name, [l1, l2], method, *args, **kwargs)
        l1.neighbors.append((l2, connection))
        l2.neighbors.append((l1, connection))
        self.named_connections[name] = connection

    def show_summary(self) -> None:
        print('LayerModel {0} with {1} layer(s):'.format(
                self.name, len(self.layers)))
        for l in self.layers:
            print('  {0} with {1} neighbor(s):'.format(l.to_repr(),
                                                       len(l.neighbors)))
            for nl, nc in l.neighbors:
                print('    {0} with {1}'.format(nl.to_repr(), nc.to_repr()))


def check_model_connectivity(model: LayerModel
                             ) -> Tuple[bool, bool]:
        """check that all layers are in the model and are connected"""
        # check that no neighbor is out of the model
        closed_neighbor = True
        layers_in_model = set(model.layers)
        for layer in model.layers:
            neighbors = set([neighbor[0] for neighbor in layer.neighbors])
            if not neighbors.issubset(layers_in_model):
                closed_neighbor = False
                break
        # check that all layers are connected to each other
        q = Queue()  # type: Queue[VariableLayer]
        q.put(model.layers[0])
        while not q.empty():
            layer = q.get()
            if not hasattr(layer, 'check'):
                layer.check = True
                for neighbor in layer.neighbors:
                    nlayer = neighbor[0]
                    if nlayer in layers_in_model:
                        q.put(nlayer)
        all_connected = True
        for layer in model.layers:
            if hasattr(layer, 'check'):
                del layer.check
            else:
                all_connected = False
        return closed_neighbor, all_connected
