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
common message passing wrapper template for layered graphical model
"""
import abc
from warnings import warn
from typing import (List, Hashable, Dict, Tuple, Optional, Iterable, Callable,
                    Union, Sequence, Any)
import torch
from torch import Tensor
import torch.nn as nn
from ...proto import LayerType, ConnectionType
from ...model import VariableLayer, Connection, Neighbor, LayerModel
from ...cpp.custom_ops import softmax0fo, logep, logsoftmax0fo
from ...utils.common import (Picklable, SavableModel, full_class_name,
                             AlwaysGet, flip_local, unzip_one)


NEIGHBOR_TYPE_MAP = {ConnectionType.DENSE: 'DenseNeighborWrapper',
                     ConnectionType.LOCAL: 'LocalNeighborWrapper'}
CONNECTION_TYPE_MAP = {ConnectionType.DENSE: 'DenseConnectionWrapper',
                       ConnectionType.LOCAL: 'LocalConnectionWrapper'}


class LayerWrapper(nn.Module):
    def __init__(self, layer: VariableLayer) -> None:
        super(LayerWrapper, self).__init__()
        # break nn.Module reference with wrapped content
        self._layer = (layer,)
        self._unary = (layer.unary,)
        self.name = layer.name
        self.size = layer.size
        self.nb_label = layer.nb_label
        self.layer_type = layer.layer_type
        self.belief = None  # compact value, one-hot when given, else in log
        self.clamped = False
        self.neighbors = nn.Module()

    @property
    def layer(self) -> VariableLayer:
        return self._layer[0]

    @property
    def unary(self) -> nn.Parameter:
        return self._unary[0]

    def get_energy(self) -> Tensor:
        return self.layer.get_energy()

    def get_neighbor(self, name: str) -> 'NeighborWrapper':
        return getattr(self.neighbors, name)

    def clear_belief(self) -> None:
        self.belief = None
        self.clamped = False

    def cleared_belief(self) -> bool:
        return self.belief is None

    def set_belief(self, extbel: Tensor,
                   clamped: Optional[Tensor]=None) -> None:
        if extbel is None:  # No observation: do nothing
            return
        self.belief = extbel.view(extbel.size(0), self.size, self.nb_label-1)
        if clamped is None:
            self.clamped = True
        elif clamped is True or clamped is False:
            self.clamped = clamped
        else:
            self.clamped = clamped[(..., *((None,) * (
                                        self.belief.dim() - clamped.dim())))]

    def get_belief_update(self) -> Optional[Tensor]:
        if self.clamped is True:
            return
        nec = self.unary.unsqueeze(0)
        for nl in self.neighbors.children():
            nluc = nl.get_inverse_neighbor().get_upd_contrib()
            nec = nec + nluc
        return nec

    def set_belief_update(self, upd: Optional[Tensor]) -> None:
        if self.clamped is True or upd is None:
            return
        if self.clamped is False:
            self.belief = upd
        else:
            self.belief = torch.where(self.clamped, self.belief, upd)

    def update_belief(self) -> None:
        self.set_belief_update(self.get_belief_update())

    def get_belief(self) -> Tensor:
        if self.clamped is True:
            return unzip_one(self.belief)
        beliefexp = softmax0fo(self.belief, -1)
        if self.clamped is False:
            return beliefexp
        else:
            return torch.where(self.clamped, unzip_one(self.belief), beliefexp)

    def get_logbelief(self) -> Tensor:
        if self.clamped is True:
            return logep(unzip_one(self.belief))
        beliefexp = logsoftmax0fo(self.belief, -1)
        if self.clamped is False:
            return beliefexp
        else:
            return torch.where(self.clamped,
                               logep(unzip_one(self.belief)), beliefexp)


class ConnectionWrapper(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, connection: Connection, left: LayerWrapper,
                 right: LayerWrapper) -> None:
        super(ConnectionWrapper, self).__init__()
        self._detached_modules = {'left': left, 'right': right,
                                  'connection': connection,
                                  'binary': connection.binary}
        self.name = connection.name
        self.method = connection.method
        self.params = connection.params
        self.get_energy = connection.get_energy

    @property
    def left(self) -> LayerWrapper:
        return self._detached_modules['left']

    @property
    def right(self) -> LayerWrapper:
        return self._detached_modules['right']

    @property
    def binary(self) -> Tensor:
        return self._detached_modules['binary']

    @property
    def connection(self) -> Connection:
        return self._detached_modules['connection']


class NeighborWrapper(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, source: LayerWrapper, target: LayerWrapper,
                 connection: ConnectionWrapper) -> None:
        super(NeighborWrapper, self).__init__()
        self._batchsize = 0
        neighbor = source.layer.get_neighbor(target.name)
        # break nn.Module reference with wrapped content
        self._detached_modules = {'neighbor': neighbor, 'source': source,
                                  'target': target, 'connection': connection}
        self.params = self.connection.params
        self.register_buffer('message', None)
        self.message = Tensor([])
        self.direct = neighbor.direct
        self.get_energy = neighbor.get_energy

    @property
    def batchsize(self) -> int:
        return self._batchsize

    @property
    def neighbor(self) -> Neighbor:
        return self._detached_modules['neighbor']

    @property
    def source(self) -> LayerWrapper:
        return self._detached_modules['source']

    @property
    def target(self) -> LayerWrapper:
        return self._detached_modules['target']

    @property
    def connection(self) -> ConnectionWrapper:
        return self._detached_modules['connection']

    def init_message(self, batchsize: int) -> None:
        if self.target.clamped is True:
            return
        self._batchsize = batchsize
        self.message = self._init_message(batchsize)

    def _init_message(self, batchsize: int) -> Tensor:
        return self.message.new_full(self.get_message_size(batchsize), 0)

    @staticmethod
    def _msg_clamp(sbelief: Tensor, neg_bin: Tensor) -> Tensor:
        retu = torch.sum(sbelief * neg_bin, dim=-1)
        return retu

    def get_message_update(self) -> Optional[Tensor]:
        if self.target.clamped is True:
            return
        inv_msg = self.get_inverse_neighbor().message
        sbelief = self.source.belief
        neg_bin = self.get_inverse_neighbor().get_energy().unsqueeze(0)
        mask = self.source.clamped
        if not isinstance(mask, bool):
            mask = mask.squeeze(-1)
        sbelief, mask, neg_bin = self._inputs_reshape(sbelief, mask, neg_bin)
        if mask is True:
            msg = self._msg_clamp(sbelief, neg_bin)
        else:
            inv_msg = inv_msg.unsqueeze(-2)
            if mask is False:
                msg = self._msg_upd(inv_msg, sbelief, neg_bin)
            else:
                msg = torch.where(mask, self._msg_clamp(sbelief, neg_bin),
                                  self._msg_upd(inv_msg, sbelief, neg_bin))
        return self._flip_msg(msg)

    def set_message_update(self, upd: Optional[Tensor]) -> None:
        if self.target.clamped is True:
            return
        self.message = upd

    def clear_message(self) -> None:
        self.message = torch.tensor([], device=self.message.device)
        self._batchsize = 0

    def cleared_message(self) -> bool:
        return self._batchsize == 0

    def rewind_message(self, dropmask: Optional[Tensor]=None) -> None:
        self.message = self.message.detach()
        if dropmask is None:  # no need to drop
            return
        sz = dropmask.size(0)
        if sz != self._batchsize:
            raise RuntimeError(('drop mask has size {} but current batchsize '
                                'is {}').format(sz, self._batchsize))
        dropmask = dropmask[(..., *((None,) *
                                    (self.message.dim() - dropmask.dim())))]
        self.message = torch.where(
                                dropmask, self._init_message(sz), self.message)

    def get_inverse_neighbor(self) -> 'NeighborWrapper':
        return getattr(self.target.neighbors, self.source.name)

    @abc.abstractmethod
    def get_message_size(self, batchsize: int) -> Tuple[int, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def _flip_msg(self, msg: Tensor) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _inputs_reshape(self, sbelief: Tensor, mask, neg_bin: Tensor,
                        ) -> Tuple[Tensor, Any, Tensor]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _msg_upd(inv_msg: Tensor, sbelief: Tensor, neg_bin: Tensor) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_upd_contrib(self) -> Tensor:
        raise NotImplementedError


class DenseNeighborWrapper(NeighborWrapper, metaclass=abc.ABCMeta):
    """Implementation of neighbor with dense connection"""
    def get_message_size(self, batchsize: int) -> Tuple[int, ...]:
        return (batchsize, self.source.size, self.target.size,
                self.target.nb_label - 1)

    def _flip_msg(self, msg: Tensor) -> Tensor:
        return msg.transpose(1, 2)

    def _inputs_reshape(self, sbelief: Tensor, mask, neg_bin: Tensor,
                        ) -> Tuple[Tensor, Any, Tensor]:
        sbelief = sbelief.unsqueeze(2).unsqueeze(1)
        if not isinstance(mask, bool):
            mask = mask.unsqueeze(2).unsqueeze(1)
        return sbelief, mask, neg_bin

    def get_upd_contrib(self) -> Tensor:
        retu = torch.sum(self.message, dim=1)
        return retu


class LocalNeighborWrapper(NeighborWrapper, metaclass=abc.ABCMeta):
    def __init__(self, source: LayerWrapper, target: LayerWrapper,
                 connection: ConnectionWrapper) -> None:
        super().__init__(source, target, connection)
        self.links = self.connection.connection.links

    def get_message_size(self, batchsize: int) -> Tuple[int, ...]:
        params = self.connection.params
        if self.direct:
            return (batchsize, params.lchannel, params.rchannel,
                    *params.kshape, *params.rshape, self.target.nb_label - 1)
        else:
            return (batchsize, params.rchannel, params.lchannel,
                    *params.kshape, *params.lshape, self.target.nb_label - 1)

    def _flip_msg(self, msg: Tensor) -> Tensor:
        return flip_local(msg, self.message.size(), 0.,
                          *self.links(self.direct))

    def _inputs_reshape(self, sbelief: Tensor, mask, neg_bin: Tensor,
                        ) -> Tuple[Tensor, Any, Tensor]:
        params = self.params
        conv_dim = params.dim
        cnl = getattr(params, ('rchannel', 'lchannel')[self.direct])
        shp = getattr(params, ('rshape', 'lshape')[self.direct])
        bsz, nlb = sbelief.size(0), sbelief.size(-1)
        sbelief = sbelief.view(bsz, 1, cnl, *((1,) * conv_dim), *shp, 1, nlb)
        if not isinstance(mask, bool):
            mask = mask.view(bsz, 1, cnl, *((1,) * conv_dim), *shp, 1)
        return sbelief, mask, neg_bin

    def get_upd_contrib(self) -> Tensor:
        msg = self.message
        conv_dim = self.params.dim
        sz = msg.size()
        return torch.sum(msg.view(*sz[0:3], -1, *sz[3 + conv_dim:]), dim=(1, 3)
                         ).view(sz[0], -1, sz[-1])


class LayerModelWrapper(nn.Module, SavableModel, metaclass=abc.ABCMeta):
    def __init__(self, model: LayerModel, frequency: int,
                 output_names: Iterable[str]=(), autoclear: bool=True
                 ) -> None:
        super(LayerModelWrapper, self).__init__()
        # break nn.Module reference with wrapped content
        self._batchsize = 0
        self._frequency = 1
        self.model = model
        self.frequency = frequency
        self.output_names = output_names
        self.autoclear = autoclear
        self._check_output_names()
        self.name = model.name
        self.nb_conditioned = model.nb_conditioned
        self.nb_observable = model.nb_observable
        self.nb_hidden = model.nb_hidden
        self.layers = self.wrap_layers()
        self.connections = self.wrap_connections()
        self.layers2connection = model.layers2connection
        self.wrap_neighbors()

    @property
    def batchsize(self) -> int:
        return self._batchsize

    @property
    def frequency(self) -> int:
        return self._frequency

    @frequency.setter
    def frequency(self, val: int) -> None:
        if not (isinstance(val, int)) or val <= 0:
            raise ValueError('Invalid frequency: should be strictly positive '
                             'integer, received {}.'.format(val))
        self._frequency = val

    def get_layer(self, name: str) -> LayerWrapper:
        return getattr(self.layers, name)

    def get_connection(self, name: str) -> ConnectionWrapper:
        return getattr(self.connections, name)

    def get_connection_name_between(self, *layernames: str) -> str:
        return self.model.layers2connection[tuple(sorted(layernames))]

    @staticmethod
    def _infer_batchsize(inputs: Sequence[Tensor]) -> int:
        bs = set([i.size(0) for i in inputs])
        if len(bs) != 1:
            raise ValueError('non-unique batchsizes: {0} values found {1}\
                             '.format(len(bs), bs))
        return bs.pop()

    def _check_output_names(self) -> None:
        layer_names = [n for n, _1 in self.model.layers.named_children()]
        for n in self.output_names:
            if n not in layer_names:
                raise RuntimeError('Unresolved output name: {}'.format(n))

    def _check_evidences(self) -> None:
        for l in self.layers.children():
            if l.layer_type is LayerType.CND and l.clamped is not True:
                raise RuntimeError(('{} is conditioned and requires full '
                                    'observation.').format(l.name))
            elif l.layer_type == LayerType.HID and l.clamped is not False:
                raise RuntimeError(('{} is hidden and requires no observation.'
                                    '').format(l.name))

    def _load_known_believes(self, inputs: Sequence[Optional[Tensor]],
                             masks: Optional[
                                 Sequence[Optional[Tensor]]] = None
                             ) -> None:
        layers = iter(self.layers.children())
        if masks is None:
            masks = AlwaysGet(None)
        for i in range(self.nb_conditioned + self.nb_observable):
            next(layers).set_belief(inputs[i], masks[i])

    def _load_messages(self) -> None:
        for l in self.layers.children():
            for n in l.neighbors.children():
                if n.cleared_message():
                    n.init_message(self.batchsize)

    def _infer_init(self, inputs: Union[Tensor,
                                        Sequence[Optional[Tensor]]],
                    masks: Optional[Sequence[Optional[Tensor]]]=None
                    ) -> None:
        if torch.is_tensor(inputs):
            inputs = (inputs,)
        if len(inputs) != self.nb_conditioned + self.nb_observable:
            raise RuntimeError('need {} inputs, found {}'.format(
                self.nb_conditioned + self.nb_observable, len(inputs)))
        if masks is not None and len(inputs) != len(masks):
            raise RuntimeError('need {} masks, found {}'.format(
                len(inputs), len(masks)))
        # check for batch size when needed
        if self._batchsize == 0:
            self._batchsize = self._infer_batchsize(inputs)
        # load all message values
        self._load_messages()
        # load all belief values
        self._load_known_believes(inputs, masks)
        self._check_evidences()
        for l in self.layers.children():
            l.update_belief()

    def forward(self, inputs: Union[Tensor,
                                    Sequence[Optional[Tensor]]],
                masks: Optional[Sequence[Optional[Tensor]]] = None
                ) -> List[Tensor]:
        # initialise system for inference
        self._infer_init(inputs, masks)
        # repeat message passing update [self.frequency] times
        for f in range(self.frequency):
            self._infer_step()
        # grab outputs
        return_val = []
        layers = self.layers
        for n in self.output_names:
            return_val.append(getattr(layers, n).get_logbelief())
        # clear messages at the end if not needed, else save messages
        if self.autoclear:
            self.clear()
        else:
            self.rewind()
        return return_val

    def rewind(self, dropmask: Optional[Tensor]=None) -> None:
        for lyr in self.layers.children():
            for n in lyr.neighbors.children():
                n.rewind_message(dropmask)

    def clear(self) -> None:
        """clear stored messages and believes"""
        for l in self.layers.children():
            l.clear_belief()
            for n in l.neighbors.children():
                n.clear_message()
        self._batchsize = 0

    def cleared(self) -> bool:
        """check if stored messages and believes are cleared"""
        return self.batchsize == 0

    def show_summary(self) -> None:
        self.model.show_summary()

    def save_model(self, path: str,
                   addons: Optional[Dict[Hashable, Picklable]] = None) -> None:
        if addons is None:
            addons = {}
        dic = {'cls': type(self),
               'frequency': self.frequency,
               'output_names': self.output_names,
               'autoclear': self.autoclear,
               'batchsize': self.batchsize,
               'state_dict': {},  # redundant for ng.LayerModel
               'wrapper_state_dict': self.state_dict(),
               **addons}
        self.model.save_model(path, dic)

    @classmethod
    def load_model(cls, path: str
                   ) -> ('LayerModelWrapper', Dict[Hashable, Picklable]):
        ngmodel, dic = LayerModel.load_model(path)
        if dic['cls'] != cls:
            warn(('expect instance of type {0},\nreceived {1}.\nattempting to '
                  'load as the given class ...').format(
                full_class_name(cls), full_class_name(dic['cls'])))
        model = dic['cls'](ngmodel, frequency=dic['frequency'],
                           output_names=dic['output_names'],
                           autoclear=dic['autoclear'])
        bsz = dic['batchsize']
        model._batchsize = bsz
        if bsz != 0:
            for l in model.layers.children():
                for n in l.neighbors.values():
                    n.init_message(bsz)
        state_dict = dic['wrapper_state_dict']
        if state_dict:
            model.load_state_dict(state_dict)
        del dic['frequency']
        del dic['autoclear']
        del dic['output_names']
        del dic['batchsize']
        del dic['wrapper_state_dict']
        return model, dic

    def wrap_layers(self) -> nn.Module:
        layerwrappers = nn.Module()
        for n, m in self.model.layers.named_children():
            layerwrappers.add_module(n, LayerWrapper(m))
        return layerwrappers

    def wrap_connections(self) -> nn.Module:
        cnntwrappers = nn.Module()
        layerwrappers = self.layers
        for cname, cnnt in self.model.connections.named_children():
            cnntwrappers.add_module(
                cname, ConnectionWrapper(
                    cnnt, getattr(layerwrappers, cnnt.left.name),
                    getattr(layerwrappers, cnnt.right.name)))
        return cnntwrappers

    @abc.abstractmethod
    def wrap_neighbors(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _infer_step(self) -> None:
        raise NotImplementedError


def wrapNeighbors(layerwrappers: nn.Module,
                  cnntwrappers: nn.Module,
                  lyrs2cnnt: Dict[Tuple[str, str], str],
                  neighborwrap_builder: Callable[
                      [LayerWrapper, LayerWrapper, ConnectionWrapper],
                      NeighborWrapper]
                  ) -> None:
    for sname, lw in layerwrappers.named_children():
        for nname, _ in lw.layer.neighbors.named_children():
            n = neighborwrap_builder(
                getattr(layerwrappers, sname), getattr(layerwrappers, nname),
                getattr(cnntwrappers, lyrs2cnnt[tuple(sorted((sname, nname)))])
            )
            lw.neighbors.add_module(nname, n)
