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
high-level wrapper for message passing algorithms
"""
from typing import Dict, Union, Hashable, Iterable, Tuple
from .. import proto, model as impl
from ..utils.common import Picklable
import importlib
from .common import common as cc


INPUT_MODES_MAP = {'step': 'LayerModelWrapper'}
METHODS = {'pro': ('loopy', 'trw', 'seqtrw')}


def layerModelWrapper(model: Union[proto.LayerModel, impl.LayerModel],
                      input_mode: str, infer_mode: str, method: str,
                      frequency: int, output_names: Iterable[str] = (),
                      autoclear: bool = True, *args, **kwargs
                      ) -> cc.LayerModelWrapper:
    if input_mode not in INPUT_MODES_MAP:
        raise ValueError('Unknown input mode {}, should be in {}'.format(
                         input_mode, tuple(INPUT_MODES_MAP.keys())))
    if infer_mode not in METHODS:
        raise ValueError('Unknown inference mode {}, should be in {}'.format(
                         infer_mode, tuple(METHODS.keys())))
    if method not in METHODS[infer_mode]:
        raise ValueError('Unknown method {}, should be in {}'.format(method,
                         METHODS[infer_mode]))
    if isinstance(model, proto.LayerModel):
        model = impl.LayerModel(model)
    return getattr(importlib.import_module('.'.join(('', infer_mode+'infer',
                                                     method)),
                                           'lgm.msgpass'),
                   INPUT_MODES_MAP[input_mode])(model, frequency, output_names,
                                                autoclear, *args, **kwargs)


def loadWrapper(save_path: str
                ) -> Tuple[cc.LayerModelWrapper, Dict[Hashable, Picklable]]:
    ngmodel, dic = impl.LayerModel.load_model(save_path)
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
        model.load_state_dict(state_dict, False)
    del dic['frequency']
    del dic['output_names']
    del dic['autoclear']
    del dic['batchsize']
    del dic['wrapper_state_dict']
    return model, dic
