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
file containing LGM models
"""
import lgm.proto as proto
from lgm.proto import LayerType, ConnectionType


NB_CLASS = 10


def dense() -> proto.LayerModel:
    layer_graph = proto.LayerModel('dense')
    layer_graph.add_layer((1, 28, 28), 2, LayerType.CND, 'i')
    layer_graph.add_layer(100, 2, LayerType.HID, 'h')
    layer_graph.add_layer(1, NB_CLASS, LayerType.HID, 'o')
    layer_graph.connect_layers(['i', 'h'], ConnectionType.DENSE, 'ih')
    layer_graph.connect_layers(['h', 'o'], ConnectionType.DENSE, 'ho')
    return layer_graph


def local() -> proto.LayerModel:
    layer_graph = proto.LayerModel('local_mininet')
    layer_graph.add_layer((1, 28, 28), 2, LayerType.CND, 'i')
    layer_graph.add_layer((6, 13, 13), 2, LayerType.HID, 'l1')
    layer_graph.add_layer((16, 5, 5), 2, LayerType.HID, 'l2')
    layer_graph.add_layer(100, 2, LayerType.HID, 'd')
    layer_graph.add_layer(1, NB_CLASS, LayerType.HID, 'o')
    layer_graph.connect_layers(
            ['i', 'l1'], ConnectionType.LOCAL, 'l1', kernel_shape=(5, 5),
            stride=(2, 2), doubleoffset=(4, 4))
    layer_graph.connect_layers(
            ['l1', 'l2'], ConnectionType.LOCAL, 'l2', kernel_shape=(5, 5),
            stride=(2, 2), doubleoffset=(4, 4))
    layer_graph.connect_layers(['l2', 'd'], ConnectionType.DENSE, 'ld')
    layer_graph.connect_layers(['d', 'o'], ConnectionType.DENSE, 'do')
    return layer_graph
