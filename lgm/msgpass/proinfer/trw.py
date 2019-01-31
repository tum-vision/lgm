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
parallel TRW method for probabilistic inference on layered graphical models
"""
from . import common
from ..common import common as cc, parallel as p, trw as t


class DenseNeighborWrapper(t.DenseNeighborWrapper,
                           common.DenseNeighborWrapper):
    pass


class LocalNeighborWrapper(t.LocalNeighborWrapper,
                           common.LocalNeighborWrapper):
    pass


def neighborWrapperBuilder(source: cc.LayerWrapper, target: cc.LayerWrapper,
                           connection: cc.ConnectionWrapper
                           ) -> cc.NeighborWrapper:
    return globals()[cc.NEIGHBOR_TYPE_MAP[connection.method]](
                                                source, target, connection)


class LayerModelWrapper(p.LayerModelWrapper, common.LayerModelWrapper):
    def wrap_neighbors(self) -> None:
        cc.wrapNeighbors(self.layers, self.connections,
                         self.layers2connection, neighborWrapperBuilder)
