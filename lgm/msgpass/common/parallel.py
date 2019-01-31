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
common parts for parallel message passing algorithm
"""
import abc
from . import common


class LayerModelWrapper(common.LayerModelWrapper, metaclass=abc.ABCMeta):
    def _infer_step(self) -> None:
        msg_lists = []
        for l in self.layers.children():
            msg_lists.append([n.get_message_update()
                              for n in l.neighbors.children()])
        for l, msgs in zip(self.layers.children(), msg_lists):
            for n, m in zip(l.neighbors.children(), msgs):
                n.set_message_update(m)
        for l in self.layers.children():
            l.update_belief()
