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
setup script for package installation
"""
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
# check Python version
import sys
v = sys.version_info
if v < (3,):
    raise RuntimeError('Python 2 is not supported.')
# check typing support
try:
    import typing
except ImportError:
    raise RuntimeError('"typing" module is not available. Please install the '
                       'backported package or use python >= 3.5.')
# attempt to import torch to check availability
try:
    import torch
except ImportError:
    raise RuntimeError('PyTorch is not installed properly. Please go to '
                       'pytorch.org and follow the installation guide.')
# check PyTorch version
v = torch.version.__version__
if not v >= '1.0.0':
    raise RuntimeError('Expect PyTorch version 1.0.0+, found {}.'.format(v))
else:
    print('Found PyTorch {}.'.format(v))
# install package
setup(
    name="lgm",
    version="0.1-a1",
    author='Yuesong Shen',
    author_email='yuesong.shen@tum.de',
    ext_modules=[
        CppExtension('lgm.cpp.custom_ops_cpp',
                     ['lgm/cpp/custom_ops.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages()
)
