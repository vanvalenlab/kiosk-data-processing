# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/kiosk-data-processing/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for post-processing functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from data_processing import postprocessing


class TestPostProcessing(object):

    def test_mibi(self):
        channels = 3
        img = np.random.rand(300, 300, channels)
        mibi_img = postprocessing.mibi(img)
        np.testing.assert_equal(mibi_img.shape, (300, 300, 1))

    def test_deepcell(self):
        channels = 4
        img = np.random.rand(300, 300, channels)
        deepcell_img = postprocessing.deepcell(img)
        np.testing.assert_equal(deepcell_img.shape, (300, 300, 1))

    def test_watershed(self):
        channels = np.random.randint(4, 8)
        img = np.random.rand(300, 300, channels)
        watershed_img = postprocessing.watershed(img)
        np.testing.assert_equal(watershed_img.shape, (300, 300, 1))