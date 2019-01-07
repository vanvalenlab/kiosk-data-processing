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
"""Tests for utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

from data_processing.pbs import types_pb2
from data_processing.pbs.tensor_pb2 import TensorProto
from data_processing.pbs.process_pb2 import ProcessRequest
from data_processing.pbs.process_pb2 import ProcessResponse
from data_processing import utils


def _get_image(img_h=300, img_w=300, channels=1):
    bias = np.random.rand(img_w, img_h, channels) * 64
    variance = np.random.rand(img_w, img_h, channels) * (255 - 64)
    img = np.random.rand(img_w, img_h, channels) * variance + bias
    return img


class TestUtils(object):

    def test_make_tensor_proto(self):
        # test with numpy array
        data = _get_image(300, 300, 1)
        proto = utils.make_tensor_proto(data, 'DT_FLOAT')
        assert isinstance(proto, (TensorProto,))
        # test with value
        data = 10.0
        proto = utils.make_tensor_proto(data, types_pb2.DT_FLOAT)
        assert isinstance(proto, (TensorProto,))

    def test_protobuf_request_to_dict(self):
        # test valid request
        data = _get_image(300, 300, 1)
        tensor_proto = utils.make_tensor_proto(data, 'DT_FLOAT')
        request = ProcessRequest()
        request.inputs['prediction'].CopyFrom(tensor_proto)
        request_dict = utils.protobuf_request_to_dict(request)
        assert isinstance(request_dict, (dict,))
        np.testing.assert_allclose(request_dict['prediction'], data)
        # test scalar input
        data = 3
        tensor_proto = utils.make_tensor_proto(data, 'DT_FLOAT')
        request = ProcessRequest()
        request.inputs['prediction'].CopyFrom(tensor_proto)
        request_dict = utils.protobuf_request_to_dict(request)
        assert isinstance(request_dict, (dict,))
        np.testing.assert_allclose(request_dict['prediction'], data)
        # test bad dtype
        # logs an error, but should throw a KeyError as well.
        data = _get_image(300, 300, 1)
        tensor_proto = utils.make_tensor_proto(data, 'DT_FLOAT')
        request = ProcessRequest()
        request.inputs['prediction'].CopyFrom(tensor_proto)
        request.inputs['prediction'].dtype = 32
        with pytest.raises(KeyError):
            request_dict = utils.protobuf_request_to_dict(request)

    def test_get_function(self):
        big = utils.get_function('PRE', 'NORMALIZE')
        small = utils.get_function('pre', 'normalize')
        mixed = utils.get_function('pRe', 'nOrmAliZe')
        np.testing.assert_equal(big, small)
        np.testing.assert_equal(big, mixed)
        with pytest.raises(KeyError):
            _ = utils.get_function('bad', 'normalize')
        with pytest.raises(KeyError):
            _ = utils.get_function('post', 'bad')
