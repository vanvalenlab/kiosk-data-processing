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
"""Utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import six
import numpy as np
import dict_to_protobuf

from data_processing.pbs.tensor_pb2 import TensorProto
from data_processing.pbs.types_pb2 import DESCRIPTOR
from data_processing.settings import PROCESSING_FUNCTIONS


logger = logging.getLogger('data_processing.utils')


dtype_to_number = {
    i.name: i.number for i in DESCRIPTOR.enum_types_by_name['DataType'].values
}

# TODO: build this dynamically
number_to_dtype_value = {
    1: 'float_val',
    2: 'double_val',
    3: 'int_val',
    4: 'int_val',
    5: 'int_val',
    6: 'int_val',
    7: 'string_val',
    8: 'scomplex_val',
    9: 'int64_val',
    10: 'bool_val',
    18: 'dcomplex_val',
    19: 'half_val',
    20: 'resource_handle_val'
}


def protobuf_request_to_dict(pb):
    # TODO: 'unicode' object has no attribute 'ListFields'
    # response_dict = protobuf_to_dict(grpc_response)
    # return response_dict
    grpc_dict = dict()
    for k in pb.inputs:
        shape = [x.size for x in pb.inputs[k].tensor_shape.dim]
        logger.debug('Key: %s, shape: %s', k, shape)
        dtype_constant = pb.inputs[k].dtype
        if dtype_constant not in number_to_dtype_value:
            grpc_dict[k] = 'value not found'
            logger.error('Tensor output data type not supported. '
                         'Returning empty dict.')
        dt = number_to_dtype_value[dtype_constant]
        if shape == [1]:
            grpc_dict[k] = eval('pb.inputs[k].' + dt)[0]
        else:
            grpc_dict[k] = np.array(eval('pb.inputs[k].' + dt)).reshape(shape)
    return grpc_dict


def make_tensor_proto(data, dtype):
    tensor_proto = TensorProto()

    if isinstance(dtype, six.string_types):
        dtype = dtype_to_number[dtype]

    dim = [{'size': 1}]
    values = [data]

    if hasattr(data, 'shape'):
        dim = [{'size': dim} for dim in data.shape]
        values = list(data.reshape(-1))

    tensor_proto_dict = {
        'dtype': dtype,
        'tensor_shape': {
            'dim': dim
        },
        number_to_dtype_value[dtype]: values
    }

    dict_to_protobuf.dict_to_protobuf(tensor_proto_dict, tensor_proto)

    return tensor_proto


def get_function(process_type, function_name):
    """Based on the function category and name, return the function"""
    clean = lambda x: str(x).lower()
    # first, verify the route parameters
    name = clean(function_name)
    cat = clean(process_type)
    return PROCESSING_FUNCTIONS[cat][name]
