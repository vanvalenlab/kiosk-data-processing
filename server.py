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
"""gRPC server to expose data processing functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent import futures

import os
import sys
import time
import logging

import numpy as np
import grpc
from grpc._cython import cygrpc

from data_processing.pbs import process_pb2
from data_processing.pbs import processing_service_pb2_grpc
from data_processing.utils import get_function
from data_processing.utils import protobuf_request_to_dict
from data_processing.utils import make_tensor_proto


def initialize_logger(debug_mode=False):
    """Sets up the logger"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s]:[%(name)s]: %(message)s')
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)

    if debug_mode:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)

    logger.addHandler(console)


class ProcessingServicer(processing_service_pb2_grpc.ProcessingServiceServicer):
    """Class to define the server functions"""

    def Process(self, request, context):
        """Expose Process() and all the `data_processing` functions"""
        _logger = logging.getLogger('ProcessingServicer.Process')
        F = get_function(request.function_spec.type,
                         request.function_spec.name)

        t = time.time()
        data = protobuf_request_to_dict(request)
        image = data['image']
        _logger.info('Loaded data into numpy array with shape %s in %s s',
                     image.shape, time.time() - t)

        t = time.time()
        processed_image = F(image)
        _logger.info('%s processed data into shape %s in %s s',
                     str(F.__name__).capitalize(), processed_image.shape,
                     time.time() - t)

        t = time.time()
        response = process_pb2.ProcessResponse()
        tensor_proto = make_tensor_proto(processed_image, 'DT_INT32')
        response.outputs['results'].CopyFrom(tensor_proto)  # pylint: disable=E1101
        _logger.info('Prepared response object in %s s', time.time() - t)
        return response

    def StreamProcess(self, request_iterator, context):
        """Enable client to stream large payload for processing"""
        _logger = logging.getLogger('ProcessingServicer.StreamProcess')

        # intialize values.  should be same in each request.
        F = None
        shape = None  # need the shape as frombytes will laoad the data as 1D
        dtype = None  # need the dtype in case it is not `float`
        arrbytes = []

        t = time.time()
        # get all the bytes from every request
        for request in request_iterator:
            shape = tuple(request.shape)
            dtype = str(request.dtype)
            F = get_function(request.function_spec.type,
                             request.function_spec.name)
            data = request.inputs['data']
            arrbytes.append(data)

        npbytes = b''.join(arrbytes)
        _logger.info('Got client stream of %s bytes', len(npbytes))

        t = time.time()
        image = np.frombuffer(npbytes, dtype=dtype).reshape(shape)
        _logger.info('Loaded data into numpy array with shape %s in %s s',
                     image.shape, time.time() - t)

        t = time.time()
        processed_image = F(image)

        processed_shape = processed_image.shape  # to reshape client-side
        _logger.info('%s processed %s data into shape %s in %s s',
                     str(F.__name__).capitalize(), processed_image.dtype,
                     processed_shape, time.time() - t)

        # Send the numpy array back in responses of `chunk_size` bytes
        chunk_size = 4 * 1024 * 1024  # 4 MB
        bytearr = processed_image.tobytes()  # the bytes to stream back
        _t = time.time()
        for i, j in enumerate(range(0, len(bytearr), chunk_size)):
            _logger.info('Streaming %s / %s bytes',
                         (i + 1) * chunk_size, len(bytearr))
            t = time.time()
            response = process_pb2.ChunkedProcessResponse()
            # pylint: disable=E1101
            response.shape[:] = processed_shape
            response.outputs['data'] = bytearr[j: j + chunk_size]
            response.dtype = str(processed_image.dtype)
            # pylint: enable=E1101
            _logger.debug('Creating response object took: %s', time.time() - t)
            yield response

        _logger.info('Finished all responses in: %s s', time.time() - _t)


if __name__ == '__main__':
    initialize_logger()
    _logger = logging.getLogger()
    LISTEN_PORT = os.getenv('LISTEN_PORT', 8080)

    # define custom server options
    options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
             (cygrpc.ChannelArgKey.max_receive_message_length, -1)]

    # create a gRPC server with custom options
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=options)

    # use the generated function `add_ProcessingServicer_to_server`
    # to add the defined class to the server
    processing_service_pb2_grpc.add_ProcessingServiceServicer_to_server(
        ProcessingServicer(), server)

    _logger.info('Starting server. Listening on port %s', LISTEN_PORT)
    server.add_insecure_port('[::]:{}'.format(LISTEN_PORT))
    server.start()

    # since server.start() will not block,
    # a sleep-loop is added to keep alive
    try:
        while True:
            time.sleep(86400)  # 24 hours
    except KeyboardInterrupt:
        server.stop(0)
