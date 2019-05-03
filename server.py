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
import timeit
import logging

import numpy as np
import grpc
from grpc._cython import cygrpc

import prometheus_client
from py_grpc_prometheus import prometheus_server_interceptor
# from python_grpc_prometheus import prometheus_server_interceptor

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

        t = timeit.default_timer()
        data = protobuf_request_to_dict(request)
        image = data['image']
        _logger.info('Loaded data into numpy array of shape %s in %s seconds.',
                     image.shape, timeit.default_timer() - t)

        t = timeit.default_timer()
        processed_image = F(image)
        _logger.info('%s processed data into shape %s in %s seconds.',
                     str(F.__name__).capitalize(), processed_image.shape,
                     timeit.default_timer() - t)

        t = timeit.default_timer()
        response = process_pb2.ProcessResponse()
        tensor_proto = make_tensor_proto(processed_image, 'DT_INT32')
        response.outputs['results'].CopyFrom(tensor_proto)  # pylint: disable=E1101
        _logger.info('Prepared response object in %s seconds.',
                     timeit.default_timer() - t)
        return response

    def StreamProcess(self, request_iterator, context):
        """Enable client to stream large payload for processing"""
        _logger = logging.getLogger('ProcessingServicer.StreamProcess')

        # intialize values.  should be same in each request.
        F = None
        shape = None  # need the shape as frombytes will laoad the data as 1D
        dtype = None  # need the dtype in case it is not `float`
        arrbytes = []

        t = timeit.default_timer()
        # get all the bytes from every request
        for request in request_iterator:
            shape = tuple(request.shape)
            dtype = str(request.dtype)
            F = get_function(request.function_spec.type,
                             request.function_spec.name)
            data = request.inputs['data']
            arrbytes.append(data)

        npbytes = b''.join(arrbytes)
        _logger.info('Got client request stream of %s bytes', len(npbytes))

        t = timeit.default_timer()
        image = np.frombuffer(npbytes, dtype=dtype).reshape(shape)
        _logger.info('Loaded data into numpy array of shape %s in %s seconds.',
                     image.shape, timeit.default_timer() - t)

        t = timeit.default_timer()
        processed_image = F(image)
        processed_shape = processed_image.shape  # to reshape client-side
        _logger.info('%s processed %s data into shape %s in %s seconds.',
                     str(F.__name__).capitalize(), processed_image.dtype,
                     processed_shape, timeit.default_timer() - t)

        # Send the numpy array back in responses of `chunk_size` bytes
        t = timeit.default_timer()
        chunk_size = 64 * 1024  # 64 kB is recommended payload size
        bytearr = processed_image.tobytes()  # the bytes to stream back
        _logger.info('Streaming %s bytes in %s responses',
                     len(bytearr), chunk_size % len(bytearr))
        for i in range(0, len(bytearr), chunk_size):
            response = process_pb2.ChunkedProcessResponse()
            # pylint: disable=E1101
            response.shape[:] = processed_shape
            response.outputs['data'] = bytearr[i: i + chunk_size]
            response.dtype = str(processed_image.dtype)
            # pylint: enable=E1101
            yield response

        _logger.info('Streamed %s bytes in %s seconds.',
                     len(bytearr), timeit.default_timer() - t)


if __name__ == '__main__':
    initialize_logger()
    LOGGER = logging.getLogger(__name__)
    LISTEN_PORT = os.getenv('LISTEN_PORT', '8080')
    WORKERS = int(os.getenv('WORKERS', '10'))
    PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', '8000'))
    PROMETHEUS_ENABLED = os.getenv('PROMETHEUS_ENABLED', 'true')
    PROMETHEUS_ENABLED = PROMETHEUS_ENABLED.lower() == 'true'

    # Add the required interceptor(s) where you create your grpc server, e.g.
    PSI = prometheus_server_interceptor.PromServerInterceptor()

    INTERCEPTORS = (PSI,) if PROMETHEUS_ENABLED else ()

    # define custom server options
    OPTIONS = [(cygrpc.ChannelArgKey.max_send_message_length, -1),
               (cygrpc.ChannelArgKey.max_receive_message_length, -1)]

    # create a gRPC server with custom options
    SERVER = grpc.server(futures.ThreadPoolExecutor(max_workers=WORKERS),
                         interceptors=INTERCEPTORS,
                         options=OPTIONS)

    # use the generated function `add_ProcessingServicer_to_server`
    # to add the defined class to the server
    processing_service_pb2_grpc.add_ProcessingServiceServicer_to_server(
        ProcessingServicer(), SERVER)

    # start the http server where prometheus can fetch the data from.
    if PROMETHEUS_ENABLED:
        LOGGER.info('Starting prometheus server. Listening on port %s',
                    PROMETHEUS_PORT)
        prometheus_client.start_http_server(PROMETHEUS_PORT)

    LOGGER.info('Starting server. Listening on port %s', LISTEN_PORT)
    SERVER.add_insecure_port('[::]:{}'.format(LISTEN_PORT))
    SERVER.start()

    # since SERVER.start() will not block,
    # a sleep-loop is added to keep alive
    try:
        while True:
            time.sleep(86400)  # 24 hours
    except KeyboardInterrupt:
        SERVER.stop(0)
