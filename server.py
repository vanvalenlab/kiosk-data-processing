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

import os
import time

import grpc
from concurrent import futures

from data_processing.pbs import process_pb2
from data_processing.pbs import processing_service_pb2
from data_processing.pbs import processing_service_pb2_grpc

from data_processing.utils import get_function


# create a class to define the server functions, derived from
# calculator_pb2_grpc.CalculatorServicer
class ProcessingServicer(processing_service_pb2_grpc.ProcessingServiceServicer):

    # expose Process() and all the `data_processing` functions
    def Process(self, request, context):
        F = get_function(request.function_spec.type,
                         request.function_spec.name)

        response = process_pb2.ProcessResponse()
        response.value = F(request.inputs)
        return response


if __name__ == '__main__':
    LISTEN_PORT = os.getenv('LISTEN_PORT', 8080)

    # create a gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # use the generated function `add_ProcessingServicer_to_server`
    # to add the defined class to the server
    processing_service_pb2_grpc.add_ProcessingServiceServicer_to_server(
            ProcessingServicer(), server)

    print('Starting server. Listening on port', LISTEN_PORT)
    server.add_insecure_port('[::]:{}'.format(LISTEN_PORT))
    server.start()

    # since server.start() will not block,
    # a sleep-loop is added to keep alive
    try:
        while True:
            time.sleep(86400)  # 24 hours
    except KeyboardInterrupt:
        server.stop(0)
