import grpc
from concurrent import futures
import time

# import the generated classes
from data_processing.pbs import process_pb2
from data_processing.pbs import processing_service_pb2
from data_processing.pbs import processing_service_pb2_grpc

# import the processing functions
from data_processing.utils import get_function


PORT = 8080

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


# create a gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

# use the generated function `add_ProcessingServicer_to_server`
# to add the defined class to the server
processing_service_pb2_grpc.add_ProcessingServiceServicer_to_server(
        ProcessingServicer(), server)

# listen on port 50051
print('Starting server. Listening on port', PORT)
server.add_insecure_port('[::]:{}'.format(PORT))
server.start()

# since server.start() will not block,
# a sleep-loop is added to keep alive
try:
    while True:
        time.sleep(86400)  # 24 hours
except KeyboardInterrupt:
    server.stop(0)
