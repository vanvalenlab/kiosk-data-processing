# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: process.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import data_processing.pbs.tensor_pb2 as tensor__pb2
import data_processing.pbs.function_pb2 as function__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='process.proto',
  package='tensorflow.serving',
  syntax='proto3',
  serialized_options=_b('\370\001\001'),
  serialized_pb=_b('\n\rprocess.proto\x12\x12tensorflow.serving\x1a\x0ctensor.proto\x1a\x0e\x66unction.proto\"\xe8\x01\n\x0eProcessRequest\x12\x37\n\rfunction_spec\x18\x01 \x01(\x0b\x32 .tensorflow.serving.FunctionSpec\x12>\n\x06inputs\x18\x02 \x03(\x0b\x32..tensorflow.serving.ProcessRequest.InputsEntry\x12\x15\n\routput_filter\x18\x03 \x03(\t\x1a\x46\n\x0bInputsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.tensorflow.TensorProto:\x02\x38\x01\"\x9d\x01\n\x0fProcessResponse\x12\x41\n\x07outputs\x18\x01 \x03(\x0b\x32\x30.tensorflow.serving.ProcessResponse.OutputsEntry\x1aG\n\x0cOutputsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.tensorflow.TensorProto:\x02\x38\x01\"\xfb\x01\n\x15\x43hunkedProcessRequest\x12\x37\n\rfunction_spec\x18\x01 \x01(\x0b\x32 .tensorflow.serving.FunctionSpec\x12\x45\n\x06inputs\x18\x02 \x03(\x0b\x32\x35.tensorflow.serving.ChunkedProcessRequest.InputsEntry\x12\x15\n\routput_filter\x18\x03 \x03(\t\x12\r\n\x05shape\x18\x04 \x03(\x03\x12\r\n\x05\x64type\x18\x05 \x01(\t\x1a-\n\x0bInputsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\"\xb0\x01\n\x16\x43hunkedProcessResponse\x12H\n\x07outputs\x18\x01 \x03(\x0b\x32\x37.tensorflow.serving.ChunkedProcessResponse.OutputsEntry\x12\r\n\x05shape\x18\x04 \x03(\x03\x12\r\n\x05\x64type\x18\x05 \x01(\t\x1a.\n\x0cOutputsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\x42\x03\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensor__pb2.DESCRIPTOR,function__pb2.DESCRIPTOR,])




_PROCESSREQUEST_INPUTSENTRY = _descriptor.Descriptor(
  name='InputsEntry',
  full_name='tensorflow.serving.ProcessRequest.InputsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.serving.ProcessRequest.InputsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.serving.ProcessRequest.InputsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=230,
  serialized_end=300,
)

_PROCESSREQUEST = _descriptor.Descriptor(
  name='ProcessRequest',
  full_name='tensorflow.serving.ProcessRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='function_spec', full_name='tensorflow.serving.ProcessRequest.function_spec', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inputs', full_name='tensorflow.serving.ProcessRequest.inputs', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_filter', full_name='tensorflow.serving.ProcessRequest.output_filter', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_PROCESSREQUEST_INPUTSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=68,
  serialized_end=300,
)


_PROCESSRESPONSE_OUTPUTSENTRY = _descriptor.Descriptor(
  name='OutputsEntry',
  full_name='tensorflow.serving.ProcessResponse.OutputsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.serving.ProcessResponse.OutputsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.serving.ProcessResponse.OutputsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=389,
  serialized_end=460,
)

_PROCESSRESPONSE = _descriptor.Descriptor(
  name='ProcessResponse',
  full_name='tensorflow.serving.ProcessResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='outputs', full_name='tensorflow.serving.ProcessResponse.outputs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_PROCESSRESPONSE_OUTPUTSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=303,
  serialized_end=460,
)


_CHUNKEDPROCESSREQUEST_INPUTSENTRY = _descriptor.Descriptor(
  name='InputsEntry',
  full_name='tensorflow.serving.ChunkedProcessRequest.InputsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.serving.ChunkedProcessRequest.InputsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.serving.ChunkedProcessRequest.InputsEntry.value', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=669,
  serialized_end=714,
)

_CHUNKEDPROCESSREQUEST = _descriptor.Descriptor(
  name='ChunkedProcessRequest',
  full_name='tensorflow.serving.ChunkedProcessRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='function_spec', full_name='tensorflow.serving.ChunkedProcessRequest.function_spec', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inputs', full_name='tensorflow.serving.ChunkedProcessRequest.inputs', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_filter', full_name='tensorflow.serving.ChunkedProcessRequest.output_filter', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='tensorflow.serving.ChunkedProcessRequest.shape', index=3,
      number=4, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dtype', full_name='tensorflow.serving.ChunkedProcessRequest.dtype', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_CHUNKEDPROCESSREQUEST_INPUTSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=463,
  serialized_end=714,
)


_CHUNKEDPROCESSRESPONSE_OUTPUTSENTRY = _descriptor.Descriptor(
  name='OutputsEntry',
  full_name='tensorflow.serving.ChunkedProcessResponse.OutputsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.serving.ChunkedProcessResponse.OutputsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.serving.ChunkedProcessResponse.OutputsEntry.value', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=847,
  serialized_end=893,
)

_CHUNKEDPROCESSRESPONSE = _descriptor.Descriptor(
  name='ChunkedProcessResponse',
  full_name='tensorflow.serving.ChunkedProcessResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='outputs', full_name='tensorflow.serving.ChunkedProcessResponse.outputs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='tensorflow.serving.ChunkedProcessResponse.shape', index=1,
      number=4, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dtype', full_name='tensorflow.serving.ChunkedProcessResponse.dtype', index=2,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_CHUNKEDPROCESSRESPONSE_OUTPUTSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=717,
  serialized_end=893,
)

_PROCESSREQUEST_INPUTSENTRY.fields_by_name['value'].message_type = tensor__pb2._TENSORPROTO
_PROCESSREQUEST_INPUTSENTRY.containing_type = _PROCESSREQUEST
_PROCESSREQUEST.fields_by_name['function_spec'].message_type = function__pb2._FUNCTIONSPEC
_PROCESSREQUEST.fields_by_name['inputs'].message_type = _PROCESSREQUEST_INPUTSENTRY
_PROCESSRESPONSE_OUTPUTSENTRY.fields_by_name['value'].message_type = tensor__pb2._TENSORPROTO
_PROCESSRESPONSE_OUTPUTSENTRY.containing_type = _PROCESSRESPONSE
_PROCESSRESPONSE.fields_by_name['outputs'].message_type = _PROCESSRESPONSE_OUTPUTSENTRY
_CHUNKEDPROCESSREQUEST_INPUTSENTRY.containing_type = _CHUNKEDPROCESSREQUEST
_CHUNKEDPROCESSREQUEST.fields_by_name['function_spec'].message_type = function__pb2._FUNCTIONSPEC
_CHUNKEDPROCESSREQUEST.fields_by_name['inputs'].message_type = _CHUNKEDPROCESSREQUEST_INPUTSENTRY
_CHUNKEDPROCESSRESPONSE_OUTPUTSENTRY.containing_type = _CHUNKEDPROCESSRESPONSE
_CHUNKEDPROCESSRESPONSE.fields_by_name['outputs'].message_type = _CHUNKEDPROCESSRESPONSE_OUTPUTSENTRY
DESCRIPTOR.message_types_by_name['ProcessRequest'] = _PROCESSREQUEST
DESCRIPTOR.message_types_by_name['ProcessResponse'] = _PROCESSRESPONSE
DESCRIPTOR.message_types_by_name['ChunkedProcessRequest'] = _CHUNKEDPROCESSREQUEST
DESCRIPTOR.message_types_by_name['ChunkedProcessResponse'] = _CHUNKEDPROCESSRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ProcessRequest = _reflection.GeneratedProtocolMessageType('ProcessRequest', (_message.Message,), dict(

  InputsEntry = _reflection.GeneratedProtocolMessageType('InputsEntry', (_message.Message,), dict(
    DESCRIPTOR = _PROCESSREQUEST_INPUTSENTRY,
    __module__ = 'process_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.serving.ProcessRequest.InputsEntry)
    ))
  ,
  DESCRIPTOR = _PROCESSREQUEST,
  __module__ = 'process_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.ProcessRequest)
  ))
_sym_db.RegisterMessage(ProcessRequest)
_sym_db.RegisterMessage(ProcessRequest.InputsEntry)

ProcessResponse = _reflection.GeneratedProtocolMessageType('ProcessResponse', (_message.Message,), dict(

  OutputsEntry = _reflection.GeneratedProtocolMessageType('OutputsEntry', (_message.Message,), dict(
    DESCRIPTOR = _PROCESSRESPONSE_OUTPUTSENTRY,
    __module__ = 'process_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.serving.ProcessResponse.OutputsEntry)
    ))
  ,
  DESCRIPTOR = _PROCESSRESPONSE,
  __module__ = 'process_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.ProcessResponse)
  ))
_sym_db.RegisterMessage(ProcessResponse)
_sym_db.RegisterMessage(ProcessResponse.OutputsEntry)

ChunkedProcessRequest = _reflection.GeneratedProtocolMessageType('ChunkedProcessRequest', (_message.Message,), dict(

  InputsEntry = _reflection.GeneratedProtocolMessageType('InputsEntry', (_message.Message,), dict(
    DESCRIPTOR = _CHUNKEDPROCESSREQUEST_INPUTSENTRY,
    __module__ = 'process_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.serving.ChunkedProcessRequest.InputsEntry)
    ))
  ,
  DESCRIPTOR = _CHUNKEDPROCESSREQUEST,
  __module__ = 'process_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.ChunkedProcessRequest)
  ))
_sym_db.RegisterMessage(ChunkedProcessRequest)
_sym_db.RegisterMessage(ChunkedProcessRequest.InputsEntry)

ChunkedProcessResponse = _reflection.GeneratedProtocolMessageType('ChunkedProcessResponse', (_message.Message,), dict(

  OutputsEntry = _reflection.GeneratedProtocolMessageType('OutputsEntry', (_message.Message,), dict(
    DESCRIPTOR = _CHUNKEDPROCESSRESPONSE_OUTPUTSENTRY,
    __module__ = 'process_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.serving.ChunkedProcessResponse.OutputsEntry)
    ))
  ,
  DESCRIPTOR = _CHUNKEDPROCESSRESPONSE,
  __module__ = 'process_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.ChunkedProcessResponse)
  ))
_sym_db.RegisterMessage(ChunkedProcessResponse)
_sym_db.RegisterMessage(ChunkedProcessResponse.OutputsEntry)


DESCRIPTOR._options = None
_PROCESSREQUEST_INPUTSENTRY._options = None
_PROCESSRESPONSE_OUTPUTSENTRY._options = None
_CHUNKEDPROCESSREQUEST_INPUTSENTRY._options = None
_CHUNKEDPROCESSRESPONSE_OUTPUTSENTRY._options = None
# @@protoc_insertion_point(module_scope)
