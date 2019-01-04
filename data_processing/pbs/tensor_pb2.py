# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensor.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import data_processing.pbs.resource_handle_pb2 as resource__handle__pb2
import data_processing.pbs.tensor_shape_pb2 as tensor__shape__pb2
import data_processing.pbs.types_pb2 as types__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensor.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_options=_b('\n\030org.tensorflow.frameworkB\014TensorProtosP\001\370\001\001'),
  serialized_pb=_b('\n\x0ctensor.proto\x12\ntensorflow\x1a\x15resource_handle.proto\x1a\x12tensor_shape.proto\x1a\x0btypes.proto\"\x9a\x03\n\x0bTensorProto\x12#\n\x05\x64type\x18\x01 \x01(\x0e\x32\x14.tensorflow.DataType\x12\x32\n\x0ctensor_shape\x18\x02 \x01(\x0b\x32\x1c.tensorflow.TensorShapeProto\x12\x16\n\x0eversion_number\x18\x03 \x01(\x05\x12\x16\n\x0etensor_content\x18\x04 \x01(\x0c\x12\x14\n\x08half_val\x18\r \x03(\x05\x42\x02\x10\x01\x12\x15\n\tfloat_val\x18\x05 \x03(\x02\x42\x02\x10\x01\x12\x16\n\ndouble_val\x18\x06 \x03(\x01\x42\x02\x10\x01\x12\x0f\n\x07int_val\x18\x07 \x03(\x05\x12\x12\n\nstring_val\x18\x08 \x03(\x0c\x12\x18\n\x0cscomplex_val\x18\t \x03(\x02\x42\x02\x10\x01\x12\x15\n\tint64_val\x18\n \x03(\x03\x42\x02\x10\x01\x12\x14\n\x08\x62ool_val\x18\x0b \x03(\x08\x42\x02\x10\x01\x12\x18\n\x0c\x64\x63omplex_val\x18\x0c \x03(\x01\x42\x02\x10\x01\x12\x37\n\x13resource_handle_val\x18\x0e \x03(\x0b\x32\x1a.tensorflow.ResourceHandleB-\n\x18org.tensorflow.frameworkB\x0cTensorProtosP\x01\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[resource__handle__pb2.DESCRIPTOR,tensor__shape__pb2.DESCRIPTOR,types__pb2.DESCRIPTOR,])




_TENSORPROTO = _descriptor.Descriptor(
  name='TensorProto',
  full_name='tensorflow.TensorProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dtype', full_name='tensorflow.TensorProto.dtype', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tensor_shape', full_name='tensorflow.TensorProto.tensor_shape', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version_number', full_name='tensorflow.TensorProto.version_number', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tensor_content', full_name='tensorflow.TensorProto.tensor_content', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='half_val', full_name='tensorflow.TensorProto.half_val', index=4,
      number=13, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='float_val', full_name='tensorflow.TensorProto.float_val', index=5,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='double_val', full_name='tensorflow.TensorProto.double_val', index=6,
      number=6, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='int_val', full_name='tensorflow.TensorProto.int_val', index=7,
      number=7, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='string_val', full_name='tensorflow.TensorProto.string_val', index=8,
      number=8, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scomplex_val', full_name='tensorflow.TensorProto.scomplex_val', index=9,
      number=9, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='int64_val', full_name='tensorflow.TensorProto.int64_val', index=10,
      number=10, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bool_val', full_name='tensorflow.TensorProto.bool_val', index=11,
      number=11, type=8, cpp_type=7, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dcomplex_val', full_name='tensorflow.TensorProto.dcomplex_val', index=12,
      number=12, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resource_handle_val', full_name='tensorflow.TensorProto.resource_handle_val', index=13,
      number=14, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=85,
  serialized_end=495,
)

_TENSORPROTO.fields_by_name['dtype'].enum_type = types__pb2._DATATYPE
_TENSORPROTO.fields_by_name['tensor_shape'].message_type = tensor__shape__pb2._TENSORSHAPEPROTO
_TENSORPROTO.fields_by_name['resource_handle_val'].message_type = resource__handle__pb2._RESOURCEHANDLE
DESCRIPTOR.message_types_by_name['TensorProto'] = _TENSORPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TensorProto = _reflection.GeneratedProtocolMessageType('TensorProto', (_message.Message,), dict(
  DESCRIPTOR = _TENSORPROTO,
  __module__ = 'tensor_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.TensorProto)
  ))
_sym_db.RegisterMessage(TensorProto)


DESCRIPTOR._options = None
_TENSORPROTO.fields_by_name['half_val']._options = None
_TENSORPROTO.fields_by_name['float_val']._options = None
_TENSORPROTO.fields_by_name['double_val']._options = None
_TENSORPROTO.fields_by_name['scomplex_val']._options = None
_TENSORPROTO.fields_by_name['int64_val']._options = None
_TENSORPROTO.fields_by_name['bool_val']._options = None
_TENSORPROTO.fields_by_name['dcomplex_val']._options = None
# @@protoc_insertion_point(module_scope)
