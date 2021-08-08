# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import subprocess
import tempfile
from distutils.version import LooseVersion
try:
    from tensorflow.compiler.tf2xla import tf2xla_pb2
except ImportError:
    from tensorflow.neuron.python.tf2xla import tf2xla_pb2
from tensorflow.compiler.xla.service import hlo_pb2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.neuron.python import utils
from tensorflow.neuron.python.hlo.optimize import HloOptimizer
from tensorflow.neuron.python.neuron_cc import find_neuron_cc


_SUPPORTED_OPERATOR_TYPES = '''
Add
AddN
AddV2
Any
AvgPool
AvgPool3D
BatchMatMul
BatchMatMulV2
BatchToSpaceND
BiasAdd
Cast
Concat
ConcatV2
Const
Conv2D
Conv2DBackpropInput
Conv3D
Conv3DBackpropInputV2
Cumsum
DepthwiseConv2dNative
Einsum
Elu
Erf
Exp
ExpandDims
FusedBatchNorm
FusedBatchNormV2
FusedBatchNormV3
Greater
Identity
LeakyRelu
LogicalAnd
LogicalNot
MatMul
Max
MaxPool
MaxPool3D
Maximum
Minimum
Mean
Mul
Neg
NotEqual
Pack
Pad
Pow
RealDiv
Relu
Relu6
Reshape
Rsqrt
Select
SelectV2
Selu
Sigmoid
Softmax
Softplus
Softsign
SpaceToBatchND
Split
SplitV
Sqrt
Square
SquaredDifference
Squeeze
StridedSlice
Sub
Sum
Tanh
Tile
Transpose
Unpack
'''


def list_operators():
    supported_operator_types = set(_SUPPORTED_OPERATOR_TYPES.split('\n'))
    supported_operator_types.remove('')
    return supported_operator_types


def compile_savetemps(graph_def, inputs, outputs, node_name, dumper=None):
    # form tf2xla Config
    tf2xla_config = tf2xla_pb2.Config()
    for tensors, container in zip([inputs, outputs], [tf2xla_config.feed, tf2xla_config.fetch]):
        for ts in tensors:
            item = container.add()
            op_name, output_index = ts.name.split(':')
            output_index = int(output_index)
            item.id.node_name = op_name
            item.id.output_index = output_index
            item.shape.CopyFrom(ts.shape)
            item.type = ts.dtype.as_datatype_enum

    # call aws_neuron_tf2hlo
    temp_path = hlo_pb2.__file__
    for _ in range(4):
        temp_path = os.path.dirname(temp_path)
    aws_neuron_tf2hlo_path = os.path.join(temp_path, 'neuron', 'tf2hlo', 'aws_neuron_tf2hlo')
    graph_def_name = 'graph_def.pb'
    tf2xla_config_name = 'tf2xla_config.pb'
    hlo_snapshot_name = 'hlo_snapshot.pb'
    tfn_args, compiler_args = utils.parse_neuron_cc_flags(dest_set={'--dump-prefix'})
    with tempfile.TemporaryDirectory() as workdir:
        if tfn_args.dump_prefix is not None:
            workdir = os.path.join(os.path.realpath(tfn_args.dump_prefix), node_name)
            os.makedirs(workdir, exist_ok=True)
        compiler_args.append('--dump-prefix={}'.format(workdir))
        graph_def_path = os.path.join(workdir, graph_def_name)
        tf2xla_config_path = os.path.join(workdir, tf2xla_config_name)
        hlo_snapshot_path = os.path.join(workdir, hlo_snapshot_name)
        with open(graph_def_path, 'wb') as f:
            f.write(graph_def.SerializeToString())
        with open(tf2xla_config_path, 'wb') as f:
            f.write(tf2xla_config.SerializeToString())
        command = [aws_neuron_tf2hlo_path, '--graph={}'.format(graph_def_path),
                   '--config={}'.format(tf2xla_config_path),
                   '--out_session_module={}'.format(hlo_snapshot_path)]
        proc = subprocess.run(command, cwd=workdir)
        if proc.returncode != 0:
            return b'', None, None
        hlo_snapshot = hlo_pb2.HloSnapshot()
        with open(hlo_snapshot_path, 'rb') as f:
            hlo_snapshot.ParseFromString(f.read())
        hlo_module = hlo_snapshot.hlo.hlo_module
        executable, new_inputs, new_outputs = hlo2neff(hlo_module, compiler_args, dumper)
    return executable, new_inputs, new_outputs


def hlo2neff(hlo_module, args=None, dumper=None):
    hlo_opt = HloOptimizer(hlo_module)
    hlo_opt.fold_no_op_instructions()
    hlo_opt.dead_code_elimination()
    hlo_opt.flip_broadcast_gather()
    hlo_opt.constant_folding()
    hlo_opt.dead_code_elimination()
    hlo_opt.batchify_reshape_dot_reshape()
    hlo_opt.fold_no_op_instructions()
    hlo_opt.dead_code_elimination()
    hlo_opt.maybe_enable_rtr_shuffle()
    hlo_opt.maybe_enable_dynamic_batch_size()
    hlo_opt.maybe_rewrite_batch_size()
    parsed_args, _ = utils.parse_neuron_cc_flags(args)
    _maybe_dump_bytes_as(parsed_args, hlo_opt.get_snapshot().SerializeToString, 'hlo_snapshot_opt.pb')
    dumper.maybe_embed_io_tensors_into_hlo_snapshots()
    neff_bytes = hlo_opt_to_neff_bytes(hlo_opt, args)
    inputs, outputs = hlo_opt.engrave_io_tensors()
    if parsed_args.dynamic_batch_size:
        for ts in inputs + outputs:
            ts.batch_axis = 0
    return neff_bytes, inputs, outputs


def hlo_opt_to_neff_bytes(hlo_opt, args):
    neff_bytes = _run_neuron_cc_with_dump_prefix(hlo_opt, args)
    if not neff_bytes:
        try:
            from tensorflow_neuron.neuroncc.hlo2neuron.driver import hlo_opt_to_neff_bytes as hlo_opt_to_neff_bytes_fallback
        except ImportError:
            return b''
        logging.warning('running a fall-back code generator to mitigate compilation failure')
        parsed_args, _ = utils.parse_neuron_cc_flags(args)
        if parsed_args.neuroncore_pipeline_cores is not None:
            raise RuntimeError('--neuroncore-pipeline-cores is unsupported in the fall-back code generator')
        try:
            neff_bytes = hlo_opt_to_neff_bytes_fallback(hlo_opt, args)
        except Exception as err:
            logging.debug('fall-back code generator failed due to {}: {}'.format(type(err).__name__, err))
            return b''

        def lazy_neff_bytes():
            return neff_bytes

        _maybe_dump_bytes_as(parsed_args, lazy_neff_bytes, 'hlo_snapshot_opt.neff')
    return neff_bytes


def _run_neuron_cc_with_dump_prefix(hlo_opt, args):
    parsed_args, compiler_args = utils.parse_neuron_cc_flags(args)
    workdir = parsed_args.dump_prefix
    neff_bytes = b''
    input_path = os.path.join(workdir, 'hlo_module.pb')
    output_path = os.path.join(workdir, 'hlo_module.neff')
    with open(input_path, 'wb') as f:
        f.write(hlo_opt.get_snapshot().hlo.hlo_module.SerializeToString())
    command = [find_neuron_cc(), 'compile', input_path, '--framework', 'XLA',
               '--pipeline', 'compile', 'SaveTemps', '--output', output_path, '--verbose=35']
    command.append('--fp32-cast={}'.format(parsed_args.fp32_cast))
    if parsed_args.neuroncore_pipeline_cores is None:
        command.append('--enable-fast-context-switch')
    else:
        command.append('--neuroncore-pipeline-cores={}'.format(parsed_args.neuroncore_pipeline_cores))
    command = _insert_private_cc_flags(command, parsed_args)
    command.extend(compiler_args)
    with open(os.path.join(workdir, 'neuron_cc_xla.log'), 'w') as f:
        proc = subprocess.run(command, cwd=workdir, stdout=f, stderr=f)
    if proc.returncode == 0:
        with open(output_path, 'rb') as f:
            neff_bytes = f.read()
    return neff_bytes


def _maybe_dump_bytes_as(parsed_args, lazy_content, name):
    if parsed_args.dump_prefix is not None:
        with open(os.path.join(parsed_args.dump_prefix, name), 'wb') as f:
            f.write(lazy_content())


def _insert_private_cc_flags(command, parsed_args):
    if 'matmult' in parsed_args.fp32_cast:
        try:
            import neuroncc
        except ImportError:
            pass
        else:
            if hasattr(neuroncc, '__version__'):
                if LooseVersion(neuroncc.__version__) < LooseVersion('1.7.0.0'):
                    # Enable bitcasted-transpose for the high-precision use case
                    command.append('--tensorizer-options=--no-disable-bitcasted-transpose')
    return command
