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


def compile_savetemps(graph_def, inputs, outputs, node_name):
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
    tfn_args, compiler_args = utils.parse_neuron_cc_flags()
    with tempfile.TemporaryDirectory() as workdir:
        if tfn_args.dump_prefix is not None:
            workdir = os.path.join(os.path.realpath(tfn_args.dump_prefix), node_name)
            os.makedirs(workdir, exist_ok=True)
            compiler_args.append('--dump-prefix={}'.format(workdir))
        if tfn_args.log_level is not None:
            compiler_args.append('--log-level={}'.format(tfn_args.log_level))
        if tfn_args.dynamic_batch_size:
            compiler_args.append('--dynamic-batch-size')
        compiler_args = _relay_parsed_args(compiler_args, tfn_args)
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
        executable, new_inputs, new_outputs = hlo2neff(hlo_module, node_name, compiler_args)
    return executable, new_inputs, new_outputs


def hlo2neff(hlo_module, node_name, args=None):
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
    neff_bytes = hlo_opt_to_neff_bytes(hlo_opt, node_name, args)
    inputs, outputs = hlo_opt.engrave_io_tensors()
    if parsed_args.dynamic_batch_size:
        for ts in inputs + outputs:
            ts.batch_axis = 0
    return neff_bytes, inputs, outputs


def hlo_opt_to_neff_bytes(hlo_opt, node_name, args):
    parsed_args, unknown_args = utils.parse_neuron_cc_flags(args)
    compiler_args = ['--verbose=35']
    if parsed_args.neuroncore_pipeline_cores is None:
        compiler_args.append('--enable-fast-context-switch')
    compiler_args = _relay_parsed_args(compiler_args, parsed_args)
    compiler_args.extend(unknown_args)
    with tempfile.TemporaryDirectory() as workdir:
        if parsed_args.dump_prefix is not None:
            workdir = os.path.join(os.path.realpath(parsed_args.dump_prefix), node_name)
            os.makedirs(workdir, exist_ok=True)
        neff_bytes = _run_neuron_cc_under_workdir(hlo_opt, workdir, compiler_args)
        if not neff_bytes:
            logging.warning('running a fall-back code generator to mitigate compilation failure')
            try:
                from tensorflow_neuron.neuroncc.hlo2neuron.driver import hlo_opt_to_neff_bytes as hlo_opt_to_neff_bytes_fallback
            except ImportError:
                return b''
            if parsed_args.neuroncore_pipeline_cores is not None:
                raise RuntimeError('--neuroncore-pipeline-cores is unsupported in the fall-back code generator')
            neff_bytes = hlo_opt_to_neff_bytes_fallback(hlo_opt, args)

            def lazy_neff_bytes():
                return neff_bytes

            _maybe_dump_bytes_as(parsed_args, lazy_neff_bytes, 'hlo_snapshot_opt.neff')
    return neff_bytes


def _relay_parsed_args(unknown_args, parsed_args):
    unknown_args.append('--fp32-cast={}'.format(parsed_args.fp32_cast))
    if parsed_args.neuroncore_pipeline_cores is not None:
        unknown_args.append('--neuroncore-pipeline-cores={}'.format(parsed_args.neuroncore_pipeline_cores))
    return unknown_args


def _run_neuron_cc_under_workdir(hlo_opt, workdir, compiler_args):
    neff_bytes = b''
    input_path = os.path.join(workdir, 'hlo_module.pb')
    output_path = os.path.join(workdir, 'hlo_module.neff')
    parsed_args, _ = utils.parse_neuron_cc_flags(compiler_args)
    if (len(hlo_opt.outputs) == 1 and not parsed_args.fp32_cast.startswith('matmult')) or parsed_args.neuroncore_pipeline_cores is not None:
        with open(input_path, 'wb') as f:
            f.write(hlo_opt.get_snapshot().hlo.hlo_module.SerializeToString())
        command = [find_neuron_cc(), 'compile', input_path, '--framework', 'XLA',
                   '--pipeline', 'compile', 'SaveTemps', '--output', output_path]
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

