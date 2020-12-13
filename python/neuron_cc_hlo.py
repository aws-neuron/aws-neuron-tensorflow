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
from tensorflow.compiler.tf2xla import tf2xla_pb2
from tensorflow.compiler.xla.service import hlo_pb2
from hlo2neuron.driver import hlo2neff


def list_operators():
    supported_operator_types = {
        'Add',
        'AddN',
        'AddV2',
        'BatchMatMul',
        'BatchMatMulV2',
        'BiasAdd',
        'Cast',
        'Const',
        'Conv2D',
        'ExpandDims',
        'Identity',
        'MatMul',
        'MaxPool',
        'Mean',
        'Mul',
        'Pad',
        'Pow',
        'Relu',
        'Reshape',
        'Rsqrt',
        'Softmax',
        'SquaredDifference',
        'Squeeze',
        'StridedSlice',
        'Sub',
        'Tanh',
        'Transpose',
    }
    return supported_operator_types


def compile_savetemps(graph_def, inputs, outputs, workdir=None, compiler_args=None):
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
    temp_path = tf2xla_pb2.__file__
    for _ in range(3):
        temp_path = os.path.dirname(temp_path)
    aws_neuron_tf2hlo_path = os.path.join(temp_path, 'neuron', 'tf2hlo', 'aws_neuron_tf2hlo')
    graph_def_name = 'graph_def.pb'
    tf2xla_config_name = 'tf2xla_config.pb'
    hlo_snapshot_name = 'hlo_snapshot.pb'
    if workdir is None:
        workdir_obj = tempfile.TemporaryDirectory()
        workdir = workdir_obj.name
    else:
        workdir = os.path.realpath(workdir)
        os.makedirs(workdir, exist_ok=True)
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
    executable, new_inputs, new_outputs = hlo2neff(hlo_module)
    input_names = [ts.name for ts in new_inputs]
    output_names = [ts.name for ts in new_outputs]
    return executable, input_names, output_names