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
import sys
import os
import json
import subprocess
import tempfile
import logging
from distutils import spawn
from distutils.version import LooseVersion
from tensorflow_neuron import __version__
from tensorflow_neuron.python import utils


_NEURON_CC_CLI = ['neuron-cc']


def configure_compiler_cli(neuron_cc_cli):
    _NEURON_CC_CLI.clear()
    _NEURON_CC_CLI.extend(neuron_cc_cli)


def list_operators():
    neuron_cc = find_neuron_cc()
    if neuron_cc is None:
        return set()
    command = [neuron_cc, 'list-operators', '--framework', 'TENSORFLOW']
    try:
        list_operators_output = subprocess.check_output(command)
    except subprocess.CalledProcessError:
        logging.warning('neuron-cc is not behaving correctly. Please check neuron-cc '
                        'installation, or reinstall by "pip install --force neuron-cc".')
        return set()
    supported_op_types = {op_type.strip() for op_type in list_operators_output.decode()[:-1].split('\n')}
    tf_reserved_ops = [
        'Placeholder',
        'IdentityN',
    ]
    return supported_op_types.difference(tf_reserved_ops)


def compile_savetemps(graph_def, inputs, outputs, node_name, dumper=None):
    """Returns raw neff bytes (empty bytes if neuron-cc crashed)
    """
    error_return_value = b'', None, None
    # form io-config
    io_config = {
        'inputs': {ts.name: [[dim.size for dim in ts.shape.dim], ts.dtype.name] for ts in inputs},
        'outputs': [ts.name for ts in outputs],
    }

    # find neuron-cc and setup workdir
    neuron_cc = find_neuron_cc()
    if neuron_cc is None:
        return error_return_value
    neuron_cc_input_name = 'graph_def.pb'
    neuron_executable_name = 'graph_def.neff'
    tfn_args, compiler_args = utils.parse_neuron_cc_flags()
    with tempfile.TemporaryDirectory() as workdir:
        if tfn_args.dump_prefix is not None:
            workdir = os.path.join(os.path.realpath(tfn_args.dump_prefix), node_name)
            os.makedirs(workdir, exist_ok=True)
        input_path = os.path.join(workdir, neuron_cc_input_name)
        output_path = os.path.join(workdir, neuron_executable_name)
        with open(input_path, 'wb') as f:
            f.write(graph_def.SerializeToString())
        command = [neuron_cc, 'compile', input_path, '--framework', 'TENSORFLOW',
                   '--pipeline', 'compile', 'SaveTemps', '--output', output_path]
        command.extend(['--io-config', json.dumps(io_config)])
        command.extend(compiler_args)
        if tfn_args.log_level is not None:
            verbose = tfn_args.log_level
            if verbose not in {0, 1, 2}:  # set to default for unrecognized values
                verbose = 35
            command.append('--verbose={}'.format(verbose))
        proc = subprocess.run(command, cwd=workdir)
        if proc.returncode != 0:
            return error_return_value
        with open(output_path, 'rb') as f:
            executable = f.read()
    return executable, inputs, outputs


def find_neuron_cc():
    path = '{}:{}'.format(os.path.dirname(sys.executable), os.environ.get('PATH', ''))
    neuron_cc_cli_name, *_ = _NEURON_CC_CLI
    return spawn.find_executable(neuron_cc_cli_name, path)


def read_default_args():
    _, *args = _NEURON_CC_CLI
    return args


def supports_xla():
    if not hasattr(neuroncc, '__version__'):
        return False
    ncc_ver = LooseVersion(neuroncc.__version__)
    dev_delim_ver = LooseVersion('1.1.0.0')
    dev_min_ver = LooseVersion('1.0.35000.0')
    rel_min_ver = LooseVersion('1.7.0.0')
    return dev_min_ver <= ncc_ver < dev_delim_ver or rel_min_ver <= ncc_ver


try:
    import neuroncc
except ImportError:
    neuroncc = None
else:
    if LooseVersion(__version__) >= LooseVersion('2.0.0') and supports_xla():
        from tensorflow_neuron.python.neuron_cc_hlo import list_operators, compile_savetemps
