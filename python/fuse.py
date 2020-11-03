# Copyright 2020 AWS Neuron. All Rights Reserved.
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
import json
import tempfile
import collections
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import wraps, partial
from contextlib import contextmanager
import tensorflow
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops, constant_op
from tensorflow.python.ops import array_ops, variables, variable_scope, init_ops
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.neuron.python.ops.gen_neuron_op import neuron_op
from tensorflow.neuron.python.graph_util import (
    normalize_operators, most_popular_namescope, logging_show_info,
    get_model_config, find_neuron_cc, erase_large_constants)


_neuron_sess_run_decorated = False
_neuron_graph_to_hook = {}
_neuron_cc_input_name = 'graph_def.pb'
_neuron_executable_name = 'graph_def.neff'
_neuron_grad_dict = {}
_neuron_grad_func_set = set()


@deprecated(None, 'Please refer to AWS documentation on Neuron integrated TensorFlow 2.0.')
@tf_export('neuron.fuse')
def fuse(func=None, *, compiler_args=None, name=None, asynchronous=True, timeout=None,
         verbose=0, workdir=None, input_shapes=None, output_shapes=None,
         batch_size=None, dynamic_batch_size=False, executable=b'', grad_func=None):
    if func is None:
        return partial(
            fuse, compiler_args=compiler_args, name=name, asynchronous=asynchronous, timeout=timeout,
            verbose=verbose, workdir=workdir, input_shapes=input_shapes, output_shapes=output_shapes,
            batch_size=batch_size, dynamic_batch_size=dynamic_batch_size, executable=executable,
            grad_func=grad_func)
    @wraps(func)
    def wrapper(*args, **kwargs):
        eager = executing_eagerly()
        if eager:
            is_asynchronous = False
            ops.disable_eager_execution()
        else:
            is_asynchronous = asynchronous

        # if this is a fused gradient, then change NeuronOp's inputs into something hackable
        is_gradient = False
        if args and _is_neuron_op(args[0]) and func in _neuron_grad_func_set:
            input_list = args[0].inputs
            new_input_list = list(input_list)
            args[0]._inputs_val = new_input_list
            is_gradient = True

        inputs_mgr = TensorManager(is_gradient)
        inputs_mgr.track((args, kwargs))

        default_graph = ops.get_default_graph()
        if default_graph not in _neuron_graph_to_hook:
            _neuron_graph_to_hook[default_graph] = NeuronGraphHook(default_graph)
        graph_hook = _neuron_graph_to_hook[default_graph]

        fuse_graph = ops.Graph()
        with graph_hook.fuse_graph_scope() as latest_fg_var_list:
            with fuse_graph.as_default():
                inputs_mgr.build_placeholder_mapping()
                new_args, new_kwargs = inputs_mgr.build((args, kwargs))
                func_outputs = func(*new_args, **new_kwargs)

        # restore NeuronOp's hacked inputs
        if is_gradient:
            args[0]._inputs_val = input_list
            inputs_mgr.is_gradient = False

        input_tensors = inputs_mgr.tensors()
        placeholders = inputs_mgr.mapped_tensors()
        inputs_mgr.mapping = {value: key for key, value in inputs_mgr.mapping.items()}
        inputs_mgr.build((args, kwargs))
        if name is None:
            op_name = most_popular_namescope(op.name for op in fuse_graph.get_operations()
                                             if op.name not in inputs_mgr.new_op_names)
        else:
            op_name = name
        outputs_mgr = TensorManager()
        outputs_mgr.track(func_outputs)
        outputs = outputs_mgr.tensors()
        if dynamic_batch_size:
            input_batch_axis = _dynamic_batch_size_axis(placeholders)
            output_batch_axis = _dynamic_batch_size_axis(outputs)
            if dynamic_batch_size == 'force':
                output_batch_axis = [0 for _ in outputs]  # todo: infer from graph + placeholders
        else:
            input_batch_axis = [-1 for _ in placeholders]
            output_batch_axis = [-1 for _ in outputs]
        if input_shapes is not None:
            for ts, shape in zip(placeholders, input_shapes):
                ts.set_shape(shape)
        if output_shapes is not None:
            for ts, shape in zip(outputs, output_shapes):
                ts.set_shape(shape)
        if batch_size is not None:
            for ts in placeholders:
                if ts.shape.rank:
                    shape = ts.shape.as_list()
                    if shape[0] is None:
                        shape[0] = batch_size
                        ts.set_shape(shape)
            for ts in outputs:
                if ts.shape.rank:
                    shape = ts.shape.as_list()
                    if shape[0] is None:
                        shape[0] = batch_size
                        ts.set_shape(shape)
        io_config = _io_config(placeholders, outputs)
        neuron_get_cc_job_func = partial(
            graph_hook.neuron_get_cc_job, fuse_graph, latest_fg_var_list,
            workdir=workdir, io_config=io_config, compiler_args=compiler_args,
            verbose=verbose, timeout=timeout, op_name=op_name)
        executable_content = executable
        if not executable_content and not is_asynchronous:
            neuron_cc_job, neff_path = neuron_get_cc_job_func()
            neuron_cc_job()
            with open(neff_path, 'rb') as f:
                executable_content = f.read()
        model_config = get_model_config(executable_content)
        if eager:
            # hack to allow enable_eager_execution; see tensorflow/python/framework/ops.py
            global_default_graph = ops._default_graph_stack._global_default_graph
            ops._default_graph_stack._global_default_graph = None
            ops.enable_eager_execution()
            ops._default_graph_stack._global_default_graph = global_default_graph
        fuse_graph_def = fuse_graph.as_graph_def()
        erase_large_constants(fuse_graph_def)
        with ops.name_scope(op_name):
            output_tensors = neuron_op(
                input_tensors=input_tensors, graph_def=fuse_graph_def.SerializeToString(),
                input_names=[ts.name for ts in placeholders],
                input_shapes=[ts.shape for ts in placeholders],
                input_batch_axis=input_batch_axis,
                output_names=[ts.name for ts in outputs],
                output_dtypes=[ts.dtype for ts in outputs],
                output_shapes=[ts.shape for ts in outputs],
                output_batch_axis=output_batch_axis,
                executable=executable_content,
                model_config=model_config,
            )
        if is_asynchronous and not executable_content:
            graph_hook.map_cc_job_func[output_tensors[0].op] = neuron_get_cc_job_func
            global _neuron_sess_run_decorated
            if not _neuron_sess_run_decorated:
                session.Session.run = neuron_decorate_run(session.Session.run)
                _neuron_sess_run_decorated = True
        if callable(grad_func):
            _neuron_grad_dict[output_tensors[0]] = grad_func
            _neuron_grad_func_set.add(getattr(grad_func, '__wrapped__', grad_func))
        outputs_mgr.mapping = {inner: outer for inner, outer in zip(outputs, output_tensors)}
        return outputs_mgr.build(func_outputs)
    return wrapper


def _is_neuron_op(op):
    return isinstance(op, ops.Operation) and op.type == 'NeuronOp'


def _dynamic_batch_size_axis(tensors):
    tensor_batch_axis = []
    for ts in tensors:
        batch_axis = -1
        if ts.shape.rank:
            shape = ts.shape.as_list()
            if shape[0] is None:
                batch_axis = 0
        tensor_batch_axis.append(batch_axis)
    return tensor_batch_axis


def neuron_decorate_run(func):
    @wraps(func)
    def wrapper(sess, *args, **kwargs):
        graph_hook = _neuron_graph_to_hook.get(sess.graph, None)
        if graph_hook is None or graph_hook.compiler_all_executed:
            return func(sess, *args, **kwargs)
        graph_hook.var_dg_to_numpy = {}
        if graph_hook.var_dg_list:
            decorated_run = session.Session.run
            session.Session.run = decorated_run.__wrapped__
            with session.Session(graph=sess.graph) as temp_sess:
                try:
                    temp_sess.run(*args, **kwargs)
                except:
                    pass
                var_dg_list_numpy = temp_sess.run(graph_hook.var_dg_list)
            for var, var_np in zip(graph_hook.var_dg_list, var_dg_list_numpy):
                graph_hook.var_dg_to_numpy[var] = var_np
            session.Session.run = decorated_run
        neuron_cc_job_list = []
        neuron_op_neff_path_list = []
        for op, get_cc_job_func in graph_hook.map_cc_job_func.items():
            neuron_cc_job, neff_path = get_cc_job_func()
            neuron_cc_job_list.append(neuron_cc_job)
            neuron_op_neff_path_list.append([op, neff_path])
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [executor.submit(job) for job in neuron_cc_job_list]
            for fut in futures:
                fut.result().check_returncode()
        for op, neff_path in neuron_op_neff_path_list:
            with open(neff_path, 'rb') as f:
                executable = f.read()
            model_config = get_model_config(executable)
            op._set_attr('executable', attr_value_pb2.AttrValue(s=compat.as_bytes(executable)))
            model_config = attr_value_pb2.AttrValue.ListValue(i=model_config)
            op._set_attr('model_config', attr_value_pb2.AttrValue(list=model_config))
        graph_hook.compiler_all_executed = True
        return func(sess, *args, **kwargs)
    return wrapper


class TensorManager:

    def __init__(self, is_gradient=False):
        self.mapping = collections.OrderedDict()
        self.new_op_names = set()
        self.is_gradient = is_gradient

    def track(self, values):
        if isinstance(values, (ops.Tensor, variables.Variable)):
            self.mapping[values] = values
        elif isinstance(values, (list, tuple, set, frozenset)):
            for val in values:
                self.track(val)
        elif isinstance(values, collections.Mapping):
            for key_val in values.items():
                self.track(key_val)
        elif self.is_gradient and _is_neuron_op(values):
            self.track(values.inputs)

    def tensors(self):
        return list(self.mapping.keys())

    def mapped_tensors(self):
        return list(self.mapping.values())

    def build_placeholder_mapping(self):
        for ts in self.mapping.keys():
            try:
                name = ts.op.name
            except AttributeError:
                name = None
            placeholder = array_ops.placeholder(ts.dtype, ts.shape, name=name)
            self.new_op_names.add(placeholder.op.name)
            self.mapping[ts] = placeholder

    def build(self, values):
        if isinstance(values, (ops.Tensor, variables.Variable)):
            return self.mapping.get(values, values)
        elif isinstance(values, (tuple, frozenset)):
            return type(values)(self.build(val) for val in values)
        elif isinstance(values, (list, set)):
            contents = []
            while values:
                contents.append(self.build(values.pop()))
            if isinstance(values, list):
                values.extend(contents[::-1])
            else:  # set
                values.update(contents[::-1])
            return values
        elif isinstance(values, collections.Mapping):
            for key, val in values.items():
                new_key, new_val = self.build((key, val))
                if new_key is not key:
                    values.pop(key)
                values[new_key] = new_val
            return values
        elif self.is_gradient and _is_neuron_op(values):
            for idx, val in enumerate(values.inputs):
                values.inputs[idx] = self.build(val)
            return values
        else:
            return values


class NeuronGraphHook:

    tensorflowVariable = variables.VariableV1
    original_get_variable = variable_scope.VariableScope.get_variable

    def __init__(self, graph):
        self.outer_graph = graph
        self.var_dg_list = []
        self.graph_var_fg_to_dg = {}
        self.compiler_all_executed = False
        self.map_cc_job_func = {}

    @contextmanager
    def fuse_graph_scope(self):
        latest_fg_var_list = []

        def fuseVariable(initial_value, *args, **kwargs):
            with self.outer_graph.as_default():
                var_dg = NeuronGraphHook.tensorflowVariable(initial_value, *args, **kwargs)
            with ops.name_scope(var_dg.op.name):
                zeros_initializer = partial(init_ops.Zeros(var_dg.dtype), shape=var_dg.shape)
            var_fg = NeuronGraphHook.tensorflowVariable(zeros_initializer, *args, **kwargs)
            self._register_var_fg_dg(var_fg, var_dg, latest_fg_var_list)
            return var_fg

        current_var_scope = variable_scope.get_variable_scope().name
        new_var_store = variable_scope._VariableStore()  # to avoid var_fg name conflict

        def fuse_get_variable(var_scope, var_store, name, *args, **kwargs):
            var_scope_name = var_scope._name
            var_scope._name = os.path.join(current_var_scope, var_scope.name)
            with self.outer_graph.as_default():
                var_dg = NeuronGraphHook.original_get_variable(var_scope, var_store, name, *args, **kwargs)
            with ops.name_scope(var_dg.op.name):
                zeros_initializer = init_ops.Zeros(var_dg.dtype)
            if len(args) > 3:  # get_variable(name, shape=None, dtype=None, initializer=None, ...
                args = list(args)
                args[3] = zeros_initializer
                args = tuple(args)
            else:
                kwargs['initializer'] = zeros_initializer
            var_fg = NeuronGraphHook.original_get_variable(var_scope, new_var_store, name, *args, **kwargs)
            var_scope._name = var_scope_name
            self._register_var_fg_dg(var_fg, var_dg, latest_fg_var_list)
            return var_fg

        tensorflow.Variable = fuseVariable
        variable_scope.VariableScope.get_variable = fuse_get_variable
        try:
            yield latest_fg_var_list
        finally:
            tensorflow.Variable = NeuronGraphHook.tensorflowVariable
            variable_scope.VariableScope.get_variable = NeuronGraphHook.original_get_variable

    def _register_var_fg_dg(self, var_fg, var_dg, latest_fg_var_list):
        latest_fg_var_list.append(var_fg)
        self.var_dg_list.append(var_dg)
        if var_fg.graph not in self.graph_var_fg_to_dg:
            self.graph_var_fg_to_dg[var_fg.graph] = {}
        self.graph_var_fg_to_dg[var_fg.graph][var_fg] = var_dg

    def neuron_get_cc_job(self, fuse_graph, latest_fg_var_list, workdir,
                          io_config, compiler_args, verbose, timeout, op_name):
        var_fg_name_to_numpy = {}
        for var_fg in latest_fg_var_list:
            var_dg = self.graph_var_fg_to_dg[fuse_graph][var_fg]
            var_fg_name_to_numpy[var_fg.op.name] = self.var_dg_to_numpy[var_dg]
        node_rename_map = {}
        remove_node_names = set()
        with fuse_graph.as_default():
            for op in fuse_graph.get_operations():
                if op.name in var_fg_name_to_numpy:
                    remove_node_names.add(op.name)
                    consumers = op.outputs[0].consumers()
                    remove_node_names.update(cop.name for cop in consumers if cop.type == 'Assign')
                    var_numpy = var_fg_name_to_numpy[op.name]
                    with ops.name_scope(op_name):
                        const = constant_op.constant_v1(var_numpy, name='neuron_const')
                    node_rename_map[const.op.name] = op.name
        graph_def = fuse_graph.as_graph_def()
        if remove_node_names:
            fuse_graph_def = graph_def
            graph_def = graph_pb2.GraphDef()
            graph_def.node.MergeFrom(
                node for node in fuse_graph_def.node if node.name not in remove_node_names)
        for node in graph_def.node:
            if node.name in node_rename_map:
                node.name = node_rename_map[node.name]
        if workdir is None:
            tempdir = tempfile.TemporaryDirectory()
        else:
            TempDir = collections.namedtuple('TempDir', 'name')
            prefix = os.path.join(os.path.abspath(workdir), 'tmp')
            os.makedirs(workdir, exist_ok=True)
            tempdir = TempDir(tempfile.mkdtemp(prefix=prefix))
        graph_def = normalize_operators(graph_def)
        with open(os.path.join(tempdir.name, _neuron_cc_input_name), 'wb') as f:
            f.write(graph_def.SerializeToString())
        neuron_cc = find_neuron_cc()
        command = [neuron_cc, 'compile', _neuron_cc_input_name, '--framework', 'TENSORFLOW',
                   '--output', _neuron_executable_name, '--pipeline', 'compile', 'SaveTemps',
                   '--io-config', io_config]
        if compiler_args is not None:
            command.extend(compiler_args)
        neff_path = os.path.join(tempdir.name, _neuron_executable_name)
        info_string = 'fusing subgraph "{}" with neuron-cc'.format(op_name)
        if not verbose:
            logfile_path = os.path.join(tempdir.name, 'log-fe.txt')
            if workdir is not None:
                info_string = '{}; log file is at {}'.format(info_string, logfile_path)
        def neuron_cc_job():
            with logging_show_info():
                logging.info(info_string)
                if verbose:
                    if isinstance(verbose, int):
                        command.extend(['--verbose', str(int(verbose))])
                    logging.info('calling neuron-cc with: {}'.format(' '.join(command)))
            local_tempdir = tempdir  # keep tempdir alive
            call_neuron_cc = partial(subprocess.run, command, cwd=local_tempdir.name, timeout=timeout)
            if verbose:
                return call_neuron_cc()
            else:
                with open(logfile_path, 'w') as logfd:
                    return call_neuron_cc(stdout=logfd, stderr=logfd)
        return neuron_cc_job, neff_path


def _io_config(input_tensors, output_tensors):
    inputs = {ts.name: [ts.shape.as_list(), ts.dtype.name] for ts in input_tensors}
    outputs = [ts.name for ts in output_tensors]
    return json.dumps({'inputs': inputs, 'outputs': outputs})


@ops.RegisterGradient('NeuronOp')
def neuron_op_grad(op, grad):
    return _neuron_grad_dict[op.outputs[0]](op, grad)
