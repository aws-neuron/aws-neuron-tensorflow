"""
Copyright (C) 2019, Amazon.com. All Rights Reserved
"""
import os
import json
import tempfile
import collections
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import wraps, partial
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, variables
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.neuron.ops.gen_neuron_op import neuron_op
from tensorflow.neuron.python.graph_util import (
    normalize_operators, most_popular_namescope, logging_show_info,
    _neff_get_cores_from_executable, find_neuron_cc)


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
        with session.Session(graph=ops.Graph()) as sess:
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
            op_name = most_popular_namescope(op.name for op in sess.graph.get_operations()
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
        graph_def = normalize_operators(sess.graph.as_graph_def()).SerializeToString()
        io_config = _io_config(placeholders, outputs)
        neuron_get_cc_job_func = partial(
            neuron_get_cc_job, graph_def=graph_def, workdir=workdir,
            io_config=io_config, compiler_args=compiler_args, verbose=verbose,
            timeout=timeout, op_name=op_name)
        executable_content = executable
        if not executable_content and not is_asynchronous:
            neuron_cc_job, neff_path = neuron_get_cc_job_func()
            neuron_cc_job()
            with open(neff_path, 'rb') as f:
                executable_content = f.read()
        model_config = _get_model_config(executable_content)
        if eager:
            try:
                ops.enable_eager_execution()
            except ValueError:
                ops.enable_eager_execution()
        with ops.name_scope(op_name):
            output_tensors = neuron_op(
                input_tensors=input_tensors, graph_def=graph_def,
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
            output_tensors[0].op.neuron_get_cc_job = neuron_get_cc_job_func
            if not hasattr(session.BaseSession._do_run, 'neuron_decorated'):
                session.BaseSession._do_run = neuron_decorate_run(session.BaseSession._do_run)
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


def neuron_get_cc_job(graph_def, workdir, io_config, compiler_args, verbose, timeout, op_name):
    if workdir is None:
        tempdir = tempfile.TemporaryDirectory()
    else:
        TempDir = collections.namedtuple('TempDir', 'name')
        prefix = os.path.join(os.path.abspath(workdir), 'tmp')
        os.makedirs(workdir, exist_ok=True)
        tempdir = TempDir(tempfile.mkdtemp(prefix=prefix))
    with open(os.path.join(tempdir.name, _neuron_cc_input_name), 'wb') as f:
        f.write(graph_def)
    neuron_cc = find_neuron_cc()
    command = [neuron_cc, 'compile', _neuron_cc_input_name, '--framework', 'TENSORFLOW',
               '--output', _neuron_executable_name, '--pipeline', 'compile', 'SaveTemps',
               '--io-config', io_config]
    if compiler_args is not None:
        command.extend(compiler_args)
    neff_path = os.path.join(tempdir.name, _neuron_executable_name)
    if verbose:
        with logging_show_info():
            logging.info('calling neuron-cc with: {}'.format(' '.join(command)))
        def neuron_cc_job():
            local_tempdir = tempdir  # keep tempdir alive when neuron-cc is running
            return subprocess.run(command, cwd=tempdir.name, timeout=timeout)
    else:
        logfile_path = os.path.join(tempdir.name, 'log-fe.txt')
        info_string = 'fusing subgraph "{}" with neuron-cc'.format(op_name)
        if workdir is not None:
            info_string = '{}; log file is at {}'.format(info_string, logfile_path)
        with logging_show_info():
            logging.info(info_string)
        def neuron_cc_job():
            local_tempdir = tempdir
            with open(logfile_path, 'w') as logfd:
                return subprocess.run(command, cwd=tempdir.name, stdout=logfd, stderr=logfd, timeout=timeout)
    return neuron_cc_job, neff_path


def _get_model_config(executable):
    if not executable:
        return []
    opt_num_cores, min_num_cores = _neff_get_cores_from_executable(executable)
    est_infer_timeout = len(executable) / 1e8
    infer_timeout = max(est_infer_timeout, 10)
    model_config = [-1, opt_num_cores, opt_num_cores, infer_timeout]
    return model_config


def neuron_decorate_run(func):
    @wraps(func)
    def wrapper(sess, *args, **kwargs):
        if not hasattr(sess, 'neuron_compiler_all_executed'):
            neuron_cc_job_list = []
            neuron_op_list = []
            for op in sess.graph.get_operations():
                if op.type == 'NeuronOp' and callable(getattr(op, 'neuron_get_cc_job', None)):
                    neuron_cc_job, neff_path = op.neuron_get_cc_job()
                    neuron_cc_job_list.append(neuron_cc_job)
                    neuron_op_list.append([op, neff_path])
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = [executor.submit(job) for job in neuron_cc_job_list]
                for fut in futures:
                    fut.result().check_returncode()
            for op, neff_path in neuron_op_list:
                with open(neff_path, 'rb') as f:
                    executable = f.read()
                model_config = _get_model_config(executable)
                op._set_attr('executable', attr_value_pb2.AttrValue(s=compat.as_bytes(executable)))
                model_config = attr_value_pb2.AttrValue.ListValue(i=model_config)
                op._set_attr('model_config', attr_value_pb2.AttrValue(list=model_config))
                op.neuron_get_cc_job = None
        sess.neuron_compiler_all_executed = None
        return func(sess, *args, **kwargs)
    wrapper.neuron_decorated = True
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


def _io_config(input_tensors, output_tensors):
    inputs = {ts.name: [ts.shape.as_list(), ts.dtype.name] for ts in input_tensors}
    outputs = [ts.name for ts in output_tensors]
    return json.dumps({'inputs': inputs, 'outputs': outputs})


@ops.RegisterGradient('NeuronOp')
def neuron_op_grad(op, grad):
    return _neuron_grad_dict[op.outputs[0]](op, grad)
