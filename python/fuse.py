"""
Copyright (C) 2019, Amazon.com. All Rights Reserved
"""
import os
import json
import tempfile
import collections
import subprocess
from functools import wraps, partial
from distutils import spawn
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.python.neuron.python.graph_util import normalize_operators, most_popular_namescope


_neuron_cc_input_name = 'graph_def.pb'
_neuron_executable_name = 'graph_def.neff'


@tf_export('neuron.fuse')
def fuse(func=None, *, compiler_args=None, name=None, greedy=False, timeout=None,
         verbose=0, workdir=None):
    if func is None:
        return partial(fuse, compiler_args=compiler_args, name=name, greedy=greedy,
                       timeout=timeout, verbose=verbose, workdir=workdir)
    @wraps(func)
    def wrapper(*args, **kwargs):
        # need to import here; otherwise bazel sees @tf_export in gen_neuron_op
        from tensorflow.python.neuron.ops.gen_neuron_op import neuron_op

        eager = executing_eagerly()
        if eager:
            is_greedy = True
            ops.disable_eager_execution()
        else:
            is_greedy = greedy
        inputs_mgr = TensorManager()
        if workdir is None:
            tempdir = tempfile.TemporaryDirectory()
        else:
            TempDir = collections.namedtuple('TempDir', 'name')
            prefix = os.path.join(os.path.abspath(workdir), 'tmp')
            tempdir = TempDir(tempfile.mkdtemp(prefix=prefix))
        inputs_mgr.track((args, kwargs))
        with session.Session(graph=ops.Graph()) as sess:
            inputs_mgr.build_placeholder_mapping()
            new_args, new_kwargs = inputs_mgr.build((args, kwargs))
            func_outputs = func(*new_args, **new_kwargs)
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
        graph_def = normalize_operators(sess.graph.as_graph_def(add_shapes=True))
        with open(os.path.join(tempdir.name, _neuron_cc_input_name), 'wb') as f:
            serialized_graph_def = graph_def.SerializeToString()
            f.write(serialized_graph_def)
        neuron_cc = spawn.find_executable('neuron-cc')
        command = [neuron_cc, 'compile', _neuron_cc_input_name, '--framework', 'TENSORFLOW',
                   '--output', _neuron_executable_name, '--io-config', _io_config(placeholders, outputs)]
        if compiler_args is not None:
            command.extend(compiler_args)
        if workdir is None:
            if verbose:
                popen_kwargs = {}
            else:
                popen_kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logfd = None
        else:
            logfile_path = os.path.join(tempdir.name, 'log-fe.txt')
            logfd = open(logfile_path, 'w')
            popen_kwargs = dict(stdout=logfd, stderr=logfd)
            if verbose:
                verbosity = logging.get_verbosity()
                logging.set_verbosity(logging.INFO)
                logging.info('Writing compiler logs to {}'.format(logfile_path))
                logging.set_verbosity(verbosity)
        proc = subprocess.Popen(command, cwd=tempdir.name, **popen_kwargs)
        Compiler = collections.namedtuple('Compiler', 'proc logfd workdir command')
        compiler = Compiler(proc, logfd, tempdir.name, subprocess.list2cmdline(command))
        if not op_name:
            op_name = None
        executable = _get_executable(compiler, timeout, op_name) if is_greedy else ''
        if eager:
            try:
                ops.enable_eager_execution()
            except ValueError:
                ops.enable_eager_execution()
        with ops.name_scope(op_name):
            output_tensors = neuron_op(
                input_tensors=input_tensors, graph_def=serialized_graph_def,
                input_names=[ts.name for ts in placeholders],
                input_shapes=[ts.shape for ts in placeholders],
                input_batch_axis=[-1 for ts in placeholders],
                output_names=[ts.name for ts in outputs],
                output_dtypes=[ts.dtype for ts in outputs],
                output_shapes=[ts.shape for ts in outputs],
                output_batch_axis=[-1 for ts in outputs],
                executable=executable,
            )
        if not is_greedy:
            iop = output_tensors[0].op
            iop.neuron_workspace = tempdir  # keep tempdir object alive
            iop.neuron_wait = partial(neuron_wait, op=iop, compiler=compiler, timeout=timeout)
            if not hasattr(session.BaseSession._do_run, 'neuron_decorated'):
                session.BaseSession._do_run = neuron_decorate_run(session.BaseSession._do_run)
        inner_to_outer = {}
        for inner, outer in zip(outputs, output_tensors):
            outer.set_shape(inner.shape)
            inner_to_outer[inner] = outer
        outputs_mgr.mapping = inner_to_outer
        return outputs_mgr.build(func_outputs)
    return wrapper


def neuron_wait(op, compiler, timeout):
    executable = _get_executable(compiler, timeout, op.name)
    op._set_attr('executable', attr_value_pb2.AttrValue(s=compat.as_bytes(executable)))
    op.neuron_wait = None
    op.neuron_workspace = None


def _get_executable(compiler, timeout, op_name):
    compiler.proc.wait(timeout=timeout)
    if compiler.logfd is not None:
        compiler.logfd.close()
    if compiler.proc.returncode != 0:
        raise RuntimeError(
            'Failed to compile op {} with command "{}"'.format(op_name, compiler.command))
    return open(os.path.join(compiler.workdir, _neuron_executable_name), 'rb').read()


def neuron_wait_all(sess):
    for op in sess.graph.get_operations():
        if op.type == 'NeuronOp' and hasattr(op, 'neuron_wait') and callable(op.neuron_wait):
            op.neuron_wait()


def neuron_decorate_run(func):
    @wraps(func)
    def wrapper(sess, *args, **kwargs):
        if not hasattr(sess, 'neuron_wait_all_executed'):
            neuron_wait_all(sess)
        sess.neuron_wait_all_executed = None
        return func(sess, *args, **kwargs)
    wrapper.neuron_decorated = True
    return wrapper


class TensorManager:

    def __init__(self):
        self.mapping = collections.OrderedDict()
        self.new_op_names = set()

    def track(self, values):
        if isinstance(values, ops.Tensor):
            self.mapping[values] = values
        elif isinstance(values, (list, tuple, set, frozenset)):
            for val in values:
                self.track(val)
        elif isinstance(values, collections.Mapping):
            for key_val in values.items():
                self.track(key_val)

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
        if isinstance(values, ops.Tensor):
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
        else:
            return values


def _io_config(input_tensors, output_tensors):
    inputs = {ts.name: [ts.shape.as_list(), ts.dtype.name] for ts in input_tensors}
    outputs = [ts.name for ts in output_tensors]
    return json.dumps({'inputs': inputs, 'outputs': outputs})
