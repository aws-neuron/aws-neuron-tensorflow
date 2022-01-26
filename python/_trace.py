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
from collections import abc
from distutils.version import LooseVersion
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.eager import function
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.framework.errors import InvalidArgumentError
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.compiler.xla.service import hlo_pb2
from tensorflow.neuron.python import meta_graph_util as mgu
from tensorflow.neuron.python import graph_def_util as gdu
from tensorflow.neuron.python import utils
from tensorflow.neuron.python.neuron_cc import list_operators, supports_xla
from tensorflow.neuron.python.hlo.optimize import HloOp
from tensorflow.neuron.python.custom_call import CustomCallLowering
from tensorflow_neuron import __version__


def trace(func, example_inputs, subgraph_builder_function=None):
    """
    Description
    -----------

    Trace a ``keras.Model`` or a Python callable that can be decorated by
    ``tf.function``, and return an AWS-Neuron-optimized ``keras.Model`` that
    can execute on AWS Machine Learning Accelerators. Tracing is ideal for
    ``keras.Model`` that accepts a list of ``tf.Tensor`` objects and returns
    a list of ``tf.Tensor`` objects. It is expected that users will provide
    example inputs, and the ``trace`` function will execute ``func``
    symbolically and convert it to a ``keras.Model``.

    The returned ``keras.Model`` will support inference only. Attributes or
    variables held by the original function or ``keras.Model`` will be dropped.

    The returned ``keras.Model`` can be exported as SavedModel and served using
    TensorFlow Serving. Please see :ref:`tensorflow-serving` for more
    information about exporting to saved model and serving using TensorFlow
    Serving.

    Options can be passed to Neuron compiler via the environment variable
    ``NEURON_CC_FLAGS``. For example, the syntax
    ``env NEURON_CC_FLAGS="--neuroncore-pipeline-cores=4"`` directs Neuron
    compiler to compile each subgraph to fit in the specified number of
    NeuronCores. This number can be less than the total available NeuronCores
    on an Inf1 instance. See  :ref:`neuron-compiler-cli-reference` for more
    information about compiler options.

    Arguments
    ---------

    -   **func:** The ``keras.Model`` or function to be traced.
    -   **example_inputs:** A ``tf.Tensor`` or a tuple/list/dict of
        ``tf.Tensor`` objects for tracing the function. When ``example_inputs``
        is a ``tf.Tensor`` or a list of ``tf.Tensor`` objects, we expect
        ``func`` to have calling signature ``func(example_inputs)``. Otherwise,
        the expectation is that inference on ``func`` is done by calling
        ``func(*example_inputs)`` when ``example_inputs`` is a ``tuple``,
        or ``func(**example_inputs)`` when ``example_inputs`` is a ``dict``.
        The case where ``func`` accepts mixed positional and keyword arguments
        is currently unsupported.
    -   **subgraph_builder_function:** (Optional) A callable with signature

        ``subgraph_builder_function(node : NodeDef) -> bool``
        (``NodeDef`` is defined in tensorflow/core/framework/node_def.proto)

        that is used as a call-back function to determine which part of
        the tensorflow GraphDef given by tracing ``func`` will be placed on
        Machine Learning Accelerators.

        If ``subgraph_builder_function`` is not provided, then ``trace`` will
        automatically place operations on Machine Learning Accelerators or
        on CPU to maximize the execution efficiency.

        If it is provided, and ``subgraph_builder_function(node)`` returns
        ``True``, and placing ``node`` on Machine Learning Accelerators
        will not cause deadlocks during execution, then ``trace`` will place
        ``node`` on Machine Learning Accelerators. If
        ``subgraph_builder_function(node)`` returns ``False``, then ``trace``
        will place ``node`` on CPU.

    Returns
    -------

    -  An AWS-Neuron-optimized ``keras.Model``.


    Example Usage
    -------------

    .. code:: python

        import tensorflow as tf
        import tensorflow.neuron as tfn

        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        model = tf.keras.Model(inputs=[input0], outputs=[dense0])
        example_inputs = tf.random.uniform([1, 3])
        model_neuron = tfn.trace(model, example_inputs)  # trace

        model_dir = './model_neuron'
        model_neuron.save(model_dir)
        model_neuron_reloaded = tf.keras.models.load_model(model_dir)


    Example Usage with Manual Device Placement Using `subgraph_builder_function`
    -------------

    .. code:: python

        import tensorflow as tf
        import tensorflow.neuron as tfn

        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        reshape0 = tf.keras.layers.Reshape([1, 3])(dense0)
        output0 = tf.keras.layers.Dense(2)(reshape0)
        model = tf.keras.Model(inputs=[input0], outputs=[output0])
        example_inputs = tf.random.uniform([1, 3])

        def subgraph_builder_function(node):
            return node.op == 'MatMul'

        model_neuron = tfn.trace(
            model, example_inputs,
            subgraph_builder_function=subgraph_builder_function,
        )

    """
    if not supports_xla():
        raise RuntimeError(
            'tfn.trace requires neuron-cc version >= 1.6.0.0; please update to latest neuron-cc '
            'by `pip install neuron-cc -U --extra-index-url=https://pip.repos.neuron.amazonaws.com`')
    if not isinstance(example_inputs, tuple):
        example_inputs = (example_inputs,)
    if not isinstance(func, (def_function.Function, function.ConcreteFunction)):
        input_signature = None
        if all(isinstance(item, ops.Tensor) for item in example_inputs):
            input_signature = [TensorSpec(ts.shape, ts.dtype) for ts in example_inputs]
        func = def_function.function(input_signature=input_signature)(func)
    original_func = func
    if not isinstance(func, function.ConcreteFunction):
        func = func.get_concrete_function(*example_inputs)
    dumper = OptionalDumper()

    # convert all variables to constants
    with utils.change_grappler_logging_level_according_to_cc_flags():
        try:
            if LooseVersion(__version__) < LooseVersion('2.2.0'):
                cfunc = convert_variables_to_constants_v2(func)
            else:
                cfunc = convert_variables_to_constants_v2(func, aggressive_inlining=True)
        except InvalidArgumentError as err:
            if all(op.type != 'StatefulPartitionedCall' for op in func.graph.get_operations()):
                raise err
            error_msg = (
                "convert_variables_to_constants_v2 failed to unpack a StatefulPartitionedCall"
                " operator; this may be due to StatefulPartitionedCall wrapping custom"
                " operators, which is caused by saving or serializing models containing"
                " custom operators. If you are tracing a keras.Model, try to call tfn.trace"
                " before saving and reloading the model, and please make sure that model"
                " layers do not contain StatefulPartitionedCall wrapping custom operators."
            )
            raise InvalidArgumentError(err.node_def, err.op, error_msg)
    graph_def = cfunc.graph.as_graph_def(add_shapes=True)
    if not any(node.op in {'Placeholder', 'PlaceholderWithDefault'} for node in graph_def.node):
        logging.warning('{} does not seem to have any input; returning an uncompiled callable'.format(func))
        return original_func
    original_graph_def = graph_def

    # encode known shapes
    feed_dict = _get_feed_dict(func, example_inputs)
    shape_feed_dict = {name: tensor.shape for name, tensor in feed_dict.items()}
    graph_def = gdu.encode_inferred_shapes(graph_def, shape_feed_dict)
    dumper.maybe_dump_graph_def_as(graph_def, 'graph_def_shaped.pb')

    # call main-graph grappler passes
    graph_def = _run_shaper_and_fuser(graph_def, feed_dict, func, cfunc, subgraph_builder_function)
    graph_def = gdu.inline_shape_inputs_in_subgraphs(graph_def)
    dumper.maybe_dump_graph_def_as(graph_def, 'graph_def_fused.pb')
    dumper.maybe_compute_io_tensors(graph_def, original_graph_def, func, feed_dict)

    # call graph_def_util/meta_graph_util passes
    graph_def = gdu.run_graph_def_pass_in_subgraphs(graph_def, gdu.convert_shape_to_constant)
    graph_def = mgu.run_grappler_on_subgraphs(graph_def)
    custom_call_lowering = CustomCallLowering()
    graph_def = gdu.run_graph_def_pass_in_subgraphs(graph_def, custom_call_lowering.lower)
    dumper.maybe_dump_graph_def_as(graph_def, 'graph_def_fused_optimized.pb')
    graph_def = gdu.run_compiler_on_subgraphs(graph_def, dumper)
    graph_def = gdu.run_graph_def_pass_in_subgraphs(graph_def, custom_call_lowering.restore)
    graph_def = gdu.restore_compiler_failures(graph_def, original_graph_def)
    graph_def = gdu.run_graph_def_pass_in_subgraphs(graph_def, gdu.erase_large_constants)
    graph_def = gdu.set_execution_plan(graph_def)
    graph_def = gdu.maybe_relax_placeholder_shapes(graph_def)

    # wrap GraphDef as a WrappedFunction
    cfunc = _wrap_graph_def_as_concrete_function(graph_def, func)

    # wrap ConcreteFunction as a keras model
    model = AwsNeuronModel(cfunc, func.structured_outputs)
    _make_keras_model_savable(model, example_inputs)
    return model


class OptionalDumper:

    def __init__(self):
        tfn_args, _ = utils.parse_neuron_cc_flags()
        self.dump_prefix = tfn_args.dump_prefix
        self.dump_tensor_map = None

    def maybe_dump_graph_def_as(self, graph_def, filename):
        if self.dump_prefix is None:
            return
        os.makedirs(self.dump_prefix, exist_ok=True)
        with open(os.path.join(self.dump_prefix, filename), 'wb') as f:
            f.write(graph_def.SerializeToString())

    def maybe_compute_io_tensors(self, graph_def, original_graph_def, func, feed_dict):
        if self.dump_prefix is None:
            return
        input_names = _get_input_names(func)
        tensor_name_map = {}
        for nop in gdu.get_neuron_nodes(graph_def):
            for idx, name in enumerate(nop.attr[gdu.knOutputNames].list.s):
                neuron_tensor_name = nop.name if idx == 0 else '{}:{}'.format(nop.name, idx)
                tensor_name_map[neuron_tensor_name] = name.decode()
        self.dump_tensor_map = {}
        dump_tensor_names = []
        for nop in gdu.get_neuron_nodes(graph_def):
            in_names = [tensor_name_map.get(name, name) for name in nop.input]
            in_names = [name if ':' in name else '{}:0'.format(name) for name in in_names]
            out_names = [name.decode() for name in nop.attr[gdu.knOutputNames].list.s]
            dump_tensor_names.extend(in_names)
            dump_tensor_names.extend(out_names)
            self.dump_tensor_map[nop.name] = in_names, out_names
        dumper_cfunc = wrap_function.function_from_graph_def(original_graph_def, input_names, dump_tensor_names)
        flat_inputs = [feed_dict[name] for name in input_names]
        dumper_outputs = dumper_cfunc(*flat_inputs)
        name_to_tensor = {name: tensor for name, tensor in zip(dump_tensor_names, dumper_outputs)}
        for op_name, (in_names, out_names) in self.dump_tensor_map.items():
            in_tensors = [name_to_tensor[name] for name in in_names]
            out_tensors = [name_to_tensor[name] for name in out_names]
            self.dump_tensor_map[op_name] = in_tensors, out_tensors

    def maybe_dump_hlo_snapshots_with_inputs_outputs(self, hlo_opt):
        if self.dump_prefix is None:
            return
        for op_name, (inputs, outputs) in self.dump_tensor_map.items():
            inputs = [tensor.numpy() for tensor in inputs]
            outputs = [tensor.numpy() for tensor in outputs]
            workdir = os.path.join(self.dump_prefix, op_name)
            if not os.path.isdir(workdir):
                continue
            hlo_ss_path = os.path.join(workdir, 'hlo_snapshot.pb')
            if os.path.isfile(hlo_ss_path):
                hlo_ss = hlo_pb2.HloSnapshot()
                with open(hlo_ss_path, 'rb') as f:
                    hlo_ss.ParseFromString(f.read())
                _embed_inputs_outputs_into_hlo_snapshot(hlo_ss, inputs, outputs)
                with open(hlo_ss_path, 'wb') as f:
                    f.write(hlo_ss.SerializeToString())
            hlo_ss_opt = hlo_opt.get_snapshot()
            if hlo_opt.input_shuffles is not None:
                inputs = [_shuffle(val, shf) for val, shf in zip(inputs, hlo_opt.input_shuffles)]
            _embed_inputs_outputs_into_hlo_snapshot(hlo_ss_opt, inputs, outputs)
            hlo_ss_opt_path = os.path.join(workdir, 'hlo_snapshot_opt.pb')
            with open(hlo_ss_opt_path, 'wb') as f:
                f.write(hlo_ss_opt.SerializeToString())


def _embed_inputs_outputs_into_hlo_snapshot(hlo_snapshot, inputs, outputs):
    hps = hlo_snapshot.hlo.hlo_module.host_program_shape
    iter_inputs = zip(hps.parameters, inputs), hlo_snapshot.arguments
    iter_outputs = zip(hps.result.tuple_shapes, outputs), hlo_snapshot.result.tuple_literals
    for iterator, args in iter_inputs, iter_outputs:
        for arg_shape, value in iterator:
            arg = args.add()
            arg.shape.CopyFrom(arg_shape)
            attr_name = HloOp.xla_dtype_to_literal_attr_name[arg.shape.element_type]
            literals = getattr(arg, attr_name)
            if isinstance(literals, bytes):
                setattr(arg, attr_name, value.tobytes())
            else:
                literals[:] = value.ravel()


def _shuffle(value, shuffle):
    return value.ravel()[shuffle].reshape(value.shape)


class AwsNeuronModel(Model):

    def __init__(self, func, structured_outputs):
        super().__init__(trainable=False, autocast=False)
        self.aws_neuron_function = func
        self._aws_neuron_output_type = None
        if isinstance(structured_outputs, abc.Mapping):
            self._aws_neuron_output_type = type(structured_outputs)

    def call(self, inputs, *args):
        flat_inputs = nest.flatten((inputs, args))
        if isinstance(inputs, abc.Mapping) and not args:
            if set(inputs.keys()) == set(self.aws_neuron_function._arg_keywords):
                flat_inputs = [inputs[kw] for kw in self.aws_neuron_function._arg_keywords]
        outputs = self.aws_neuron_function(*flat_inputs)
        if self._aws_neuron_output_type is not None:
            outputs = self._aws_neuron_output_type(**outputs)
        return outputs


def _get_feed_dict(func, inputs):
    if len(inputs) == 1:
        inputs, = inputs
    input_names = _get_input_names(func)
    inputs_list = [inputs] if isinstance(inputs, ops.Tensor) else inputs
    inputs_list = nest.flatten(inputs_list)
    return {name: ts for name, ts in zip(input_names, inputs_list)}


def _run_shaper_and_fuser(graph_def, feed_dict, func, cfunc, subgraph_builder_function):
    new_graph_def = _run_grappler_on_main_graph(graph_def, cfunc, subgraph_builder_function)
    name_mapping = {}
    for node in gdu.get_neuron_nodes(new_graph_def):
        output_names = node.attr[gdu.knOutputNames].list.s
        for port, name in enumerate(output_names):
            name_mapping['{}:{}'.format(node.name, port)] = name.decode()
    need_shape_names = []
    for node in gdu.get_neuron_nodes(new_graph_def):
        is_compilable, _ = gdu.neuron_node_is_compilable(node)
        if not is_compilable:
            input_shapes = node.attr[gdu.knInputShapes].list.shape
            for name, shape in zip(node.input, input_shapes):
                if not TensorShape(shape).is_fully_defined():
                    if ':' not in name:
                        name = '{}:0'.format(name)
                    need_shape_names.append(name_mapping.get(name, name))
    if need_shape_names:
        logging.warning('determining {} tensor shapes concretely from runtime'.format(len(need_shape_names)))
        input_names = _get_input_names(func)
        shaper_cfunc = wrap_function.function_from_graph_def(graph_def, input_names, need_shape_names)
        flat_inputs = [feed_dict[name] for name in input_names]
        shaper_outputs = shaper_cfunc(*flat_inputs)
        shape_feed_dict = {name: tensor.shape for name, tensor in feed_dict.items()}
        shape_feed_dict.update((name, ts.shape) for name, ts in zip(need_shape_names, shaper_outputs))
        graph_def = gdu.encode_inferred_shapes(graph_def, shape_feed_dict)
        graph_def = _run_grappler_on_main_graph(graph_def, cfunc, subgraph_builder_function)
    else:
        graph_def = new_graph_def
    return graph_def


def _run_grappler_on_main_graph(graph_def, cfunc, subgraph_builder_function):
    # produce GraphDef by running grappler passes
    opt_config, meta_graph_def = _build_optimize_graph_args(graph_def, cfunc)
    rewriter_config = opt_config.graph_options.rewrite_options
    optimizers = rewriter_config.optimizers
    optimizers.append('debug_stripper')
    pruning_passes = ['pruning', 'dependency']
    if subgraph_builder_function is None:
        optimizers.extend(pruning_passes)
    optimizers.append('aws_neuron_static_shape_inference')
    optimizers.append('aws_neuron_split_conv2d_same_padding')
    optimizers.append('aws_neuron_mark_ops_in_fixed_shape_context')

    # configure operator fusion
    fuser_config = rewriter_config.custom_optimizers.add()
    fuser_config.name = 'aws_neuron_fuse_supported_operators'
    fuser_param_map = fuser_config.parameter_map
    fuser_param_map['supported_op_types'].list.s.extend(item.encode() for item in list_operators())
    if subgraph_builder_function is None:
        fuser_param_map['fuse_foldable_nodes'].b = True
        fuser_param_map['prune_small_subgraphs_ratio'].f = 0.8
        no_fuse_ops = _find_pad_ops_preceding_conv2d(cfunc.graph)
        no_fuse_ops.extend(_find_int64_select_ops(cfunc.graph))
    else:
        force_fuse_ops = [node.name for node in graph_def.node if subgraph_builder_function(node)]
        fuser_param_map['force_fuse_ops'].list.s.extend(item.encode() for item in force_fuse_ops)
        no_fuse_ops = [node.name for node in graph_def.node]
    if no_fuse_ops:
        fuser_param_map['no_fuse_ops'].list.s.extend(item.encode() for item in no_fuse_ops)

    # call all grappler passes
    with utils.change_grappler_logging_level_according_to_cc_flags():
        graph_def = tf_optimizer.OptimizeGraph(opt_config, meta_graph_def)
    if subgraph_builder_function is not None:
        opt_config, meta_graph_def = _build_optimize_graph_args(graph_def, cfunc)
        optimizers = opt_config.graph_options.rewrite_options.optimizers
        optimizers.extend(pruning_passes)
        with utils.change_grappler_logging_level_according_to_cc_flags():
            graph_def = tf_optimizer.OptimizeGraph(opt_config, meta_graph_def)
    return graph_def


def _build_optimize_graph_args(graph_def, cfunc):
    meta_graph_def = meta_graph_pb2.MetaGraphDef(graph_def=graph_def)
    sig_def = mgu.build_signature_def(cfunc.inputs, cfunc.outputs)
    meta_graph_def.signature_def['serving_default'].CopyFrom(sig_def)
    opt_config = config_pb2.ConfigProto()
    rewriter_config = opt_config.graph_options.rewrite_options
    rewriter_config.meta_optimizer_iterations = 1
    rewriter_config.min_graph_nodes = -1
    return opt_config, meta_graph_def


def _find_pad_ops_preceding_conv2d(graph):
    # exclude Pad that precedes Conv2D for HLO frontend
    no_fuse_ops = []
    supported_op_types = list_operators()
    for op in graph.get_operations():
        if op.type == 'Pad':
            consumers = op.outputs[0].consumers()
            if consumers and consumers[0].type == 'Conv2D':
                curr_op = op
                pad_input_ops = [curr_op]
                while curr_op.inputs and curr_op.type in supported_op_types:
                    curr_op = curr_op.inputs[0].op
                    pad_input_ops.append(curr_op)
                if len(pad_input_ops) <= 3:
                    no_fuse_ops.append(op.inputs[1].op.name)
                    no_fuse_ops.extend(piop.name for piop in pad_input_ops)
    return no_fuse_ops


def _find_int64_select_ops(graph):
    predicate = lambda op: op.type == 'Select' and op.get_attr('T') == dtypes.int64
    return [op.name for op in graph.get_operations() if predicate(op)]


def _get_input_names(func):
    captured_inputs = {ts.name for _, ts in func.graph.captures}
    return [ts.name for ts in func.inputs if ts.name not in captured_inputs]


def _get_output_names(func):
    outputs, structured_outputs = func.outputs, func.structured_outputs
    if isinstance(structured_outputs, dict):
        # return a map from output argument name to symbolic tensor name
        # in order to let the WrappedFunction's return dictionary have the correct keys
        tensor_specs = nest.flatten(structured_outputs, expand_composites=True)
        tensor_spec_name_map = {spec.name: name for name, spec in structured_outputs.items()}
        tensor_spec_names = [tensor_spec_name_map[spec.name] for spec in tensor_specs]
        return {name: ts.name for ts, name in zip(outputs, tensor_spec_names)}
    elif len(outputs) == 1 and isinstance(structured_outputs, ops.Tensor):
        output, = outputs
        return output.name
    else:
        return [ts.name for ts in outputs]


def _wrap_graph_def_as_concrete_function(graph_def, func_ref):
    # Note: if input_names is a dictionary (such as `{ts.name: ts.name for ts in example_inputs}`),
    # then the WrappedFunction may occationally have feeding tensors going to the wrong inputs.
    input_names = _get_input_names(func_ref)
    output_names = _get_output_names(func_ref)
    cfunc = wrap_function.function_from_graph_def(graph_def, input_names, output_names)

    # TODO: remove this hack once https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/eager/wrap_function.py#L377 is fixed
    try:
        if cfunc._arg_keywords != func_ref._arg_keywords:
            cfunc._arg_keywords = func_ref._arg_keywords
    except AttributeError:
        pass
    return cfunc


def _make_keras_model_savable(model, example_inputs):
    if len(example_inputs) == 1:
        example_inputs, = example_inputs
    # hack to propagate metadata for saving
    if hasattr(model, '_set_save_spec'):
        set_save_spec = model._set_save_spec
    elif hasattr(model, '_set_input_attrs'):
        set_save_spec = model._set_input_attrs
    else:
        set_save_spec = None
        logging.warning('Not setting inputs for the traced model {}; you may need to run inference '
                        'before trying to save it'.format(model))
    if set_save_spec is not None:
        set_save_spec(example_inputs)
