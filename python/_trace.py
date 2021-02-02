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
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.eager import function
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.neuron.python import meta_graph_util as mgu
from tensorflow.neuron.python import graph_def_util as gdu
from tensorflow.neuron.python import utils
from tensorflow.neuron.python.neuron_cc import list_operators


def trace(func, example_inputs, must_compile=False):
    """Convert a `ConcreteFunction` to a Neuron-optimized `ConcreteFunction`.

    Args:
        func: A `ConcreteFunction`.
        example_inputs: A `tf.Tensor` or a tuple/list of `tf.Tensor`s for tracing the function.

    Returns:
        A Neuron-optimized `ConcreteFunction`
    """
    if isinstance(example_inputs, ops.Tensor):
        example_inputs = example_inputs,
    if not isinstance(func, function.ConcreteFunction):
        func = def_function.function(func).get_concrete_function(example_inputs)
    if must_compile:
        logging.warning('Enabling must_compile; neuron-cc failures will be thrown as exceptions')
    tfn_args, compiler_args = utils.parse_neuron_cc_flags()

    # Note: input_names is also used for constructing the output ConcreteFunction,
    # where using a dictionary (such as `{ts.name: ts.name for ts in func_inputs}`) can cause
    # the resulted ConcreteFunction to get feed tensors going to the wrong inputs occationally
    captured_inputs = {ts.ref() for _, ts in func.graph.captures}
    input_names = [ts.name for ts in func.inputs if ts.ref() not in captured_inputs]

    # convert all variables to constants
    cfunc = convert_to_constants.convert_variables_to_constants_v2(func)
    graph_def = cfunc.graph.as_graph_def(add_shapes=True)
    original_graph_def = graph_def

    # encode known shapes
    shape_feed_dict = {name: ts.shape for name, ts in zip(input_names, example_inputs)}
    graph_def = gdu.encode_inferred_shapes(graph_def, shape_feed_dict)

    # produce GraphDef by running grappler passes
    meta_graph_def = meta_graph_pb2.MetaGraphDef(graph_def=graph_def)
    sig_def_grappler = mgu.build_signature_def(cfunc.inputs, cfunc.outputs)
    meta_graph_def.signature_def['serving_default'].CopyFrom(sig_def_grappler)
    opt_config = config_pb2.ConfigProto()
    rewriter_config = opt_config.graph_options.rewrite_options
    rewriter_config.meta_optimizer_iterations = 1
    rewriter_config.min_graph_nodes = -1
    graph_passes = [
        'debug_stripper',
        'pruning',
        'dependency',
        'aws_neuron_static_shape_inference',
    ]
    rewriter_config.optimizers.extend(graph_passes)

    # configure operator fusion
    fuser_config = rewriter_config.custom_optimizers.add()
    fuser_config.name = 'aws_neuron_fuse_supported_operators'
    fuser_param_map = fuser_config.parameter_map
    # dynamically determine minimum_segment_size from graph size
    fuser_param_map['minimum_segment_size'].i = len(graph_def.node) // 10
    fuser_param_map['fuse_foldable_nodes'].b = True
    fuser_param_map['supported_op_types'].list.s.extend(item.encode() for item in list_operators())

    # call all grappler passes
    graph_def = tf_optimizer.OptimizeGraph(opt_config, meta_graph_def)

    # call graph_def_util passes
    subgraph_passes = [
        'constfold',
        'debug_stripper',
        'constfold',
        'pruning',
        'dependency',
        'constfold',
        'remap',
        'constfold',
        'memory',
        'constfold',
        'common_subgraph_elimination',
        'constfold',
        'arithmetic',
        'constfold',
        'loop',
        'constfold',
    ]
    graph_def = gdu.run_graph_def_pass_in_subgraphs(graph_def, gdu.convert_shape_to_constant)
    graph_def = mgu.run_grappler_on_subgraphs(graph_def, subgraph_passes)
    graph_def = gdu.run_compiler_on_subgraphs(
        graph_def, tfn_args.dump_prefix, compiler_args, must_compile)
    graph_def = gdu.restore_compiler_failures(graph_def, original_graph_def)
    graph_def = gdu.run_graph_def_pass_in_subgraphs(graph_def, gdu.erase_large_constants)
    graph_def = gdu.set_execution_plan(graph_def)

    # wrap GraphDef as a WrappedFunction
    output_names = _get_output_names(func.outputs, func.structured_outputs)
    cfunc = wrap_function.function_from_graph_def(graph_def, input_names, output_names)

    # some hacks to ensure the new ConcreteFunction will have the same calling signature
    cfunc.graph.structured_input_signature = func.graph.structured_input_signature
    cfunc._set_function_spec(func._function_spec)

    # TODO: remove this hack once https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/eager/wrap_function.py#L377 is fixed
    cfunc_input_names = {argdef.name for argdef in cfunc.function_def.signature.input_arg}
    flat_input_signature = nest.flatten(func.structured_input_signature, expand_composites=True)
    input_name_set = {tss.name for tss in flat_input_signature}
    if cfunc_input_names != input_name_set:
        try:
            cfunc._arg_keywords = func._arg_keywords
        except AttributeError:
            pass

    return cfunc


def _get_output_names(tensors, structured_signature):
    if isinstance(structured_signature, dict):
        # return a map from output argument name to symbolic tensor name
        # in order to let the WrappedFunction's return dictionary have the correct keys
        tensor_specs = nest.flatten(structured_signature, expand_composites=True)
        tensor_spec_name_map = {spec.name: name for name, spec in structured_signature.items()}
        tensor_spec_names = [tensor_spec_name_map[spec.name] for spec in tensor_specs]
        return {name: ts.name for ts, name in zip(tensors, tensor_spec_names)}
    else:
        return [ts.name for ts in tensors]
