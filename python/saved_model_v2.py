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
from tensorflow.python import saved_model
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.util import nest
from tensorflow.neuron.python import meta_graph_util as mgu
from tensorflow.neuron.python import graph_def_util as gdu
from tensorflow.neuron.python import utils
from tensorflow.neuron.python.neuron_cc import list_operators


DEFAULT_MINIMUM_SEGMENT_SIZE = 1


def compile(model_dir, new_model_dir, tags=None, model_feed_dict=None, must_compile=False):
    """Convert a `SavedModel` to a Neuron-optimized `SavedModel`.

    Args:
        model_dir: The path of the original `SavedModel`.
        new_model_dir: The path to which the Neuron-optimized `SavedModel` will be stored.
        tags: A tag or sequence of tags identifying the MetaGraph to load. Optional
            if the SavedModel contains a single MetaGraph, as for those exported from
            `tf.saved_model.save`.

    Returns:
        Dictionary with operator counts before/after optimization, etc.
    """
    if must_compile:
        logging.warning('Enabling must_compile; neuron-cc failures will be thrown as exceptions')
    tfn_args, compiler_args = utils.parse_neuron_cc_flags()

    # load SavedModel
    model = saved_model.load(model_dir, tags=tags)

    # get ConcreteFunction from the SavedModel
    saved_model_proto, debug_info = loader_impl.parse_saved_model_with_debug_info(model_dir)
    signature_def = saved_model_proto.meta_graphs[0].signature_def
    signature_def_key_default = saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    if len(signature_def) == 1:
        signature_def_key = list(signature_def.keys())[0]
    elif signature_def_key_default in signature_def:
        signature_def_key = signature_def_key_default
        logging.warning('Selecting default SignatureDef "{}" when loading SavedModel {}'.format(signature_def_key, model_dir))
    else:
        raise NotImplementedError('Not yet supporting SavedModel {} with multiple signatures')
    model = saved_model.load(model_dir, tags=tags)
    sig_def = signature_def[signature_def_key]
    signature_def_key_prefix = '{}_'.format(signature_def_key)
    for key, value in sig_def.inputs.items():
        if value.name.startswith(signature_def_key_prefix):
            value.name = value.name[len(signature_def_key_prefix):]
    wfunc = model.signatures[signature_def_key]

    # convert all variables to constants
    cfunc = convert_to_constants.convert_variables_to_constants_v2(wfunc)
    graph_def = cfunc.graph.as_graph_def(add_shapes=True)
    original_graph_def = graph_def

    if model_feed_dict is None:
        shape_feed_dict = None
    else:
        shape_feed_dict = {}
        for key, value in model_feed_dict.items():
            input_name = sig_def.inputs[key].name
            if input_name:
                shape_feed_dict[input_name] = value.shape
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
    fuser_param_map['minimum_segment_size'].i = DEFAULT_MINIMUM_SEGMENT_SIZE
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

    # re-wrap GraphDef as a WrappedFunction
    captured_inputs = {ts.ref() for _, ts in wfunc.graph.captures}
    original_inputs = [ts for ts in wfunc.inputs if ts.ref() not in captured_inputs]

    # inputs can be a list of symbolic tensors
    # Note: if using a dictionary (such as `{ts.name: ts.name for ts in original_inputs}`), then
    # the resulted WrappedFunction will occationally get feed tensors going to the wrong inputs
    inputs = [ts.name for ts in original_inputs]

    # outputs need to be a map from output argument name to symbolic tensor name
    # in order to let the WrappedFunction's return dictionary have the correct keys
    tss_name_map = {tss.name: name for name, tss in wfunc.structured_outputs.items()}
    outputs_list = nest.flatten(wfunc.structured_outputs, expand_composites=True)
    outputs = {tss_name_map[tss.name]: ts.name for ts, tss in zip(wfunc.outputs, outputs_list)}

    # wrap GraphDef as a WrappedFunction
    cfunc = wrap_function.function_from_graph_def(graph_def, inputs, outputs)

    # TODO: remove this hack once https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/eager/wrap_function.py#L377 is fixed
    cfunc_input_names = {argdef.name for argdef in cfunc.function_def.signature.input_arg}
    if cfunc_input_names != set(sig_def.inputs.keys()):
        try:
            cfunc._arg_keywords = wfunc._arg_keywords
        except AttributeError:
            pass

    # save the new ConcreteFunction as a new SavedModel
    signatures = {signature_def_key: cfunc}
    saved_model.save(model, new_model_dir, signatures)

    # report compilation result
    num_ops_tfn, num_ops_on_neuron = gdu.compiled_graph_op_counts(graph_def)
    on_neuron_ratio = float(num_ops_on_neuron) / num_ops_tfn if num_ops_tfn != 0 else 0.0
    utils.model_conversion_report(model_dir, new_model_dir, on_neuron_ratio)
    return dict(OnNeuronRatio=on_neuron_ratio)
