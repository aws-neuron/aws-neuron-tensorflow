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
from tensorflow.neuron.python import meta_graph_util as mgu
from tensorflow.neuron.python import graph_def_util as gdu


def compile(model_dir, new_model_dir, tags=None, model_feed_dict=None,
            minimum_segment_size=2, op_whitelist=None, no_fuse_ops=None, force_fuse_ops=None,
            **kwargs):
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
        shape_feed_dict = {sig_def.inputs[key].name: value.shape for key, value in model_feed_dict.items()}
    graph_def = gdu.encode_inferred_shapes(graph_def, shape_feed_dict)

    # produce GraphDef by running grappler passes
    meta_graph_def = meta_graph_pb2.MetaGraphDef(graph_def=graph_def)
    sig_def_grappler = mgu.build_signature_def(cfunc.inputs, cfunc.outputs)
    meta_graph_def.signature_def['serving_default'].CopyFrom(sig_def_grappler)
    opt_config = config_pb2.ConfigProto()
    rewriter_config = opt_config.graph_options.rewrite_options
    rewriter_config.meta_optimizer_iterations = 1
    rewriter_config.optimizers.append('constfold')
    rewriter_config.optimizers.append('aws_neuron_static_shape_inference')

    # configure operator fusion
    fuser_config = rewriter_config.custom_optimizers.add()
    fuser_config.name = 'aws_neuron_fuse_supported_operators'
    fuser_param_map = fuser_config.parameter_map
    fuser_param_map['minimum_segment_size'].i = minimum_segment_size
    if op_whitelist is None:
        op_whitelist = set()
        try:
            from neuroncc.driver.commands.ListOperatorsCommand import ListOperatorsCommand
            op_whitelist.update(ListOperatorsCommand(parent_command=None).known_frameworks['TENSORFLOW'].listOperators())
            op_whitelist.discard('Placeholder')
            op_whitelist.discard('IdentityN')
        except ImportError:
            logging.warning('neuron-cc is not installed. Please check neuron-cc '
                            'installation, or reinstall by "pip install --force neuron-cc".')
    fuser_param_map['op_whitelist'].list.s.extend(item.encode() for item in op_whitelist)
    if no_fuse_ops is not None:
        fuser_param_map['no_fuse_ops'].list.s.extend(item.encode() for item in no_fuse_ops)
    if force_fuse_ops is not None:
        fuser_param_map['force_fuse_ops'].list.s.extend(item.encode() for item in force_fuse_ops)

    # call all grappler passes
    graph_def = tf_optimizer.OptimizeGraph(opt_config, meta_graph_def)

    # call graph_def_util passes
    graph_def = gdu.restore_compiler_failures(graph_def, original_graph_def)

    # re-wrap GraphDef as a WrappedFunction
    captured_inputs = {ts.ref() for _, ts in wfunc.graph.captures}
    original_inputs = [ts for ts in wfunc.inputs if ts.ref() not in captured_inputs]

    # inputs can be a list of symbolic tensors
    # Note: if using a dictionary (such as `{ts.name: ts.name for ts in original_inputs}`), then
    # the resulted WrappedFunction will occationally get feed tensors going to the wrong inputs
    inputs = [ts.name for ts in original_inputs]

    # outputs need to be a map from output argument name to symbolic tensor name
    # in order to let the WrappedFunction's return dictionary have the correct keys
    outputs = {}
    for name, tensor_info in sig_def.outputs.items():
        output_tensor = wfunc.graph.get_tensor_by_name(tensor_info.name)
        if output_tensor.op.type in {'PartitionedCall', 'StatefulPartitionedCall'}:
            identity_op = output_tensor.consumers()[0]
            output_tensor = identity_op.outputs[0]
        outputs[name] = output_tensor.name

    # wrap GraphDef as a WrappedFunction
    cfunc = wrap_function.function_from_graph_def(graph_def, inputs, outputs)

    # TODO: remove this hack once https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/eager/wrap_function.py#L377 is fixed
    cfunc_input_names = {argdef.name for argdef in cfunc.function_def.signature.input_arg}
    if cfunc_input_names != set(sig_def.inputs.keys()):
        cfunc._arg_keywords = wfunc._arg_keywords

    # save the new ConcreteFunction as a new SavedModel
    signatures = {signature_def_key: cfunc}
    saved_model.save(model, new_model_dir, signatures)

    # report compilation result
    return dict(OnNeuronRatio=0.0)
