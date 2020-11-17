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


def compile(model_dir, new_model_dir, tags=None, **kwargs):
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
    graph_def = cfunc.graph.as_graph_def()

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
