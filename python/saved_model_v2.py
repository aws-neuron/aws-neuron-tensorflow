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
from tensorflow.python import saved_model
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.neuron.python import graph_def_util as gdu
from tensorflow.neuron.python import utils
from tensorflow.neuron.python._trace import trace


def compile(model_dir, new_model_dir, tags=None, model_feed_dict=None):
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
    logging.warning('tfn.saved_model.compile is not recommended in tensorflow-neuron 2.x; please'
                    ' use tfn.trace instead. Usage can be found by running help(tfn.trace) in'
                    ' interactive Python')
    # load SavedModel
    model = saved_model.load(model_dir, tags=tags)

    # get ConcreteFunction from the SavedModel
    saved_model_proto = parse_saved_model(model_dir)
    signature_def = saved_model_proto.meta_graphs[0].signature_def
    signature_def_key_default = saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    if len(signature_def) == 1:
        signature_def_key = list(signature_def.keys())[0]
    elif signature_def_key_default in signature_def:
        signature_def_key = signature_def_key_default
        logging.warning('Selecting default SignatureDef "{}" when loading SavedModel {}'.format(signature_def_key, model_dir))
    else:
        raise NotImplementedError('Not yet supporting SavedModel {} with multiple signatures')
    sig_def = signature_def[signature_def_key]
    signature_def_key_prefix = '{}_'.format(signature_def_key)
    for key, value in sig_def.inputs.items():
        if value.name.startswith(signature_def_key_prefix):
            value.name = value.name[len(signature_def_key_prefix):]
    wfunc = model.signatures[signature_def_key]

    # generate a map from symbolic tensor name to input argument name
    rev_inputs_map = {value.name: key for key, value in sig_def.inputs.items()}
    example_input_names = [sym_ts.name for sym_ts in wfunc.inputs if sym_ts.name in rev_inputs_map]
    example_inputs = [model_feed_dict[rev_inputs_map[name]] for name in example_input_names]
    cfunc = trace(wfunc, example_inputs).aws_neuron_function

    # save the new ConcreteFunction as a new SavedModel
    signatures = {signature_def_key: cfunc}
    saved_model.save(model, new_model_dir, signatures)

    # report compilation result
    num_ops_tfn, num_ops_on_neuron = gdu.compiled_graph_op_counts(cfunc.graph.as_graph_def())
    on_neuron_ratio = float(num_ops_on_neuron) / num_ops_tfn if num_ops_tfn != 0 else 0.0
    utils.model_conversion_report(model_dir, new_model_dir, on_neuron_ratio)
    return dict(OnNeuronRatio=on_neuron_ratio)
