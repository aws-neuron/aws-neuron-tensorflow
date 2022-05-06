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

'''
Defines a new API endpoint to allow for Automatic Multicore Inference
'''

import tensorflow.neuron as tfn
import tensorflow as tf
from tensorflow.neuron.python._trace import AwsNeuronModel, _make_keras_model_savable

class AwsMulticoreNeuronModel(AwsNeuronModel):
    '''
    Decorator class to show that the model has been modified for
    use in automatic multicore inference
    '''
    def __init__(self, func, structured_output):
        super().__init__(func, structured_output)

def multicore(model, example_inputs, num_cores=1):
    """
    Description
    ___________

    Performed on an already optimized AWS-Neuron-optimized model.

    The returned model will be inference only and will be of a new
    type so that the user can identify the modified variable as multicore
    enabled. When loaded, this model will call nrt_load with nc_count.

    Arguments: 
        model: AWS-Neuron-Optimized keras model
        example_inputs: Similar to trace, we require inputs
        num_cores: number of cores to be replicated


    Returns:
        AwsMulticoreNeuronModel   

    """

    if not isinstance(example_inputs, tuple):
        example_inputs = (example_inputs,)

    # TODO: Will this error out if the user loads the model instead of tracing?
    if not isinstance(model, AwsNeuronModel):
        raise ValueError("Invalid model is not AwsNeuronModel")
    func = model.aws_neuron_function
    graph_def = func.graph.as_graph_def()

    # Modify graph def to add a new attribute
    new_nodes = []
    for node in graph_def.node:
        if node.op == tNeuronOp:
            copyNode = deepcopy(node)
            newAttrValue = attr_value_pb2.AttrValue(i=num_cores)
            copyNode.attr['_automatic_multicore'].CopyFrom(newAttrValue)
            new_nodes.append(copyNode)
        else:
            new_nodes.append(node)

    mod_graph_def = graph_pb2.GraphDef()
    mod_graph_def.node.extend(new_nodes)

    cfunc = _wrap_variable_graph_def_as_concrete_function(mod_graph_def, func)
    model = AwsMulticoreNeuronModel(cfunc, func.structured_outputs)

    # TODO: Do we need to refactor this function into utils as well
    _make_keras_model_savable(model, example_inputs)

    return model


