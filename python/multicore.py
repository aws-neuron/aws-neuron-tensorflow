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

from tensorflow.neuron.python._trace import AwsNeuronModel, \
                                            _make_keras_model_savable, \
                                            _wrap_variable_graph_def_as_concrete_function
from tensorflow.neuron.python.graph_util import tag_multicore


class AwsMulticoreNeuronModel(AwsNeuronModel):
    '''
    Decorator class to show that the model has been modified for
    use in automatic multicore inference
    '''
    def __init__(self, func, structured_output):
        super().__init__(func, structured_output)

def auto_multicore(model, example_inputs, num_cores=1):
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

    if not hasattr(model, 'aws_neuron_function'):
        raise ValueError("Invalid model is not AwsNeuronModel")

    func = model.aws_neuron_function
    graph_def = func.graph.as_graph_def()

    # Modify graph def to add a new attribute
    mod_graph_def = tag_multicore(graph_def, num_cores)

    cfunc = _wrap_variable_graph_def_as_concrete_function(mod_graph_def, func)
    model = AwsMulticoreNeuronModel(cfunc, func.structured_outputs)

    _make_keras_model_savable(model, example_inputs)

    return model


