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
import tensorflow as tf
import tensorflow.neuron as tfn
from tensorflow.neuron.python.unittest_base import TestV2Only


class TestTraceKerasModel(TestV2Only):

    def test_keras_model_3in_5out_stateful(self):
        input0 = tf.keras.layers.Input(3)
        input1 = tf.keras.layers.Input(3)
        input2 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        dense1 = tf.keras.layers.Dense(3)(input1)
        dense2 = tf.keras.layers.Dense(3)(input2)
        add01 = tf.keras.layers.Add()([dense0, dense1])
        sigmoid01 = tf.keras.layers.Activation('sigmoid')(add01)
        add02 = tf.keras.layers.Add()([dense0, dense2])
        sigmoid02 = tf.keras.layers.Activation('sigmoid')(add02)
        tanh02 = tf.keras.layers.Activation('tanh')(add02)
        inputs = [input0, input1, input2]
        outputs = [sigmoid01, add01, add02, tanh02, sigmoid02]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        input0_tensor = tf.random.uniform([1, 3])
        input1_tensor = tf.random.uniform([1, 3])
        input2_tensor = tf.random.uniform([1, 3])
        model_neuron = tfn.trace(model, [input0_tensor, input1_tensor, input2_tensor])
        _assert_compiler_success_func(model_neuron)
        result_model_ref = model([input0_tensor, input1_tensor, input2_tensor])
        result_model_neuron = model_neuron([input0_tensor, input1_tensor, input2_tensor])
        assert len(result_model_ref) == len(result_model_neuron)
        for res_ref, res_neuron in zip(result_model_ref, result_model_neuron):
            self.assertAllClose(res_ref, res_neuron, rtol=1e-2, atol=1e-2)


def _assert_compiler_success_func(wfunc):
    assert any(op.type == 'NeuronOp' for op in wfunc.graph.get_operations())
