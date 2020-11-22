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
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import convert_to_constants
import tensorflow.neuron as tfn
from tensorflow.neuron.python.unittest_base import TestV2Only


_RANDOM_SEED = 15213


class TestCompileV1SavedModel(TestV2Only):

    def test_simple(self):
        np.random.seed(_RANDOM_SEED)
        model_dir = './original_saved_model_v1_0'
        new_model_dir = './neuron_saved_model_v1_to_v2_0'
        with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
            input0 = tf.compat.v1.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.compat.v1.placeholder(tf.float16, [None, 2, 2, 3], name='input1')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            conv2d2 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            relu1 = tf.nn.relu(conv2d2, name='relu1')
            inputs = {'x0': input0, 'x1': input1}
            outputs = {'y0': relu0, 'y1': relu1}
            shutil.rmtree(model_dir, ignore_errors=True)
            tf.compat.v1.saved_model.simple_save(sess, export_dir=model_dir, inputs=inputs, outputs=outputs)
        tf.random.set_seed(_RANDOM_SEED)
        feeds = {
            'x0': tf.random.uniform([1, 2, 2, 3], dtype=tf.float16),
            'x1': tf.random.uniform([1, 2, 2, 3], dtype=tf.float16),
        }
        shutil.rmtree(new_model_dir, ignore_errors=True)
        result_compile = tfn.saved_model.compile(model_dir, new_model_dir, model_feed_dict=feeds)
        assert result_compile['OnNeuronRatio'] > 0.05
        model_ref = tf.saved_model.load(model_dir)
        model_neuron = tf.saved_model.load(new_model_dir)
        wfunc_ref = model_ref.signatures['serving_default']
        wfunc_neuron = model_neuron.signatures['serving_default']
        _assert_compiler_success_v2(wfunc_neuron)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_ref = wfunc_ref(**feeds)
            result_neuron = wfunc_neuron(**feeds)
            for name in result_ref.keys():
                np.testing.assert_allclose(result_neuron[name], result_ref[name], rtol=1e-2, atol=1e-2)

    def test_3segments(self):
        np.random.seed(_RANDOM_SEED)
        model_dir = './original_saved_model_v1_0'
        new_model_dir = './neuron_saved_model_v1_to_v2_0'
        with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
            input0 = tf.compat.v1.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.compat.v1.placeholder(tf.float16, [None, 2, 2, 3], name='input1')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            conv2d2 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            relu1 = tf.nn.relu(conv2d2, name='relu1')
            inputs = {'x0': input0, 'x1': input1}
            outputs = {'y0': relu0, 'y1': relu1}
            shutil.rmtree(model_dir, ignore_errors=True)
            tf.compat.v1.saved_model.simple_save(sess, export_dir=model_dir, inputs=inputs, outputs=outputs)
        feeds = {
            'x0': tf.convert_to_tensor(np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16)),
            'x1': tf.convert_to_tensor(np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16)),
        }
        shutil.rmtree(new_model_dir, ignore_errors=True)
        result_compile = tfn.saved_model.compile(
            model_dir, new_model_dir, model_feed_dict=feeds,
            op_whitelist={'Conv2D', 'Const'}, minimum_segment_size=1,
        )
        assert result_compile['OnNeuronRatio'] > 0.05
        model_ref = tf.saved_model.load(model_dir)
        model_neuron = tf.saved_model.load(new_model_dir)
        _assert_compiler_success_v2(model_neuron.signatures['serving_default'])
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_ref = model_ref.signatures['serving_default'](**feeds)
            result_neuron = model_neuron.signatures['serving_default'](**feeds)
            for name in result_ref.keys():
                np.testing.assert_allclose(result_neuron[name], result_ref[name], rtol=1e-2, atol=1e-2)


class TestCompileKerasSavedModel(TestV2Only):

    def test_keras_models_save_model_single_input_single_output(self):
        tf.random.set_seed(_RANDOM_SEED)
        model_dir = './original_saved_model_keras_v2_1'
        new_model_dir = './neuron_saved_model_keras_v2_1'
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(28),
            tf.keras.layers.Dense(10, activation='relu'),
        ])
        shutil.rmtree(model_dir, ignore_errors=True)
        tf.keras.models.save_model(model, model_dir)
        model_ref = tf.saved_model.load(model_dir)
        wfunc_ref = model_ref.signatures['serving_default']
        input_tensor = tf.random.uniform([2, 28])
        feeds = {wfunc_ref.function_def.signature.input_arg[0].name: input_tensor}
        shutil.rmtree(new_model_dir, ignore_errors=True)
        result_compile = tfn.saved_model.compile(model_dir, new_model_dir, model_feed_dict=feeds)
        assert result_compile['OnNeuronRatio'] > 0.05
        model_neuron = tf.saved_model.load(new_model_dir)
        wfunc_neuron = model_neuron.signatures['serving_default']
        _assert_compiler_success_v2(wfunc_neuron)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_model_ref = model_ref(input_tensor)
            result_model_neuron = model_neuron(input_tensor)
            np.testing.assert_allclose(result_model_ref, result_model_neuron, rtol=1e-2, atol=1e-3)
            result_positional_ref = wfunc_ref(input_tensor)
            result_positional_neuron = wfunc_neuron(input_tensor)
            output_key = list(wfunc_ref.structured_outputs.keys())[0]
            np.testing.assert_allclose(result_positional_neuron[output_key], result_positional_ref[output_key], rtol=1e-2, atol=1e-2)
            input_key = list(wfunc_ref.structured_input_signature[1].keys())[0]
            feed_dict = {input_key: input_tensor}
            result_keyword_ref = wfunc_ref(**feed_dict)
            result_keyword_neuron = wfunc_neuron(**feed_dict)
            np.testing.assert_allclose(result_keyword_neuron[output_key], result_keyword_ref[output_key], rtol=1e-2, atol=1e-2)

    def test_keras_save_single_input_single_output(self):
        tf.random.set_seed(_RANDOM_SEED)
        model_dir = './original_saved_model_keras_v2_0'
        new_model_dir = './neuron_saved_model_keras_v2_0'
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(28),
            tf.keras.layers.Dense(10, activation='relu'),
        ])
        shutil.rmtree(model_dir, ignore_errors=True)
        model.save(model_dir)
        model_ref = tf.saved_model.load(model_dir)
        wfunc_ref = model_ref.signatures['serving_default']
        input_tensor = tf.random.uniform([2, 28])
        feeds = {wfunc_ref.function_def.signature.input_arg[0].name: input_tensor}
        shutil.rmtree(new_model_dir, ignore_errors=True)
        result_compile = tfn.saved_model.compile(model_dir, new_model_dir, model_feed_dict=feeds)
        assert result_compile['OnNeuronRatio'] > 0.05
        model_neuron = tf.saved_model.load(new_model_dir)
        wfunc_neuron = model_neuron.signatures['serving_default']
        _assert_compiler_success_v2(wfunc_neuron)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_model_ref = model_ref(input_tensor)
            result_model_neuron = model_neuron(input_tensor)
            np.testing.assert_allclose(result_model_ref, result_model_neuron, rtol=1e-2, atol=1e-2)
            result_positional_ref = wfunc_ref(input_tensor)
            result_positional_neuron = wfunc_neuron(input_tensor)
            output_key = list(wfunc_ref.structured_outputs.keys())[0]
            np.testing.assert_allclose(result_positional_neuron[output_key], result_positional_ref[output_key], rtol=1e-2, atol=1e-2)
            input_key = list(wfunc_ref.structured_input_signature[1].keys())[0]
            feed_dict = {input_key: input_tensor}
            result_keyword_ref = wfunc_ref(**feed_dict)
            result_keyword_neuron = wfunc_neuron(**feed_dict)
            np.testing.assert_allclose(result_keyword_neuron[output_key], result_keyword_ref[output_key], rtol=1e-2, atol=1e-2)

    def test_keras_models_save_model_3in_5out_stateful(self):
        tf.random.set_seed(_RANDOM_SEED)
        model_dir = './original_saved_model_keras_v2_2'
        new_model_dir = './neuron_saved_model_keras_v2_2'
        input0 = tf.keras.layers.Input(28)
        input1 = tf.keras.layers.Input(28)
        input2 = tf.keras.layers.Input(28)
        dense0 = tf.keras.layers.Dense(28)(input0)
        dense1 = tf.keras.layers.Dense(28)(input1)
        dense2 = tf.keras.layers.Dense(28)(input2)
        add01 = tf.keras.layers.Add()([dense0, dense1])
        sigmoid01 = tf.keras.layers.Activation('sigmoid')(add01)
        add02 = tf.keras.layers.Add()([dense0, dense2])
        sigmoid02 = tf.keras.layers.Activation('sigmoid')(add02)
        tanh02 = tf.keras.layers.Activation('tanh')(add02)
        inputs = [input0, input1, input2]
        outputs = [sigmoid01, add01, add02, tanh02, sigmoid02]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        shutil.rmtree(model_dir, ignore_errors=True)
        tf.keras.models.save_model(model, model_dir)
        model_ref = tf.saved_model.load(model_dir)
        wfunc_ref = model_ref.signatures['serving_default']
        input0_tensor = tf.random.uniform([2, 28])
        input1_tensor = tf.random.uniform([2, 28])
        input2_tensor = tf.random.uniform([2, 28])
        feed_dict = {
            wfunc_ref.function_def.signature.input_arg[0].name: input0_tensor,
            wfunc_ref.function_def.signature.input_arg[1].name: input1_tensor,
            wfunc_ref.function_def.signature.input_arg[2].name: input2_tensor,
        }
        shutil.rmtree(new_model_dir, ignore_errors=True)
        result_compile = tfn.saved_model.compile(model_dir, new_model_dir, model_feed_dict=feed_dict)
        assert result_compile['OnNeuronRatio'] > 0.05
        model_neuron = tf.saved_model.load(new_model_dir)
        wfunc_neuron = model_neuron.signatures['serving_default']
        _assert_compiler_success_v2(wfunc_neuron)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_model_ref = model_ref([input0_tensor, input1_tensor, input2_tensor])
            result_model_neuron = model_neuron([input0_tensor, input1_tensor, input2_tensor])
            assert len(result_model_ref) == len(result_model_neuron)
            for res_ref, res_neuron in zip(result_model_ref, result_model_neuron):
                np.testing.assert_allclose(res_ref, res_neuron, rtol=1e-2, atol=1e-2)
            result_keyword_ref = wfunc_ref(**feed_dict)
            result_keyword_neuron = wfunc_neuron(**feed_dict)
            for output_key in result_keyword_ref:
                np.testing.assert_allclose(result_keyword_neuron[output_key], result_keyword_ref[output_key], rtol=1e-2, atol=1e-2)

    def test_keras_models_save_model_3in_5out_stateless(self):
        tf.random.set_seed(_RANDOM_SEED)
        model_dir = './original_saved_model_keras_v2_3'
        new_model_dir = './neuron_saved_model_keras_v2_3'
        input0 = tf.keras.layers.Input(28)
        input1 = tf.keras.layers.Input(28)
        input2 = tf.keras.layers.Input(28)
        add01 = tf.keras.layers.Add()([input0, input1])
        sigmoid01 = tf.keras.layers.Activation('sigmoid')(add01)
        add02 = tf.keras.layers.Add()([input0, input2])
        sigmoid02 = tf.keras.layers.Activation('sigmoid')(add02)
        tanh02 = tf.keras.layers.Activation('tanh')(add02)
        inputs = [input0, input1, input2]
        outputs = [sigmoid01, add01, add02, tanh02, sigmoid02]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        shutil.rmtree(model_dir, ignore_errors=True)
        tf.keras.models.save_model(model, model_dir)
        model_ref = tf.saved_model.load(model_dir)
        wfunc_ref = model_ref.signatures['serving_default']
        input0_tensor = tf.random.uniform([2, 28])
        input1_tensor = tf.random.uniform([2, 28])
        input2_tensor = tf.random.uniform([2, 28])
        feed_dict = {
            wfunc_ref.function_def.signature.input_arg[0].name: input0_tensor,
            wfunc_ref.function_def.signature.input_arg[1].name: input1_tensor,
            wfunc_ref.function_def.signature.input_arg[2].name: input2_tensor,
        }
        shutil.rmtree(new_model_dir, ignore_errors=True)
        result_compile = tfn.saved_model.compile(model_dir, new_model_dir, model_feed_dict=feed_dict)
        assert result_compile['OnNeuronRatio'] > 0.05
        model_neuron = tf.saved_model.load(new_model_dir)
        wfunc_neuron = model_neuron.signatures['serving_default']
        _assert_compiler_success_v2(wfunc_neuron)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_model_ref = model_ref([input0_tensor, input1_tensor, input2_tensor])
            result_model_neuron = model_neuron([input0_tensor, input1_tensor, input2_tensor])
            assert len(result_model_ref) == len(result_model_neuron)
            for res_ref, res_neuron in zip(result_model_ref, result_model_neuron):
                np.testing.assert_allclose(res_ref, res_neuron, rtol=1e-2, atol=1e-3)
            result_keyword_ref = wfunc_ref(**feed_dict)
            result_keyword_neuron = wfunc_neuron(**feed_dict)
            for output_key in result_keyword_ref:
                np.testing.assert_allclose(result_keyword_neuron[output_key], result_keyword_ref[output_key], rtol=1e-2, atol=1e-2)


def _assert_compiler_success_v2(wfunc):
    cfunc = convert_to_constants.convert_variables_to_constants_v2(wfunc)
    # TODO: don't know how to extract the underlying graph with NeuronOps yet... for now
    # asserting convert_variables_to_constants_v2 cannot unroll StatefulPartitionedCall
    # due to that the grappler function optimizer doesn't recognize NeuronOp
    assert any(op.type == 'StatefulPartitionedCall' for op in cfunc.graph.get_operations())
