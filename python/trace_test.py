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
from unittest.mock import patch
import tensorflow as tf
from tensorflow.python.eager import wrap_function
import tensorflow.neuron as tfn
from tensorflow.neuron.python.unittest_base import TestV2Only, xfail_for_versions


class TestTraceKerasModel(TestV2Only):

    def test_keras_model_3in_5out_stateful(self):
        model = self._model_3in_5out()
        input0_tensor = tf.random.uniform([1, 3])
        input1_tensor = tf.random.uniform([1, 3])
        input2_tensor = tf.random.uniform([1, 3])
        model_neuron = tfn.trace(model, [input0_tensor, input1_tensor, input2_tensor])
        _assert_compiler_success_func(model_neuron.aws_neuron_function)
        result_model_ref = model([input0_tensor, input1_tensor, input2_tensor])
        result_model_neuron = model_neuron([input0_tensor, input1_tensor, input2_tensor])
        assert len(result_model_ref) == len(result_model_neuron)
        for res_ref, res_neuron in zip(result_model_ref, result_model_neuron):
            self.assertAllClose(res_ref, res_neuron, rtol=1e-2, atol=1e-2)

    def test_keras_model_3in_5out_dynamic_batch_size(self):
        model = self._model_3in_5out()
        input0_tensor = tf.random.uniform([3, 3])
        input1_tensor = tf.random.uniform([3, 3])
        input2_tensor = tf.random.uniform([3, 3])
        model_neuron = tfn.trace(model, [input0_tensor, input1_tensor, input2_tensor])
        _assert_compiler_success_func(model_neuron.aws_neuron_function)
        input0_tensor = tf.random.uniform([5, 3])
        input1_tensor = tf.random.uniform([5, 3])
        input2_tensor = tf.random.uniform([5, 3])
        result_model_ref = model([input0_tensor, input1_tensor, input2_tensor])
        result_model_neuron = model_neuron([input0_tensor, input1_tensor, input2_tensor])
        assert len(result_model_ref) == len(result_model_neuron)
        for res_ref, res_neuron in zip(result_model_ref, result_model_neuron):
            self.assertAllClose(res_ref, res_neuron, rtol=1e-2, atol=1e-2)

    def _model_3in_5out(self):
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
        return model

    def test_keras_model_1in_1out_save(self):
        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        inputs = [input0]
        outputs = [dense0]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        input0_tensor = tf.random.uniform([1, 3])
        model_neuron = tfn.trace(model, input0_tensor)
        model_dir = os.path.join(self.get_temp_dir(), 'neuron_keras_model_1in_1out_save')
        model_neuron.save(model_dir)
        _assert_compiler_success_func(model_neuron.aws_neuron_function)
        result_model_ref = model(input0_tensor)
        result_model_neuron = model_neuron(input0_tensor)
        assert len(result_model_ref) == len(result_model_neuron)
        for res_ref, res_neuron in zip(result_model_ref, result_model_neuron):
            self.assertAllClose(res_ref, res_neuron, rtol=1e-2, atol=1e-2)


class TestTraceFunction(TestV2Only):

    def test_func_1conv_save(self):
        kernel = tf.random.uniform([3, 3, 3, 32])

        def func(tensor):
            return tf.nn.conv2d(tensor, kernel, padding='VALID', strides=[1, 1, 1, 1])

        input_tensor = tf.random.uniform([1, 28, 28, 3])
        func_neuron = tfn.trace(func, input_tensor)
        model_dir = os.path.join(self.get_temp_dir(), 'neuron_func_1conv_save')
        func_neuron.save(model_dir)
        _assert_compiler_success_func(func_neuron.aws_neuron_function)
        result_func_ref = func(input_tensor)
        result_func_neuron = func_neuron(input_tensor)
        func_neuron_reloaded = tf.keras.models.load_model(model_dir)
        result_func_neuron_reloaded = func_neuron_reloaded(input_tensor)
        self.assertAllClose(result_func_neuron, result_func_ref, rtol=1e-2, atol=1e-2)
        self.assertAllClose(result_func_neuron_reloaded, result_func_ref, rtol=1e-2, atol=1e-2)

    def test_func_input_list_len_1_save(self):
        kernel = tf.random.uniform([3, 3, 3, 32])

        def func(tensor_list):
            tensor, = tensor_list
            return tf.nn.conv2d(tensor, kernel, padding='VALID', strides=[1, 1, 1, 1])

        input_tensor = tf.random.uniform([1, 28, 28, 3])
        func_neuron = tfn.trace(func, [input_tensor])
        model_dir = os.path.join(self.get_temp_dir(), 'neuron_func_input_list_len_1_save')
        func_neuron.save(model_dir)
        _assert_compiler_success_func(func_neuron.aws_neuron_function)
        result_func_ref = func([input_tensor])
        result_func_neuron = func_neuron([input_tensor])
        self.assertAllClose(result_func_neuron, result_func_ref, rtol=1e-2, atol=1e-2)

    def test_func_1conv_with_shuffle(self):
        kernel = tf.random.uniform([3, 3, 3, 6])

        def conv2d_nchw(tensor):
            tensor = tf.transpose(tensor, [0, 2, 3, 1])
            tensor = tf.nn.conv2d(tensor, kernel, padding='VALID', strides=[1, 1, 1, 1])
            return tf.transpose(tensor, [0, 3, 1, 2])

        def func_ref(tensor):
            tensor = tf.transpose(tensor, [0, 3, 1, 2])
            return conv2d_nchw(tensor)

        def func(tensor):
            return conv2d_nchw(tensor)

        input_tensor = tf.random.uniform([1, 4, 4, 3])
        input_tensor_tracing = tf.transpose(input_tensor, [0, 3, 1, 2])
        func_neuron = tfn.trace(func, input_tensor_tracing)
        cfunc = func_neuron.aws_neuron_function
        graph_def = cfunc.graph.as_graph_def()
        for node in graph_def.node:
            if node.op == 'NeuronOp':
                idx_ts = node.attr['_input_shuffles'].list.tensor.add()
                indices = tf.range(input_tensor.shape.num_elements())
                indices = tf.reshape(indices, input_tensor.shape)
                indices_t = tf.transpose(indices, [0, 3, 1, 2])
                indices_t = tf.reshape(indices_t, [-1])
                idx_ts.int64_val.extend(indices_t.numpy())
        input_names = [ts.name for ts in cfunc.inputs]
        output_names = cfunc.outputs[0].name
        cfunc = wrap_function.function_from_graph_def(graph_def, input_names, output_names)
        _assert_compiler_success_func(cfunc)
        result_func_ref = func_ref(input_tensor)
        result_func_neuron = cfunc(tf.reshape(input_tensor, input_tensor_tracing.shape))
        self.assertAllClose(result_func_neuron, result_func_ref, rtol=1e-2, atol=1e-2)

    def test_func_pad_conv(self):
        kernel = tf.random.uniform([7, 7, 3, 64])
        kernel = tf.cast(kernel, tf.float16)

        def func_ref(tensor):
            tensor = tf.pad(tensor, [[0, 0], [0, 0], [3, 3], [3, 3]])
            tensor = tf.transpose(tensor, [0, 2, 3, 1])
            tensor = tf.nn.conv2d(tensor, kernel, padding='VALID', strides=[1, 2, 2, 1])
            return tf.transpose(tensor, [0, 3, 1, 2])

        def func(tensor):
            tensor = tf.pad(tensor, [[0, 0], [0, 0], [3, 3], [3, 3]])
            return tf.nn.conv2d(tensor, kernel, padding='VALID', strides=[1, 1, 2, 2], data_format='NCHW')

        input_tensor = tf.random.uniform([1, 3, 224, 224])
        input_tensor = tf.cast(input_tensor, tf.float16)
        func_neuron = tfn.trace(func, input_tensor)
        compiled_func = func_neuron.aws_neuron_function
        _assert_compiler_success_func(compiled_func)
        neuron_op = [op for op in compiled_func.graph.get_operations() if op.type == 'NeuronOp'][0]
        neff_size = len(neuron_op.get_attr('executable'))
        assert neff_size < 2e6, 'neff too large -- replication is probably not working'
        result_func_ref = func_ref(input_tensor)
        result_func_neuron = func_neuron(input_tensor)
        self.assertAllClose(result_func_neuron, result_func_ref, rtol=1e-3, atol=1e-5)

        # test dynamic batch size inference
        input_tensor = tf.random.uniform([16, 3, 224, 224])
        input_tensor = tf.cast(input_tensor, tf.float16)
        result_func_ref = func_ref(input_tensor)
        result_func_neuron = func_neuron(input_tensor)
        self.assertAllClose(result_func_neuron, result_func_ref, rtol=1e-3, atol=1e-5)

    def test_func_rtr_conv_multiple_consumers(self):
        kernel0 = tf.random.uniform([7, 7, 3, 64])
        kernel0 = tf.cast(kernel0, tf.float16)
        kernel1 = tf.random.uniform([7, 7, 3, 64])
        kernel1 = tf.cast(kernel1, tf.float16)

        def func(tensor):
            conv0 = tf.nn.conv2d(tensor, kernel0, padding='VALID', strides=[1, 2, 2, 1], data_format='NHWC')
            conv1 = tf.nn.conv2d(tensor, kernel1, padding='VALID', strides=[1, 2, 2, 1], data_format='NHWC')
            conv2 = tf.nn.conv2d(tensor, kernel1, padding='VALID', strides=[1, 2, 2, 1], data_format='NHWC')
            return conv0 + conv1 + conv2

        input_tensor = tf.random.uniform([1, 224, 224, 3])
        input_tensor = tf.cast(input_tensor, tf.float16)
        func_neuron = tfn.trace(func, input_tensor)
        compiled_func = func_neuron.aws_neuron_function
        _assert_compiler_success_func(compiled_func)
        input_shuffles = compiled_func.graph.get_operations()[1].get_attr('_input_shuffles')
        assert any(item is not None for item in input_shuffles), 'no input_shuffles'
        neuron_op = [op for op in compiled_func.graph.get_operations() if op.type == 'NeuronOp'][0]
        neff_size = len(neuron_op.get_attr('executable'))
        assert neff_size < 5e6, 'neff too large -- replication is probably not working'
        result_func_ref = func(input_tensor)
        result_func_neuron = func_neuron(input_tensor)
        self.assertAllClose(result_func_neuron, result_func_ref, rtol=1e-2, atol=1e-5)

    def test_func_3in_5out(self):

        def func(tensor0, tensor1, tensor2):
            tensor01 = tensor0 + tensor1
            tensor12 = tensor1 + tensor2
            relu01 = tf.nn.relu(tensor01)
            relu12 = tf.nn.relu(tensor12)
            sigmoid01 = tf.nn.relu(tensor01)
            sigmoid12 = tf.nn.relu(tensor12)
            tensor0112 = tensor01 + tensor12
            return relu01, relu12, sigmoid01, sigmoid12, tensor0112

        input_tensor0 = tf.random.uniform([1, 8, 6])
        input_tensor1 = tf.random.uniform([1, 8, 6])
        input_tensor2 = tf.random.uniform([1, 8, 6])
        func_neuron = tfn.trace(func, (input_tensor0, input_tensor1, input_tensor2))
        _assert_compiler_success_func(func_neuron.aws_neuron_function)
        result_func_ref = func(input_tensor0, input_tensor1, input_tensor2)
        result_func_neuron = func_neuron(input_tensor0, input_tensor1, input_tensor2)
        assert len(result_func_neuron) == len(result_func_ref)
        for res_neuron, res_ref in zip(result_func_neuron, result_func_ref):
            self.assertAllClose(res_neuron, res_ref, rtol=1e-2, atol=1e-2)

    def test_func_list_tensor_in_5out(self):

        def func(tensor_list0, tensor2):
            tensor0, tensor1 = tensor_list0
            tensor01 = tensor0 + tensor1
            tensor12 = tensor1 + tensor2
            relu01 = tf.nn.relu(tensor01)
            relu12 = tf.nn.relu(tensor12)
            sigmoid01 = tf.nn.relu(tensor01)
            sigmoid12 = tf.nn.relu(tensor12)
            tensor0112 = tensor01 + tensor12
            return relu01, relu12, sigmoid01, sigmoid12, tensor0112

        input_tensor0 = tf.random.uniform([1, 8, 6])
        input_tensor1 = tf.random.uniform([1, 8, 6])
        input_tensor2 = tf.random.uniform([1, 8, 6])
        func_neuron = tfn.trace(func, ([input_tensor0, input_tensor1], input_tensor2))
        # TODO: cannot save for now
        # func_neuron.save('neuron_func_list_tensor_in_5out_save')
        _assert_compiler_success_func(func_neuron.aws_neuron_function)
        result_func_ref = func([input_tensor0, input_tensor1], input_tensor2)
        result_func_neuron = func_neuron([input_tensor0, input_tensor1], input_tensor2)
        assert len(result_func_neuron) == len(result_func_ref)
        for res_neuron, res_ref in zip(result_func_neuron, result_func_ref):
            self.assertAllClose(res_neuron, res_ref, rtol=1e-2, atol=1e-2)

    def test_attention_layer(self):
        query = tf.random.uniform([1, 8, 32])
        key = tf.random.uniform([1, 16, 32])
        value = tf.random.uniform([1, 16, 32])
        query_mask = tf.random.uniform([1, 8]) > 0.5
        value_mask = tf.random.uniform([1, 16]) > 0.5
        layer = tf.keras.layers.Attention(use_scale=False)
        example_inputs = [query, key, value], [query_mask, value_mask]

        def do_nothing(graph_def, *args, **kwargs):
            return graph_def

        with patch('tensorflow.neuron.python.graph_def_util.run_compiler_on_subgraphs', do_nothing):
            layer_neuron = tfn.trace(layer, example_inputs)

        result_layer = layer(*example_inputs)
        result_layer_neuron = layer_neuron(*example_inputs)
        self.assertAllClose(result_layer_neuron, result_layer, rtol=1e-2, atol=1e-2)

    def test_prune_subgraphs(self):

        def func(tensor):
            for _ in range(100):
                tensor = tf.nn.relu(tensor)
            tensor = tf.nn.tanh(tensor)
            tensor = tf.nn.relu(tensor)
            tensor = tf.nn.relu(tensor)
            return tf.nn.relu(tensor)

        def fake_list_operators():
            return {'Relu'}

        input_tensor = tf.random.uniform([1, 1])
        with patch('tensorflow.neuron._trace.list_operators', fake_list_operators):
            func_neuron = tfn.trace(func, input_tensor)
        compiled_func = func_neuron.aws_neuron_function
        op_list = compiled_func.graph.get_operations()
        assert len(op_list) == 7
        assert len([op for op in op_list if op.type == 'NeuronOp']) == 1, 'found multiple NeuronOps'


def _assert_compiler_success_func(wfunc):
    assert any(op.type == 'NeuronOp' for op in wfunc.graph.get_operations())
