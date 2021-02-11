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
import unittest
import numpy as np
import tensorflow as tf  # supposed to be tf2
import tensorflow.neuron as tfn
import random
from itertools import product
import shutil
from tensorflow.neuron.python.unittest_base import TestV2Only, xfail_for_versions

# each number represents the number of random
# parameters from that catergory
# that will be used in testing

NUM_ACTIVATIONS = 1  # Max is 6
NUM_INPUT_UNITS = 1  # Max is 1024 (would not use more than 10)
NUM_OUTPUT_UNITS = 1  # Max is 1024 (would not use more than 10)
NUM_MAGIC_NUMBERS = 3  # Max is 10
NUM_KERNEL_SIZES = 1  # Max is 2
NUM_POWERS = 2  # Max is 11


# here are the parameter lists
inputNumUnits = list(range(1, 1025))
outputNumUnits = list(range(1, 1025))
magicNumbers = [28, 14, 7, 224, 112, 56, 28, 299, 150, 75]
kernelSizes = [1, 3]
_ACTIVATIONS = ['softmax', 'relu', 'tanh', 'sigmoid', 'exponential', 'linear']
powersOfTwo = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


# randomize the order
random.seed(5634482)
random.shuffle(inputNumUnits)
random.shuffle(outputNumUnits)
random.shuffle(magicNumbers)
random.shuffle(kernelSizes)
random.shuffle(_ACTIVATIONS)
random.shuffle(powersOfTwo)

# Pick the first n params based on specified constants above
inputNumUnits = inputNumUnits[0:NUM_INPUT_UNITS]
outputNumUnits = outputNumUnits[0:NUM_OUTPUT_UNITS]
magicNumbers = magicNumbers[0:NUM_MAGIC_NUMBERS]
kernelSizes = kernelSizes[0:NUM_KERNEL_SIZES]
_ACTIVATIONS = _ACTIVATIONS[0:NUM_ACTIVATIONS]
powersOfTwo = powersOfTwo[0:NUM_POWERS]


class TestSequentialKeras(TestV2Only):
    # This function tests a basic NN with two dense layers.
    # It has 3 paramaters which vary.
    # 1. Number of input units
    # 2. Number of output units
    # 3. The type of activation function that the input layer uses
    # @parameterized.expand()

    def test_flatten_dense_dropout(self):

        param_list = list(product(inputNumUnits, _ACTIVATIONS, outputNumUnits))
        for inu, a, onu in param_list:
            # subTest allows us to generate tests dynamically
            # if one of the subTests fail, the error message
            # along with the inputs (inu a onu) will be displayed.
            # however this will still show up as 1 test even though
            # there can be many subTests

            with self.subTest(inputNumUnits=inu, activations=a, outputNumUnits=onu):
                model = tf.keras.models.Sequential(
                    [
                        # tf.keras.layers is tf2 syntax
                        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
                        tf.keras.layers.Dense(inu, activation=a),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(onu),
                    ]
                )

                model_dir = './keras_flatten_dense_dropout'

                test_input = tf.random.uniform((1, 28, 28))
                trace_compile_and_infer(model, test_input)
                feed_dict = tf2_compile(model, model_dir, example_inputs=[test_input])
                run_inference(model_dir, [test_input], feed_dict)

    def test_conv2d_conv2d_flatten_dense(self):

        param_list = list(product(inputNumUnits, _ACTIVATIONS, outputNumUnits, kernelSizes))
        for inu, a, onu, ks in param_list:
            # subTest allows us to generate tests dynamically
            # if one of the subTests fail, the error message
            # along with the inputs (inu a onu) will be displayed.
            # however this will still show up as 1 test even though
            # there can be many subTests

            with self.subTest(inputNumUnits=inu, activations=a, outputNumUnits=onu, kernelSizes=ks):
                model = tf.keras.models.Sequential(
                    [
                        # tf.keras.layers is tf2 syntax
                        tf.keras.layers.Conv2D(inu, kernel_size=ks, activation=a, input_shape=(28, 28, 1)),
                        tf.keras.layers.Conv2D(inu, kernel_size=ks, activation=a),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(onu),
                    ]
                )

                # Export SavedModel
                model_dir = './keras_conv2d_conv2d_flatten_dense'

                test_input = tf.random.uniform((1, 28, 28, 1))

                feed_dict = tf2_compile(model, model_dir, example_inputs=[test_input])
                run_inference(model_dir, [test_input], feed_dict)
                trace_compile_and_infer(model, test_input)

    #tensorflow.python.framework.errors_impl.InvalidArgumentError: The graph couldn't be sorted in topological order.
    #internal compiler error
    @unittest.expectedFailure
    def test_lstm_lstm_dense_dense(self):
        param_list = list(product(inputNumUnits, _ACTIVATIONS, outputNumUnits))
        for inu, a, onu in param_list:
            with self.subTest(inputNumUnits=inu, activations=a, outputNumUnits=onu):
                model = tf.keras.models.Sequential(
                    [
                        # tf.keras.layers is tf2 syntax
                        tf.keras.layers.LSTM(inu, activation=a, input_shape=(28, 28), return_sequences=True),
                        tf.keras.layers.LSTM(inu, activation=a),
                        tf.keras.layers.Dense(onu, activation=a),
                        tf.keras.layers.Dense(10, activation=a),
                    ]
                )

                # Export SavedModel
                model_dir = './keras_lstm_lstm_dense_dense'

                test_input = tf.random.uniform((1, 28, 28))
                trace_compile_and_infer(model, test_input)

                feed_dict = tf2_compile(model, model_dir, example_inputs=[test_input])
                run_inference(model_dir, [test_input], feed_dict)

    def test_maxpool2d(self):
        # A simple test that is only parameterized by inputNumUnits
        # which in this case describes the size of the square input

        param_list = list(inputNumUnits)
        for inu in param_list:
            # subTest allows us to generate tests dynamically
            # if one of the subTests fail, the error message
            # along with the inputs (inu a onu) will be displayed.
            # however this will still show up as 1 test even though
            # there can be many subTests

            with self.subTest(inputNumUnits=inu):
                model = tf.keras.models.Sequential(
                    [
                        # tf.keras.layers is tf2 syntax
                        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same', input_shape=(inu, inu, 1))
                    ]
                )

                # Export SavedModel
                model_dir = './keras_maxpool2d'

                test_input = tf.random.uniform((1, inu, inu, 1))
                trace_compile_and_infer(model, test_input)

                feed_dict = tf2_compile(model, model_dir, example_inputs=[test_input])
                run_inference(model_dir, [test_input], feed_dict)

class TestFunctionalKeras(TestV2Only):

    @xfail_for_versions('2.1', '2.2')
    def test_toy_resnet(self):
        inputs = tf.keras.Input(shape=(32, 32, 3), name="img")
        x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
        block_1_output = tf.keras.layers.MaxPooling2D(3)(x)

        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        block_2_output = tf.keras.layers.add([x, block_1_output])

        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        block_3_output = tf.keras.layers.add([x, block_2_output])

        x = tf.keras.layers.Conv2D(64, 3, activation="relu")(block_3_output)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(10)(x)

        model = tf.keras.Model(inputs, outputs, name="toy_resnet")
        model_dir = './keras_toy_resnet'

        test_input = tf.random.uniform((1, 32, 32, 3))
        trace_compile_and_infer(model, test_input)

        feed_dict = tf2_compile(model, model_dir, example_inputs=[test_input])
        run_inference(model_dir, [test_input], feed_dict)

    @unittest.expectedFailure
    def test_multiple_io(self):
        num_tags = 12  # Number of unique issue tags
        num_words = 10000  # Size of vocabulary obtained when preprocessing text data
        num_departments = 4  # Number of departments for predictions

        title_input = tf.keras.Input(shape=(None,), name="title")  # Variable-length sequence of ints
        body_input = tf.keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
        tags_input = tf.keras.Input(shape=(num_tags,), name="tags")  # Binary vectors of size `num_tags`

        # Embed each word in the title into a 64-dimensional vector
        title_features = tf.keras.layers.Embedding(num_words, 64)(title_input)
        # Embed each word in the text into a 64-dimensional vector
        body_features = tf.keras.layers.Embedding(num_words, 64)(body_input)

        # Reduce sequence of embedded words in the title into a single 128-dimensional vector
        title_features = tf.keras.layers.LSTM(128)(title_features)
        # Reduce sequence of embedded words in the body into a single 32-dimensional vector
        body_features = tf.keras.layers.LSTM(32)(body_features)

        # Merge all available features into a single large vector via concatenation
        x = tf.keras.layers.concatenate([title_features, body_features, tags_input])

        # Stick a logistic regression for priority prediction on top of the features
        priority_pred = tf.keras.layers.Dense(1, name="priority")(x)
        # Stick a department classifier on top of the features
        department_pred = tf.keras.layers.Dense(num_departments, name="department")(x)

        # Instantiate an end-to-end model predicting both priority and department
        model = tf.keras.Model(
            inputs=[title_input, body_input, tags_input],
            outputs=[priority_pred, department_pred],
        )

        model_dir = './keras_multiple_io'
        shutil.rmtree(model_dir, ignore_errors=True)


        # Dummy input data
        title_data = tf.random.uniform(shape=(1280,10) ,maxval=num_words, dtype=tf.int64 )
        body_data = tf.random.uniform(shape=(1280,10) ,maxval=num_words, dtype=tf.int64)
        tags_data = tf.random.uniform(shape=(1280, num_tags) ,maxval=2)

        trace_compile_and_infer(model, [title_data, body_data, tags_data])

        feed_dict = tf2_compile(model, model_dir, example_inputs=[title_data, body_data, tags_data])
        run_inference(model_dir, [title_data, body_data, tags_data], feed_dict)

class TestGraphUtil(TestV2Only):

    def test_multiple_io(self):
        input1 = tf.keras.Input(shape=[2, 2, 3], name='input1')
        input2 = tf.keras.Input(shape=[2, 2, 3], name='input2')
        conv2d1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=1, padding='VALID', name='conv2d1')(input1)
        conv2d2 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=1, padding='VALID', name='conv2d2')(input2)
        added = tf.keras.layers.Add(name='add')([conv2d1, conv2d2])
        relu1 = tf.keras.layers.Activation('relu', name='relu1')(added)
        sigmoid1 = tf.keras.layers.Activation('sigmoid', name='sigmoid1')(added)
        conv2d3 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=1, padding='VALID', name='conv2d3')(sigmoid1)
        relu2 = tf.keras.layers.Activation('relu', name='relu2')(conv2d3)

        model1 = tf.keras.models.Model(inputs=[input1, input2], outputs=[relu1, relu2], name='model1')

        test_input1 = tf.random.uniform([1, 2, 2, 3])
        test_input2 = tf.random.uniform([1, 2, 2, 3])

        model_dir = './multiple_io'

        trace_compile_and_infer(model1, [test_input1, test_input2])

        feed_dict = tf2_compile(model1, model_dir, example_inputs=[test_input1, test_input2])
        run_inference(model_dir, [test_input1, test_input2], feed_dict)

    def test_branch_merge(self):
        input1 = tf.keras.Input(shape=[2, 2, 3], name='input1')
        conv2d1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=1, padding='VALID', name='conv2d1')(input1)
        conv2d2 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=1, padding='VALID', name='conv2d2')(input1)
        added = tf.keras.layers.Add(name='add')([conv2d1, conv2d2])
        relu1 = tf.keras.layers.Activation('relu', name='relu1')(added)
        model1 = tf.keras.models.Model(inputs=input1, outputs=[relu1, added], name='model1')

        test_input = tf.random.uniform([1, 2, 2, 3])
        model_dir = './branch_merge'


        trace_compile_and_infer(model1, test_input)

        feed_dict = tf2_compile(model1, model_dir, example_inputs=[test_input])
        run_inference(model_dir, [test_input], feed_dict)

    def test_no_fuse(self):
        input1 = tf.keras.Input(shape=[2, 2, 3], name='input1')
        conv2d1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=1, padding='VALID', name='conv2d1')(input1)
        conv2d2 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=1, padding='VALID', name='conv2d2')(input1)
        conv2d3 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=1, padding='VALID', name='conv2d3')(input1)
        add1 = tf.keras.layers.Add(name='add1')([conv2d1, conv2d2])
        add2 = tf.keras.layers.Add(name='add2')([conv2d1, conv2d3])
        add3 = tf.keras.layers.Add(name='add3')([conv2d2, conv2d3])
        add4 = tf.keras.layers.Add(name='add4')([add3, add2])
        add5 = tf.keras.layers.Add(name='add5')([add2, add1])
        add6 = tf.keras.layers.Add(name='add6')([add3, add1])
        relu1 = tf.keras.layers.Activation('relu', name='relu1')(add4)
        relu2 = tf.keras.layers.Activation('relu', name='relu2')(add5)
        relu3 = tf.keras.layers.Activation('relu', name='relu3')(add6)

        model1 = tf.keras.models.Model(inputs=input1, outputs=[relu1, relu2, relu3], name='model1')
        test_input = tf.random.uniform([1, 2, 2, 3])
        model_dir = './no_fuse'

        trace_compile_and_infer(model1, test_input)
  
        feed_dict = tf2_compile(model1, model_dir, example_inputs=[test_input])
        run_inference(model_dir, [test_input], feed_dict)

    def test_inputs_short_long(self):
        input1 = tf.keras.Input(shape=[3, 5], name='input1')
        input2 = tf.keras.Input(shape=[3, 5], name='input2')
        relu1 = tf.keras.layers.Activation('relu', name='relu1')(input1)
        relu2 = tf.keras.layers.Activation('relu', name='relu2')(relu1)
        relu3 = tf.keras.layers.Activation('relu', name='relu3')(relu2)
        add1 = tf.keras.layers.Add(name='add1')([input1, relu3])
        exp1 = tf.keras.layers.Activation('exponential', name='exp1')(add1)
        sig1 = tf.keras.layers.Activation('sigmoid', name='sig1')(add1)

        model1 = tf.keras.Model(inputs=[input1, input2], outputs=[sig1, relu3])
        model_dir = './inputs_short_long'

        test_input1 = tf.random.uniform([1,3,5])
        test_input2 = tf.random.uniform([1,3,5])

        trace_compile_and_infer(model1, [test_input1, test_input2])

        feed_dict = tf2_compile(model1, model_dir, example_inputs=[test_input1, test_input2])
        run_inference(model_dir, [test_input1, test_input2], feed_dict)

    def test_short_long_mid(self):
        input1 = tf.keras.Input(shape=[3, 5], name='input1')
        input2 = tf.keras.Input(shape=[3, 5], name='input2')
        input3 = tf.keras.Input(shape=[3, 5], name='input3')
        input4 = tf.keras.Input(shape=[3, 5], name='input4')
        identity = tf.keras.layers.Lambda(lambda x : x)([input1, input2, input3, input4])
        identity1, identity2, identity3, identity4 = identity
        relu1 = tf.keras.layers.Activation('relu', name='relu1')(identity4)
        relu2 = tf.keras.layers.Activation('relu', name='relu2')(relu1)
        relu3 = tf.keras.layers.Activation('relu', name='relu3')(relu2)
        add1 = tf.keras.layers.Add(name='add1')([identity1, relu3])
        exp1 = tf.keras.layers.Activation('exponential', name='exp1')(add1)
        sig1 = tf.keras.layers.Activation('sigmoid', name='sig1')(add1)

        model1 = tf.keras.Model(inputs=[input1, input2, input3, input4], outputs=[sig1, relu3, identity4])
        model_dir = './inputs_short_long_mid'

        test_input1 = tf.random.uniform([2,3,5])
        test_input2 = tf.random.uniform([1,3,5])
        test_input3 = tf.random.uniform([2,3,5])
        test_input4 = tf.random.uniform([2,3,5])

        trace_compile_and_infer(model1, [test_input1, test_input2, test_input3, test_input4])

        feed_dict = tf2_compile(model1, model_dir, example_inputs=[test_input1, test_input2, test_input3, test_input4])
        run_inference(model_dir, [test_input1, test_input2, test_input3, test_input4], feed_dict)

def tf2_compile(model, model_dir, example_inputs=None):
    shutil.rmtree(model_dir, ignore_errors=True)
    tf.keras.models.save_model(model, model_dir)
    model_feed_dict={} 
    if example_inputs is not None:
        loaded_model = tf.saved_model.load(model_dir)
        func = loaded_model.signatures['serving_default']
        for i in range(len(example_inputs)):
            model_feed_dict[func.function_def.signature.input_arg[i].name] = example_inputs[i]
        

    result_compile = tfn.saved_model.compile(model_dir, model_dir + '_neuron', model_feed_dict=model_feed_dict)
    assert result_compile['OnNeuronRatio'] > 0.05
    return model_feed_dict


def run_inference(model_dir, test_input, model_feed_dict):
    #actually make it the neuron_model_dir
    neuron_model_dir = model_dir + '_neuron'

    neuron_model = tf.keras.models.load_model(neuron_model_dir)
    model = tf.keras.models.load_model(model_dir)

    inf_func_neuron = neuron_model.signatures['serving_default']
    inf_func_cpu = model.signatures['serving_default']

    try:
        normal_output = model(test_input)
        neuron_output = neuron_model(test_input)
        np.testing.assert_allclose(normal_output, neuron_output,rtol=.01, atol=1e-2)
        print("normal compare success", model_dir)
    except ValueError:
        print("normal compare fail", model_dir)
    
    result_keyword_ref = inf_func_neuron(**model_feed_dict)
    result_keyword_neuron = inf_func_cpu(**model_feed_dict)

    for output_key in result_keyword_ref:
        np.testing.assert_allclose(result_keyword_neuron[output_key], result_keyword_ref[output_key], rtol=1e-2, atol=1e-2)
    print("model.signature['serving_default'] compare success", model_dir)


def trace_compile_and_infer(model, example_inputs):
    neuron_model = tfn.trace(model, example_inputs)
    neuron_result = neuron_model(example_inputs)
    cpu_result = model(example_inputs)
    print(cpu_result, neuron_result)
    np.testing.assert_allclose(cpu_result, neuron_result, rtol=1e-2, atol=1e-2)
