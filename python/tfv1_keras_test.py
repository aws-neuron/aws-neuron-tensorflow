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
import shutil
import random
from itertools import product
import tensorflow as tf
import tensorflow.compat.v1 as v1
import tensorflow.neuron as tfn
import pdb
import os


#each number represents the number of random 
#parameters from that catergory 
#that will be used in testing

NUM_ACTIVATIONS = 1  #Max is 6
NUM_INPUT_UNITS = 1  #Max is 1024 (would not use more than 10)
NUM_OUTPUT_UNITS = 1 #Max is 1024 (would not use more than 10)
NUM_MAGIC_NUMBERS = 1#Max is 10
NUM_KERNEL_SIZES = 1 #Max is 2
NUM_POWERS = 1       #Max is 11
NUM_FILTERS = 1
_PARAMETER_SEED = '11251998'


#here are the parameter lists
inputNumUnits = list(range(1, 1025))
outputNumUnits = list(range(1, 1025))
filterSizes = list(range(1, 51))
magicNumbers = [28, 14, 7, 224, 112, 56, 28, 299, 150, 75]
kernelSizes = [1, 3]
activations = ['softmax',  'relu', 'tanh', 'sigmoid',  'exponential', 'linear']
powersOfTwo = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

#set random seed for shuffling
#the parameter lists
random.seed(a=_PARAMETER_SEED)



#randomize the order
random.shuffle(inputNumUnits)
random.shuffle(outputNumUnits)
random.shuffle(magicNumbers)
random.shuffle(kernelSizes)
random.shuffle(activations)
random.shuffle(powersOfTwo)

#Pick the first n params based on specified constants above 
inputNumUnits = inputNumUnits[0:NUM_INPUT_UNITS]
outputNumUnits = outputNumUnits[0:NUM_OUTPUT_UNITS]
magicNumbers = magicNumbers[0:NUM_MAGIC_NUMBERS]
kernelSizes = kernelSizes[0:NUM_KERNEL_SIZES]
activations = activations[0:NUM_ACTIVATIONS]
powersOfTwo = powersOfTwo[0:NUM_POWERS]
filterSizes = filterSizes[0:NUM_FILTERS]

#reduce starting weight size to 
#help assert_allclose pass
initializer = v1.keras.initializers.RandomNormal(stddev=.001)

class TestKerasTF(unittest.TestCase):
    def setUp(self):
        #this one will be used for np rand functions
        self.random_seed = 7051994
    
    def test_flatten_dense_dropout(self):
        
        param_list = list(product(inputNumUnits, activations, outputNumUnits))
        np.random.seed(self.random_seed)
        for inu, a, onu in param_list:
            #subTest allows us to generate tests dynamically
            #if one of the subTests fail, the error message
            #along with the inputs (inu a onu) will be displayed.
            #however this will still show up as 1 test even though
            #there can be many subTests

            with self.subTest(inputNumUnits = inu, activations = a, outputNumUnits = onu):
                with v1.Session(graph=tf.Graph()) as sess:
                    model = v1.keras.models.Sequential([
                    v1.keras.layers.Flatten(input_shape=(28,28)),
                    v1.keras.layers.Dense(inu, activation=a, kernel_initializer=initializer),
                    v1.keras.layers.Dropout(0.2),
                    v1.keras.layers.Dense(onu, kernel_initializer=initializer)])

                    
                    #compile v1
                    model_dir = './keras_flatten_dense_dropout'
                test_input = {'input' :np.random.rand(1, 28, 28)}
                
                compiled_model_dir = run_v1_compile(model, sess, model_dir, test_input)
                run_inference_if_available(model_dir, compiled_model_dir, test_input)


def test_conv2d_conv2d_flatten_dense(self):
    #this test is similar to the one above, but the
    #NN and parameters vary
    
    param_list = list(product(filterSizes, activations, outputNumUnits, kernelSizes))
    np.random.seed(self.random_seed)
    for fs, a, onu, ks in param_list:

        with v1.Session(graph=tf.Graph()) as sess:
            with self.subTest(filterSizes=fs, activations=a, outputNumUnits=onu, kernelSizes=ks):
                model = v1.keras.models.Sequential([
                v1.keras.layers.Conv2D(fs, kernel_size=ks, 
                                        activation=a, input_shape=(28,28,ks), 
                                        kernel_initializer = initializer),
                v1.keras.layers.Conv2D(fs, kernel_size=ks, activation=a,
                                        kernel_initializer=initializer),
                v1.keras.layers.Flatten(),
                v1.keras.layers.Dense(onu, kernel_initializer=initializer)])

                model_dir = 'keras_conv2d_conv2d_flatten_dense'
                test_input = {'input' : np.random.rand(1, 28, 28, ks)}

                compiled_model_dir = run_v1_compile(model, sess, model_dir, test_input)
                run_inference_if_available(model_dir, compiled_model_dir, test_input)



@unittest.expectedFailure
def test_lstm_lstm_dense_dense(self):
    #this test is similar to the one above, but the
    #NN and parameters vary
    param_list = list(product(inputNumUnits, activations, outputNumUnits))
    np.random.seed(self.random_seed)
    for inu, a, onu in param_list:

        with v1.Session(graph=tf.Graph()) as sess:
            with self.subTest(inputNumUnits=inu, activations=a, outputNumUnits=onu):
                model = v1.keras.models.Sequential([
                v1.keras.layers.LSTM(inu, activation=a, input_shape=(28,28), return_sequences=True),
                v1.keras.layers.LSTM(inu, activation=a),
                v1.keras.layers.Dense(onu, activation=a),
                v1.keras.layers.Dense(10, activation=a)])

                model_dir = './keras_lstm_lstm_dense_dense'
                test_input = {'input' : np.random.rand(1, 28, 28)}

                compiled_model_dir = run_v1_compile(model, sess, model_dir, test_input)
                run_inference_if_available(model_dir, compiled_model_dir, test_input)



def test_maxpool2d(self):
    param_list = list(inputNumUnits)
    np.random.seed(self.random_seed)
    for inu in param_list:
        with v1.Session(graph=tf.Graph()) as sess:
            with self.subTest(inputNumUnits=inu):
                    model = v1.keras.models.Sequential([
                    v1.keras.layers.MaxPool2D(pool_size=(2,2), strides=1, padding='same', input_shape=(inu, inu, 1))])

                    model_dir = './keras_maxpool2d'
                    test_input = {'input' : np.random.rand(1, inu, inu, 1)}

                    compiled_model_dir = run_v1_compile(model, sess, model_dir, test_input)
                    run_inference_if_available(model_dir, compiled_model_dir, test_input)

    def test_toy_resnet(self):
        with v1.Session(graph=tf.Graph()) as sess:
            np.random.seed(self.random_seed)

            inputs = v1.keras.Input(shape=(32, 32, 3), name="img")
            x = v1.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
            x = v1.keras.layers.Conv2D(64, 3, activation="relu")(x)
            block_1_output = v1.keras.layers.MaxPooling2D(3)(x)

            x = v1.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
            x = v1.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
            block_2_output = v1.keras.layers.add([x, block_1_output])

            x = v1.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
            x = v1.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
            block_3_output = v1.keras.layers.add([x, block_2_output])

            x = v1.keras.layers.Conv2D(64, 3, activation="relu")(block_3_output)
            x = v1.keras.layers.GlobalAveragePooling2D()(x)
            x = v1.keras.layers.Dense(256, activation="relu")(x)
            x = v1.keras.layers.Dropout(0.5)(x)
            outputs = v1.keras.layers.Dense(10)(x)

            model = v1.keras.Model(inputs, outputs, name="toy_resnet")

            model_dir = './keras_toy_resnet'
            test_input = {'input' : np.random.rand(1,32,32,3)}

            compiled_model_dir = run_v1_compile(model, sess, model_dir, test_input)
            run_inference_if_available(model_dir, compiled_model_dir, test_input)
        


def run_v1_compile(model, sess, model_dir, test_input):
    tensor_input = model.inputs[0]

    tensor_output = model.outputs[0]
    sess.run(v1.local_variables_initializer())
    sess.run(v1.global_variables_initializer())

    shutil.rmtree(model_dir, ignore_errors=True)
    v1.saved_model.simple_save(sess, model_dir, {'input' : tensor_input}
                                , {'output' : tensor_output})
    
    #compile v1
    compiled_model_dir = model_dir + '_neuron'
    shutil.rmtree(compiled_model_dir, ignore_errors=True)


    compile_output = tfn.saved_model.compile(
                        model_dir, compiled_model_dir,
                        model_feed_dict=test_input)

    return compiled_model_dir




def run_inference_if_available(model_dir, compiled_model_dir, test_input):
    #soon we will not be using tf.contrib anymore
    #this function tests inference on inf1 hardware
    
    #first, check if hardware is avaiable using
    #the NEURON_TF_COMPILE_ONLY env variable
    #if NEURON_TF_COMPILE_ONLY is not set then
    #hardware should be avaialable
    if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
        pred_ref = tf.contrib.predictor.from_saved_model(model_dir)
        pred_neuron = tf.contrib.predictor.from_saved_model(compiled_model_dir)
        result_ref = pred_ref(test_input)
        result_neuron = pred_neuron(test_input)
        np.testing.assert_allclose(result_neuron['output'], result_ref['output'], rtol=1e-2, atol=1e-3)
