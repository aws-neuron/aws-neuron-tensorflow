# Copyright 2020 AWS Neuron. All Rights Reserved.
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
import tensorflow as tf #supposed to be tf2
import random
from itertools import product
import shutil

#each number represents the number of random 
#parameters from that catergory 
#that will be used in testing

NUM_ACTIVATIONS = 1  #Max is 6
NUM_INPUT_UNITS = 1  #Max is 1024 (would not use more than 10)
NUM_OUTPUT_UNITS = 1 #Max is 1024 (would not use more than 10)
NUM_MAGIC_NUMBERS = 3#Max is 10
NUM_KERNEL_SIZES = 1 #Max is 2
NUM_POWERS = 2       #Max is 11


#here are the parameter lists
inputNumUnits = list(range(1, 1025))
outputNumUnits = list(range(1, 1025))
magicNumbers = [28, 14, 7, 224, 112, 56, 28, 299, 150, 75]
kernelSizes = [1, 3]
activations = ['softmax',  'relu', 'tanh', 'sigmoid',  'exponential', 'linear']
powersOfTwo = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


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


class TestKerasTF2(unittest.TestCase):
    #This function tests a basic NN with two dense layers.
    #It has 3 paramaters which vary.
    #1. Number of input units
    #2. Number of output units
    #3. The type of activation function that the input layer uses
    #@parameterized.expand()
    def test_flatten_dense_dropout(self):
        
        param_list = list(product(inputNumUnits, activations, outputNumUnits))
        for inu, a, onu in param_list:
            #subTest allows us to generate tests dynamically
            #if one of the subTests fail, the error message
            #along with the inputs (inu a onu) will be displayed.
            #however this will still show up as 1 test even though
            #there can be many subTests

            with self.subTest(inputNumUnits = inu, activations = a, outputNumUnits = onu):
                model = tf.keras.models.Sequential([
                #tf.keras.layers is tf2 syntax
                tf.keras.layers.Flatten(input_shape=(28,28,1)),
                tf.keras.layers.Dense(inu, activation=a),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(onu)])

                # Export SavedModel
                model_dir = './keras_flatten_dense_dropout'
                shutil.rmtree(model_dir, ignore_errors=True)
                tf.keras.models.save_model(model, model_dir)

                #we would then complie using TF Neuron with 2.0
                #support but this is just a prototype so we 
                #skip that step for now


                reloaded_model = tf.keras.models.load_model(model_dir)

                #in real test this would actually be a compiled model
                compiled_model = tf.keras.models.load_model(model_dir)

                test_input = np.random.random((1, 28, 28))
                #actual test would test compiler model on inf1
                #versus tf2 model on cpu
                np.testing.assert_allclose(
                    reloaded_model(test_input, training=False),
                    compiled_model(test_input, training=False))


    def test_conv2d_conv2d_flatten_dense(self):
        
        param_list = list(product(inputNumUnits, activations, outputNumUnits, kernelSizes))
        for inu, a, onu, ks in param_list:
            #subTest allows us to generate tests dynamically
            #if one of the subTests fail, the error message
            #along with the inputs (inu a onu) will be displayed.
            #however this will still show up as 1 test even though
            #there can be many subTests

            with self.subTest(inputNumUnits=inu, activations=a, outputNumUnits=onu, kernelSizes=ks):
                model = tf.keras.models.Sequential([
                #tf.keras.layers is tf2 syntax
                tf.keras.layers.Conv2D(inu, kernel_size=ks, 
                                        activation=a, input_shape=(28,28,1)),
                tf.keras.layers.Conv2D(inu, kernel_size=ks, activation=a),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(onu)])

                # Export SavedModel
                model_dir = './keras_conv2d_conv2d_flatten_dense'
                shutil.rmtree(model_dir, ignore_errors=True)
                tf.keras.models.save_model(model, model_dir)

                #we would then complie using TF Neuron with 2.0
                #support but this is just a prototype so we 
                #skip that step for now


                reloaded_model = tf.keras.models.load_model(model_dir)

                #in real test this would actually be a compiled model
                compiled_model = tf.keras.models.load_model(model_dir)

                test_input = np.random.random((1, 28, 28, 1))


                #actual test would test compiler model on inf1
                #versus tf2 model on cpu
                np.testing.assert_allclose(
                    reloaded_model(test_input, training=False),
                    compiled_model(test_input, training=False))

    def test_lstm_lstm_dense_dense(self):
        
        param_list = list(product(inputNumUnits, activations, outputNumUnits))
        for inu, a, onu in param_list:
            #subTest allows us to generate tests dynamically
            #if one of the subTests fail, the error message
            #along with the inputs (inu a onu) will be displayed.
            #however this will still show up as 1 test even though
            #there can be many subTests

            with self.subTest(inputNumUnits=inu, activations=a, outputNumUnits=onu):
                model = tf.keras.models.Sequential([
                #tf.keras.layers is tf2 syntax
                tf.keras.layers.LSTM(inu, activation=a, input_shape=(28,28), return_sequences=True),
                tf.keras.layers.LSTM(inu, activation=a),
                tf.keras.layers.Dense(onu, activation=a),
                tf.keras.layers.Dense(10, activation=a)])

                # Export SavedModel
                model_dir = './keras_lstm_lstm_dense_dense'
                shutil.rmtree(model_dir, ignore_errors=True)
                tf.keras.models.save_model(model, model_dir)

                #we would then complie using TF Neuron with 2.0
                #support but this is just a prototype so we 
                #skip that step for now


                reloaded_model = tf.keras.models.load_model(model_dir)

                #in real test this would actually be a compiled model
                compiled_model = tf.keras.models.load_model(model_dir)

                test_input = np.random.random((1, 28, 28))


                #actual test would test compiler model on inf1
                #versus tf2 model on cpu
                np.testing.assert_allclose(
                    reloaded_model(test_input, training=False),
                    compiled_model(test_input, training=False))






