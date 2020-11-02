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

NUM_ACTIVATIONS = 2  #Max is 6
NUM_INPUT_UNITS = 2  #Max is 1024 (would not use more than 10)
NUM_OUTPUT_UNITS = 2 #Max is 1024 (would not use more than 10)
NUM_MAGIC_NUMBERS = 2#Max is 10
NUM_KERNEL_SIZES = 2 #Max is 2
NUM_POWERS = 2       #Max is 11
NUM_FILTERS = 2
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
    #This function tests a basic NN with two dense layers.
    #It has 3 paramaters which vary.
    #1. Number of input units
    #2. Number of output units
    #3. The type of activation function that the input layer uses

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
                #old syntax
                with v1.Session(graph=tf.Graph()) as sess:
                    
                    #remove old directory where the 
                    #previous model was stored
                    model_dir = './keras_flatten_dense_dropout'
                    shutil.rmtree(model_dir, ignore_errors=True)


                    #model creation and simplesave v1
                    model = v1.keras.models.Sequential([
                    v1.keras.layers.Flatten(input_shape=(28,28)),
                    v1.keras.layers.Dense(inu, activation=a, kernel_initializer=initializer),
                    v1.keras.layers.Dropout(0.2),
                    v1.keras.layers.Dense(onu, kernel_initializer=initializer)])

                    tensor_input = model.inputs[0]

                    tensor_output = model.outputs[0]
                    sess.run(v1.local_variables_initializer())
                    sess.run(v1.global_variables_initializer())

                    v1.saved_model.simple_save(sess, model_dir, {'input' : tensor_input}
                                                , {'output' : tensor_output})
                    
                    #compile v1
                    compiled_model_dir = './keras_flatten_dense_dropout_neuron'
                    shutil.rmtree(compiled_model_dir, ignore_errors=True)

                    test_input = {'input' :np.random.rand(1, 28, 28)}

                    compile_output = tfn.saved_model.compile(
                                        model_dir, compiled_model_dir,
                                        model_feed_dict=test_input)



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

                    tensor_input = model.inputs[0]
                    tensor_output = model.outputs[0]
                    sess.run(v1.local_variables_initializer())
                    sess.run(v1.global_variables_initializer())

                    model_dir = './keras_conv2d_conv2d_flatten_dense'
                    shutil.rmtree(model_dir, ignore_errors=True)

                    v1.saved_model.simple_save(sess, model_dir, {'input' : tensor_input}
                                                , {'output' : tensor_output})



                    #compile v1
                    compiled_model_dir = './keras_conv2d_conv2d_flatten_dense_neuron'
                    shutil.rmtree(compiled_model_dir, ignore_errors=True)

                    test_input = {'input' :np.random.rand(1, 28, 28, ks)}

                    compile_output = tfn.saved_model.compile(
                                        model_dir, compiled_model_dir,
                                        model_feed_dict=test_input)

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

                    tensor_input = model.inputs[0]
                    tensor_output = model.outputs[0]
                    sess.run(v1.local_variables_initializer())
                    sess.run(v1.global_variables_initializer())

                    # Export SavedModel
                    model_dir = './keras_lstm_lstm_dense_dense'
                    shutil.rmtree(model_dir, ignore_errors=True)


                    v1.saved_model.simple_save(sess, model_dir, {'input' : tensor_input}
                                                , {'output' : tensor_output})



                    #compile v1
                    compiled_model_dir = './keras_lstm_lstm_dense_dense_neuron'
                    shutil.rmtree(compiled_model_dir, ignore_errors=True)

                    test_input = {'input' :np.random.rand(1, 28, 28)}

                    compile_output = tfn.saved_model.compile(
                                        model_dir, compiled_model_dir,
                                        model_feed_dict=test_input)

                    run_inference_if_available(model_dir, compiled_model_dir, test_input)





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
