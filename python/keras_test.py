"""
Copyright (C) 2019, Amazon.com. All Rights Reserved
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import unittest
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras.applications.resnet50 import ResNet50
import tensorflow.neuron as tfn
from tensorflow.python.neuron.python.graph_util_test import _assert_compiler_success


_RANDOM_SEED = 15213


class TestKeras(unittest.TestCase):

    def test_keras_resnet50_float16_compile(self):
        # Instantiate Keras ResNet50 model
        keras.backend.set_learning_phase(0)
        keras.backend.set_floatx('float16')
        model = ResNet50(weights='imagenet')

        # Export SavedModel
        model_dir = './keras_resnet50_float16'
        shutil.rmtree(model_dir, ignore_errors=True)
        tf.saved_model.simple_save(keras.backend.get_session(), model_dir,
                                   {'input': model.inputs[0]}, {'output': model.outputs[0]})

        # Compile using Neuron
        compiled_model_dir = './keras_resnet50_float16_neuron'
        shutil.rmtree(compiled_model_dir, ignore_errors=True)
        try:
            compile_output = tfn.saved_model.compile(
                model_dir, compiled_model_dir,
                model_feed_dict={'input' : np.random.rand(1, 224, 224, 3)})
        except ValueError as ve:
            assert 'half is not in the list of allowed values' in str(ve)
