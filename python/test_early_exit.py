import tensorflow as tf
import tensorflow.compat.v2 as v2
import tensorflow.neuron as tfn
import shutil
import numpy as np
import unittest

from tensorflow.neuron.python.unittest_base import TestV1Only

class TestEarlyExit(TestV1Only):
    def test_sequential(self):
        with self.assertRaises(NotImplementedError):
            model = v2.keras.models.Sequential([
            v2.keras.layers.Flatten(input_shape=(28,28)),
            v2.keras.layers.Dense(28, activation='relu'),
            v2.keras.layers.Dropout(0.2),
            v2.keras.layers.Dense(1)])

            model_dir = './keras_flatten_dense_dropout'
            test_input = {'input0' :np.random.rand(1, 28, 28)}

            tf.saved_model.save(model, model_dir)

            compiled_model_dir = model_dir + '_neuron'
            shutil.rmtree(compiled_model_dir, ignore_errors=True)

            tfn.saved_model.compile(
                            model_dir, compiled_model_dir,
                            model_feed_dict=test_input)

    def test_functional(self):
        with self.assertRaises(NotImplementedError):
            num_tags = 12  # Number of unique issue tags
            num_words = 10000  # Size of vocabulary obtained when preprocessing text data
            num_departments = 4  # Number of departments for predictions

            title_input = v2.keras.Input(
                shape=(None,), name="title"
            )  # Variable-length sequence of ints
            body_input = v2.keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
            tags_input = v2.keras.Input(
                shape=(num_tags,), name="tags"
            )  # Binary vectors of size `num_tags`

            # Embed each word in the title into a 64-dimensional vector
            title_features = v2.keras.layers.Embedding(num_words, 64)(title_input)
            # Embed each word in the text into a 64-dimensional vector
            body_features = v2.keras.layers.Embedding(num_words, 64)(body_input)

            # Reduce sequence of embedded words in the title into a single 128-dimensional vector
            title_features = v2.keras.layers.LSTM(128)(title_features)
            # Reduce sequence of embedded words in the body into a single 32-dimensional vector
            body_features = v2.keras.layers.LSTM(32)(body_features)

            # Merge all available features into a single large vector via concatenation
            x = v2.keras.layers.concatenate([title_features, body_features, tags_input])

            # Stick a logistic regression for priority prediction on top of the features
            priority_pred = v2.keras.layers.Dense(1, name="priority")(x)
            # Stick a department classifier on top of the features
            department_pred = v2.keras.layers.Dense(num_departments, name="department")(x)

            # Instantiate an end-to-end model predicting both priority and department
            model = v2.keras.Model(
                inputs=[title_input, body_input, tags_input],
                outputs=[priority_pred, department_pred],
            )

            
            model_dir = './keras_multiple_io'
            tf.saved_model.save(model, model_dir)

            compiled_model_dir = model_dir + '_neuron'
            shutil.rmtree(compiled_model_dir, ignore_errors=True)

            # Dummy input data
            title_data = np.random.randint(num_words, size=(1280, 10))
            body_data = np.random.randint(num_words, size=(1280, 100))
            tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

            test_input = {'input0' : title_data, 'input1' : body_data, 'input2' : tags_data}

            tfn.saved_model.compile(
                            model_dir, compiled_model_dir,
                            model_feed_dict=test_input)


