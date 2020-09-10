"""
Copyright (C) 2019, Amazon.com. All Rights Reserved
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import shutil
import subprocess
import json
import unittest
import numpy as np
import tensorflow as tf
import tensorflow.neuron as tfn
from tensorflow.python.saved_model.saved_model import signature_constants
from tensorflow.neuron.python.graph_util_test import _assert_compiler_success


_RANDOM_SEED = 15213


class TestSimpleSave(unittest.TestCase):

    def test_simple(self):
        export_dir_ref = './simple_save_ref'
        export_dir_test = './simple_save_test'
        tags = [tf.saved_model.tag_constants.SERVING]
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input1')
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

            # Save the current session using tensorflow's simple_save() method
            shutil.rmtree(export_dir_ref, ignore_errors=True)
            tf.saved_model.simple_save(sess, export_dir=export_dir_ref, inputs=inputs, outputs=outputs)

            # Save the current session using tensorflow-neuron's simple_save() method
            shutil.rmtree(export_dir_test, ignore_errors=True)
            tfn.saved_model.simple_save(sess, export_dir=export_dir_test, inputs=inputs, outputs=outputs)

        # load two predictors from neuron saved and tf simple_saved models
        pred_ref = tf.contrib.predictor.from_saved_model(export_dir_ref)
        pred_test = tf.contrib.predictor.from_saved_model(export_dir_test)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            # Test for accuracy
            model_feed_dict = {
                'x0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
                'x1': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_ref = pred_ref(model_feed_dict)
            result_test = pred_test(model_feed_dict)
            for name in result_ref.keys():
                np.testing.assert_allclose(result_test[name], result_ref[name], rtol=1e-2, atol=1e-3)


class TestConvertToInferenceModel(unittest.TestCase):

    def test_simple(self):
        np.random.seed(_RANDOM_SEED)
        model_dir = './original_saved_model0'
        new_model_dir = './neuron_saved_model0'
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input1')
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
            tf.saved_model.simple_save(sess, export_dir=model_dir, inputs=inputs, outputs=outputs)
        shutil.rmtree(new_model_dir, ignore_errors=True)
        tfn.saved_model.compile(model_dir, new_model_dir)
        pred_ref = tf.contrib.predictor.from_saved_model(model_dir)
        pred_neuron = tf.contrib.predictor.from_saved_model(new_model_dir)
        assert len(pred_neuron.graph.get_operations()) == 5
        assert pred_neuron.graph.get_operations()[2].type == 'NeuronOp'
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            model_feed_dict = {
                'x0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
                'x1': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_ref = pred_ref(model_feed_dict)
            result_neuron = pred_neuron(model_feed_dict)
            for name in result_ref.keys():
                np.testing.assert_allclose(result_neuron[name], result_ref[name], rtol=1e-2, atol=1e-3)

    def test_feed_dict(self):
        np.random.seed(_RANDOM_SEED)
        model_dir = './original_saved_model1'
        new_model_dir = './neuron_saved_model1'
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input1')
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
            tf.saved_model.simple_save(sess, export_dir=model_dir, inputs=inputs, outputs=outputs)
        model_feed_dict = {
            'x0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            'x1': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        }
        shutil.rmtree(new_model_dir, ignore_errors=True)
        tf.neuron.saved_model.compile(model_dir, new_model_dir, model_feed_dict=model_feed_dict)
        pred_ref = tf.contrib.predictor.from_saved_model(model_dir)
        pred_neuron = tf.contrib.predictor.from_saved_model(new_model_dir)
        assert len(pred_neuron.graph.get_operations()) == 5
        assert pred_neuron.graph.get_operations()[2].type == 'NeuronOp'
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_ref = pred_ref(model_feed_dict)
            result_neuron = pred_neuron(model_feed_dict)
            for name in result_ref.keys():
                np.testing.assert_allclose(result_neuron[name], result_ref[name], rtol=1e-2, atol=1e-3)

    def test_multi_tags_multi_sigdefs(self):
        np.random.seed(_RANDOM_SEED)
        model_dir = './original_saved_model_multi_tags_multi_sigdefs'
        new_model_dir0 = './neuron_saved_model_multi_tags_multi_sigdefs0'
        new_model_dir1 = './neuron_saved_model_multi_tags_multi_sigdefs1'
        new_model_dir2 = './neuron_saved_model_multi_tags_multi_sigdefs2'
        tags0 = [tf.saved_model.tag_constants.SERVING, tf.saved_model.tag_constants.GPU]
        tags1 = [tf.saved_model.tag_constants.TPU, tf.saved_model.tag_constants.GPU]
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input1')
            input2 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input2')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            output0 = tf.identity(relu0, name='output0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            conv2d2 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            relu1 = tf.nn.relu(conv2d2, name='relu1')
            output1 = tf.identity(relu1, name='output1')
            add1 = tf.add(input2, relu1, name='add1')
            relu2 = tf.nn.relu(add1, name='relu2')
            output2 = tf.identity(relu2, name='output2')
            shutil.rmtree(model_dir, ignore_errors=True)

            inputs_sigdef0 = {
                'input0': tf.saved_model.utils.build_tensor_info(input0),
                'input1': tf.saved_model.utils.build_tensor_info(input1),
            }
            outputs_sigdef0 = {
                'output0': tf.saved_model.utils.build_tensor_info(output0),
                'output1': tf.saved_model.utils.build_tensor_info(output1),
            }
            sigdef0 = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs_sigdef0,
                outputs=outputs_sigdef0,
                method_name='method0',
            )

            inputs_sigdef1 = {
                'input2': tf.saved_model.utils.build_tensor_info(input0),  # intentionally flipped
                'input1': tf.saved_model.utils.build_tensor_info(input1),
                'input0': tf.saved_model.utils.build_tensor_info(input2),  # intentionally flipped
            }
            outputs_sigdef1 = {
                'output0': tf.saved_model.utils.build_tensor_info(output2),
                'output2': tf.saved_model.utils.build_tensor_info(output0),
            }
            sigdef1 = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs_sigdef1,
                outputs=outputs_sigdef1,
                method_name='method1',
            )

            inputs_sigdef2 = {
                'input0': tf.saved_model.utils.build_tensor_info(input0),
                'input2': tf.saved_model.utils.build_tensor_info(input2),
                'input1': tf.saved_model.utils.build_tensor_info(input1),
            }
            outputs_sigdef2 = {
                'output2': tf.saved_model.utils.build_tensor_info(output2),
                'output1': tf.saved_model.utils.build_tensor_info(output1),
            }
            sigdef2 = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs_sigdef2,
                outputs=outputs_sigdef2,
                method_name='method2',
            )

            inputs_sigdef3 = {
                'input3': tf.saved_model.utils.build_tensor_info(input2),
                'input0': tf.saved_model.utils.build_tensor_info(input0),
                'input1': tf.saved_model.utils.build_tensor_info(input1),
            }
            outputs_sigdef3 = {
                'output3': tf.saved_model.utils.build_tensor_info(output2),
                'output1': tf.saved_model.utils.build_tensor_info(output1),
            }
            sigdef3 = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs_sigdef3,
                outputs=outputs_sigdef3,
                method_name='method3',
            )

            signature_def_map0 = {
                'sigdef0': sigdef0,
                'sigdef1': sigdef1,
                'sigdef2': sigdef2,
            }
            signature_def_map1 = {
                'sigdef3': sigdef3,
                'sigdef2': sigdef2,
            }

            builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
            builder.add_meta_graph_and_variables(sess, tags0, signature_def_map=signature_def_map0,
                                                 strip_default_attrs=True)
            builder.add_meta_graph(tags1, signature_def_map=signature_def_map1, strip_default_attrs=True)
            builder.save()

        feed_dict = {
            'input0:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            'input1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            'input2:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        }
        model_feed_dict0 = {key: feed_dict[inputs_sigdef0[key].name] for key in inputs_sigdef0.keys()}
        model_feed_dict1 = {key: feed_dict[inputs_sigdef1[key].name] for key in inputs_sigdef1.keys()}
        model_feed_dict2 = {key: feed_dict[inputs_sigdef2[key].name] for key in inputs_sigdef2.keys()}
        shutil.rmtree(new_model_dir0, ignore_errors=True)
        shutil.rmtree(new_model_dir1, ignore_errors=True)
        shutil.rmtree(new_model_dir2, ignore_errors=True)
        tf.neuron.saved_model.compile(model_dir, new_model_dir0, signature_def_key='sigdef0', model_feed_dict=model_feed_dict0)
        tf.neuron.saved_model.compile(model_dir, new_model_dir1, signature_def_key='sigdef1', model_feed_dict=model_feed_dict1)
        tf.neuron.saved_model.compile(model_dir, new_model_dir2, signature_def_key='sigdef2', model_feed_dict=model_feed_dict2)

        tags = 'serve,gpu'
        pred_ref0 = tf.contrib.predictor.from_saved_model(model_dir, tags=tags, signature_def_key='sigdef0')
        pred_ref1 = tf.contrib.predictor.from_saved_model(model_dir, tags=tags, signature_def_key='sigdef1')
        pred_ref2 = tf.contrib.predictor.from_saved_model(model_dir, tags=tags, signature_def_key='sigdef2')
        pred_neuron0 = tf.contrib.predictor.from_saved_model(new_model_dir0, tags=tags, signature_def_key='sigdef0')
        pred_neuron1 = tf.contrib.predictor.from_saved_model(new_model_dir1, tags=tags, signature_def_key='sigdef1')
        pred_neuron2 = tf.contrib.predictor.from_saved_model(new_model_dir2, tags=tags, signature_def_key='sigdef2')
        _assert_compiler_success(pred_neuron0.graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_ref0 = pred_ref0(model_feed_dict0)
            result_ref1 = pred_ref1(model_feed_dict1)
            result_ref2 = pred_ref2(model_feed_dict2)
            result_neuron0 = pred_neuron0(model_feed_dict0)
            result_neuron1 = pred_neuron1(model_feed_dict1)
            result_neuron2 = pred_neuron2(model_feed_dict2)
            for name in result_ref0.keys():
                np.testing.assert_allclose(result_neuron0[name], result_ref0[name], rtol=1e-2, atol=1e-3)
            for name in result_ref1.keys():
                np.testing.assert_allclose(result_neuron1[name], result_ref1[name], rtol=1e-2, atol=1e-3)
            for name in result_ref2.keys():
                np.testing.assert_allclose(result_neuron2[name], result_ref2[name], rtol=1e-2, atol=1e-3)

    def test_regress_api(self):
        np.random.seed(_RANDOM_SEED)
        model_dir = './original_saved_model_regress'
        new_model_dir = './neuron_saved_model_regress'
        tags = [tf.saved_model.tag_constants.SERVING]
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(conv2d0, np.random.uniform(-1, 1, size=[2, 2, 3, 1]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            shutil.rmtree(model_dir, ignore_errors=True)

            # Create a signature defination and save the model in the current session
            # regress only has 1 input and 1 output
            tensor_info_input0 = tf.saved_model.utils.build_tensor_info(input0)
            tensor_info_output0 = tf.saved_model.utils.build_tensor_info(conv2d1)
            serving_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={signature_constants.REGRESS_INPUTS: tensor_info_input0},
                    outputs={signature_constants.REGRESS_OUTPUTS: tensor_info_output0},
                    method_name=tf.saved_model.signature_constants.REGRESS_METHOD_NAME
                )
            )
            signature_def_map = {'bla': serving_signature}

            builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
            builder.add_meta_graph_and_variables(sess, tags, signature_def_map=signature_def_map,
                                                 strip_default_attrs=True)
            builder.save()

        model_feed_dict = {
            signature_constants.REGRESS_INPUTS: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16)
        }
        shutil.rmtree(new_model_dir, ignore_errors=True)
        tf.neuron.saved_model.compile(model_dir, new_model_dir, model_feed_dict=model_feed_dict)

        pred_ref = tf.contrib.predictor.from_saved_model(model_dir, signature_def_key='bla')
        pred_neuron = tf.contrib.predictor.from_saved_model(new_model_dir, signature_def_key='bla')
        _assert_compiler_success(pred_neuron.graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_ref = pred_ref(model_feed_dict)
            result_neuron = pred_neuron(model_feed_dict)
            for name in result_ref.keys():
                np.testing.assert_allclose(result_neuron[name], result_ref[name], rtol=1e-2, atol=1e-3)

    def test_classify_api(self):
        np.random.seed(_RANDOM_SEED)
        model_dir = './original_saved_model_classify'
        new_model_dir = './neuron_saved_model_classify'
        tags = [tf.saved_model.tag_constants.SERVING]
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(conv2d0, np.random.uniform(-1, 1, size=[2, 2, 3, 1]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            conv2d2 = tf.nn.conv2d(conv2d0, np.random.uniform(-1, 1, size=[2, 2, 3, 1]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d2')
            shutil.rmtree(model_dir, ignore_errors=True)

            # Create a signature defination and save the model in the current session
            # classify has 1 input and 2 outputs
            tensor_info_input0 = tf.saved_model.utils.build_tensor_info(input0)
            tensor_info_output0 = tf.saved_model.utils.build_tensor_info(conv2d1)
            tensor_info_output1 = tf.saved_model.utils.build_tensor_info(conv2d2)

            serving_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={signature_constants.CLASSIFY_INPUTS: tensor_info_input0},
                    outputs={signature_constants.CLASSIFY_OUTPUT_SCORES: tensor_info_output0,
                             signature_constants.CLASSIFY_OUTPUT_CLASSES: tensor_info_output1},
                    method_name=signature_constants.CLASSIFY_METHOD_NAME
                )
            )
            signature_def_map = {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: serving_signature}

            builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
            builder.add_meta_graph_and_variables(sess, tags, signature_def_map=signature_def_map,
                                                 strip_default_attrs=True)
            builder.save()

        model_feed_dict = {
            signature_constants.CLASSIFY_INPUTS: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16)
        }
        shutil.rmtree(new_model_dir, ignore_errors=True)
        tf.neuron.saved_model.compile(model_dir, new_model_dir, model_feed_dict=model_feed_dict)

        pred_ref = tf.contrib.predictor.from_saved_model(model_dir)
        pred_neuron = tf.contrib.predictor.from_saved_model(new_model_dir)
        _assert_compiler_success(pred_neuron.graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_ref = pred_ref(model_feed_dict)
            result_neuron = pred_neuron(model_feed_dict)
            for name in result_ref.keys():
                np.testing.assert_allclose(result_neuron[name], result_ref[name], rtol=1e-2, atol=1e-3)

    def test_estimator_table_init(self):
        embedding_ids = [str(x) for x in range(10)]
        emedding_vocab_file = './estimator_table_init_ids.txt'
        with open(emedding_vocab_file, 'w') as f:
            for idx in embedding_ids:
                f.write('{}\n'.format(idx))
        embedding_identity_column = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity("embedding_identity", 11), dimension=8)
        embedding_list_column = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list("embedding_list", embedding_ids), dimension=8)
        embedding_file_column = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_file("embedding_file", emedding_vocab_file), dimension=8)
        feature_columns = [embedding_identity_column, embedding_list_column, embedding_file_column]
        def input_fn():
            dataset = tf.data.experimental.RandomDataset()
            dataset = dataset.map(lambda x: (
                {
                    'embedding_identity': x % 10,
                    'embedding_list': str(x % 10),
                    'embedding_file': str(x % 10),
                },
                tf.cast(x % 9973, tf.float32) / 9973,
            ))
            dataset = dataset.batch(128)
            return dataset
        def model_fn(features, labels, mode, params, config):
            column_tensors = {}
            dense_tensor = tf.feature_column.input_layer(
                features, feature_columns, cols_to_output_tensors=column_tensors, trainable=True)
            net = tf.keras.layers.Concatenate()(list(column_tensors.values()))
            net = tf.keras.layers.Dense(4)(net)
            net = tf.keras.layers.Dense(2)(net)
            logits = tf.keras.layers.Dense(1)(net)
            head = tf.contrib.estimator.binary_classification_head()
            optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=1e-2)
            return head.create_estimator_spec(features=features, mode=mode, logits=logits,
                                              labels=labels, optimizer=optimizer)
        model_dir = './estimator_table_init'
        shutil.rmtree(model_dir, ignore_errors=True)
        estimator = tf.estimator.Estimator(model_dir=model_dir, model_fn=model_fn)
        estimator.train(input_fn, steps=10)
        export_dir_base = './estimator_table_init_saved_model'
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        shutil.rmtree(export_dir_base, ignore_errors=True)
        estimator.export_saved_model(export_dir_base, serving_input_receiver_fn)
        real_model_dir = glob.glob(os.path.join('./estimator_table_init_saved_model', '*'))[0]

        embedding_identity = tf.train.Feature(int64_list=tf.train.Int64List(value=np.random.randint(10, size=10)))
        embedding_file = tf.train.Feature(bytes_list=tf.train.BytesList(value=[emedding_vocab_file.encode()]))
        embedding_list = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'0123456789']))
        feature = {
            'embedding_identity': embedding_identity,
            'embedding_file': embedding_file,
            'embedding_list': embedding_list,
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        model_feed_dict = {'inputs': [example_proto.SerializeToString()]}
        pred_ref = tf.contrib.predictor.from_saved_model(real_model_dir)
        result_ref = pred_ref(model_feed_dict)

        neuron_model_dir = './estimator_table_init_saved_model_neuron'
        shutil.rmtree(neuron_model_dir, ignore_errors=True)
        tfn.saved_model.compile(real_model_dir, neuron_model_dir, model_feed_dict=model_feed_dict)
        pred_neuron = tf.contrib.predictor.from_saved_model(neuron_model_dir)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_neuron = pred_neuron(model_feed_dict)
            np.testing.assert_equal(result_neuron['classes'], result_ref['classes'])
            np.testing.assert_allclose(result_neuron['scores'], result_ref['scores'], rtol=1e-2, atol=1e-3)


class TestSavedModelCLIConvert(unittest.TestCase):

    @unittest.skipIf(not hasattr(tfn, 'ops'), 'tensorflow-neuron plugin does not support saved_model_cli')
    def test_saved_model_cli_convert_neuron(self):
        np.random.seed(_RANDOM_SEED)
        model_dir = './original_saved_model2'
        new_model_dir_b1 = './saved_model_cli_convert_neuron_b1'
        new_model_dir_b2 = './saved_model_cli_convert_neuron_b2'
        new_model_dir_b3 = './saved_model_cli_convert_neuron_b3'
        new_model_dir_b4 = './saved_model_cli_convert_neuron_b4'
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(new_model_dir_b1, ignore_errors=True)
        shutil.rmtree(new_model_dir_b2, ignore_errors=True)
        shutil.rmtree(new_model_dir_b3, ignore_errors=True)
        shutil.rmtree(new_model_dir_b4, ignore_errors=True)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input1')
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
            tf.saved_model.simple_save(sess, export_dir=model_dir, inputs=inputs, outputs=outputs)
        model_feed_dict_b1 = {
            'x0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            'x1': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        }
        proc = subprocess.run([
            'saved_model_cli', 'convert', '--tag_set', 'serve',
            '--dir', model_dir, '--output_dir', new_model_dir_b1, 'neuron'])
        assert proc.returncode == 0
        model_feed_dict_b2 = {
            'x0': np.random.uniform(-1, 1, size=[2, 2, 2, 3]).astype(np.float16),
            'x1': np.random.uniform(-1, 1, size=[2, 2, 2, 3]).astype(np.float16),
        }
        proc = subprocess.run([
            'saved_model_cli', 'convert', '--tag_set', 'serve',
            '--dir', model_dir, '--output_dir', new_model_dir_b2, 'neuron',
            '--batch_size', str(2)])
        assert proc.returncode == 0
        model_feed_dict_b3 = {
            'x0': np.random.uniform(-1, 1, size=[3, 2, 2, 3]).astype(np.float16),
            'x1': np.random.uniform(-1, 1, size=[3, 2, 2, 3]).astype(np.float16),
        }
        proc = subprocess.run([
            'saved_model_cli', 'convert', '--tag_set', 'serve',
            '--dir', model_dir, '--output_dir', new_model_dir_b3, 'neuron',
            '--input_shape_dict', json.dumps({'x0': [3, 2, 2, 3], 'x1': [3, 2, 2, 3]})])
        assert proc.returncode == 0
        model_feed_dict_b4 = {
            'x0': np.random.uniform(-1, 1, size=[4, 2, 2, 3]).astype(np.float16),
            'x1': np.random.uniform(-1, 1, size=[4, 2, 2, 3]).astype(np.float16),
        }
        model_feed_dict_npz = 'saved_model_cli_convert_neuron_b4_model_feed_dict.npz'
        np.savez(model_feed_dict_npz, **model_feed_dict_b4)
        proc = subprocess.run([
            'saved_model_cli', 'convert', '--tag_set', 'serve',
            '--dir', model_dir, '--output_dir', new_model_dir_b4, 'neuron',
            '--inputs', 'x0={0}[x0];x1={0}[x1]'.format(model_feed_dict_npz)])
        assert proc.returncode == 0
        pred_ref = tf.contrib.predictor.from_saved_model(model_dir)
        pred_neuron_b1 = tf.contrib.predictor.from_saved_model(new_model_dir_b1)
        pred_neuron_b2 = tf.contrib.predictor.from_saved_model(new_model_dir_b2)
        pred_neuron_b3 = tf.contrib.predictor.from_saved_model(new_model_dir_b3)
        pred_neuron_b4 = tf.contrib.predictor.from_saved_model(new_model_dir_b4)
        assert len(pred_neuron_b1.graph.get_operations()) == 5
        assert pred_neuron_b1.graph.get_operations()[2].type == 'NeuronOp'
        assert len(pred_neuron_b2.graph.get_operations()) == 5
        assert pred_neuron_b2.graph.get_operations()[2].type == 'NeuronOp'
        assert len(pred_neuron_b3.graph.get_operations()) == 5
        assert pred_neuron_b3.graph.get_operations()[2].type == 'NeuronOp'
        assert len(pred_neuron_b4.graph.get_operations()) == 5
        assert pred_neuron_b4.graph.get_operations()[2].type == 'NeuronOp'
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_ref_b1 = pred_ref(model_feed_dict_b1)
            result_ref_b2 = pred_ref(model_feed_dict_b2)
            result_ref_b3 = pred_ref(model_feed_dict_b3)
            result_ref_b4 = pred_ref(model_feed_dict_b4)
            result_neuron_b1 = pred_neuron_b1(model_feed_dict_b1)
            result_neuron_b2 = pred_neuron_b2(model_feed_dict_b2)
            result_neuron_b3 = pred_neuron_b3(model_feed_dict_b3)
            result_neuron_b4 = pred_neuron_b4(model_feed_dict_b4)
            for name in result_neuron_b1.keys():
                np.testing.assert_allclose(result_neuron_b1[name], result_ref_b1[name], rtol=1e-2, atol=1e-3)
                np.testing.assert_allclose(result_neuron_b2[name], result_ref_b2[name], rtol=1e-2, atol=1e-3)
                np.testing.assert_allclose(result_neuron_b3[name], result_ref_b3[name], rtol=1e-2, atol=1e-3)
                np.testing.assert_allclose(result_neuron_b4[name], result_ref_b4[name], rtol=1e-2, atol=1e-3)


class TestProfile(unittest.TestCase):

    def test_simple(self):
        export_dir = './simple_save_profile'
        tags = [tf.saved_model.tag_constants.SERVING]
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input1')
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

            # Save the current session using tensorflow's simple_save() method
            shutil.rmtree(export_dir, ignore_errors=True)
            tf.saved_model.simple_save(sess, export_dir=export_dir, inputs=inputs, outputs=outputs)
        tfn.saved_model.profile(export_dir)


class TestCoreBinding(unittest.TestCase):

    def test_set(self):
        model_dir = self._gen_saved_model()
        tfn.saved_model.set_core_binding(model_dir, [0, 0])
        pred_neuron = tf.contrib.predictor.from_saved_model(model_dir)
        _assert_compiler_success(pred_neuron.graph)
        model_feed_dict = {
            'x0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            'x1': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        }
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            result_neuron = pred_neuron(model_feed_dict)

    def test_inspect(self):
        model_dir = self._gen_saved_model()
        tfn.saved_model.inspect_core_binding(model_dir)

    def _gen_saved_model(self):
        export_dir = './simple_save_core_binding'
        tags = [tf.saved_model.tag_constants.SERVING]
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input1')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            add0 = tf.identity_n([add0])[0]
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            conv2d2 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            relu1 = tf.nn.relu(conv2d2, name='relu1')
            inputs = {'x0': input0, 'x1': input1}
            outputs = {'y0': relu0, 'y1': relu1}
            shutil.rmtree(export_dir, ignore_errors=True)
            tfn.saved_model.simple_save(sess, export_dir=export_dir, inputs=inputs, outputs=outputs)
            return export_dir


if __name__ == '__main__':
    unittest.main()
