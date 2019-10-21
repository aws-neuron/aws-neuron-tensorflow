"""
Copyright (C) 2019, Amazon.com. All Rights Reserved
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import json
import numpy as np
import tensorflow as tf
import tensorflow.neuron as tfn
from tensorflow.python.saved_model.saved_model import signature_constants
from tensorflow.python.neuron.python.graph_util_test import _assert_compiler_success


_RANDOM_SEED = 15213


def test_simple_save():
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
        tf.saved_model.simple_save(sess, export_dir=export_dir_ref, inputs=inputs, outputs=outputs)

        # Save the current session using tensorflow-neuron's simple_save() method
        tfn.saved_model.simple_save(sess, export_dir=export_dir_test, inputs=inputs, outputs=outputs)

    # load two predictors from neuron saved and tf simple_saved models
    pred_ref = tf.contrib.predictor.from_saved_model(export_dir_ref)
    pred_test = tf.contrib.predictor.from_saved_model(export_dir_test)
    _assert_compiler_success(pred_test.graph)
    if 'NEURON_RTD_ADDRESS' in os.environ:
        # Test for accuracy
        model_feed_dict = {
            'x0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            'x1': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        }
        result_ref = pred_ref(model_feed_dict)
        result_test = pred_test(model_feed_dict)
        for name in result_ref.keys():
            np.testing.assert_allclose(result_test[name], result_ref[name], rtol=1e-2, atol=1e-3)


def test_convert_to_inference_model():
    np.random.seed(_RANDOM_SEED)
    model_dir = './original_saved_model0'
    new_model_dir = './kaena_saved_model0'
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
    tfn.saved_model.compile(model_dir, new_model_dir)
    pred_ref = tf.contrib.predictor.from_saved_model(model_dir)
    pred_tonga = tf.contrib.predictor.from_saved_model(new_model_dir)
    assert len(pred_tonga.graph.get_operations()) == 5
    assert pred_tonga.graph.get_operations()[2].type == 'InferentiaOp'
    if 'NEURON_RTD_ADDRESS' in os.environ:
        model_feed_dict = {
            'x0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            'x1': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        }
        result_ref = pred_ref(model_feed_dict)
        result_tonga = pred_tonga(model_feed_dict)
        for name in result_ref.keys():
            np.testing.assert_allclose(result_tonga[name], result_ref[name], rtol=1e-2, atol=1e-3)


def test_convert_to_inference_model_with_feed_dict():
    np.random.seed(_RANDOM_SEED)
    model_dir = './original_saved_model1'
    new_model_dir = './kaena_saved_model1'
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
    model_feed_dict = {
        'x0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        'x1': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
    }
    tf.neuron.saved_model.compile(model_dir, new_model_dir, model_feed_dict=model_feed_dict)
    pred_ref = tf.contrib.predictor.from_saved_model(model_dir)
    pred_tonga = tf.contrib.predictor.from_saved_model(new_model_dir)
    assert len(pred_tonga.graph.get_operations()) == 5
    assert pred_tonga.graph.get_operations()[2].type == 'InferentiaOp'
    if 'NEURON_RTD_ADDRESS' in os.environ:
        result_ref = pred_ref(model_feed_dict)
        result_tonga = pred_tonga(model_feed_dict)
        for name in result_ref.keys():
            np.testing.assert_allclose(result_tonga[name], result_ref[name], rtol=1e-2, atol=1e-3)

def test_convert_to_inference_model_regress_api():
    np.random.seed(_RANDOM_SEED)
    model_dir = './original_saved_model_regress'
    new_model_dir = './kaena_saved_model_regress'
    tags = [tf.saved_model.tag_constants.SERVING]
    with tf.Session(graph=tf.Graph()) as sess:
        input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
        conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                               strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
        conv2d1 = tf.nn.conv2d(conv2d0, np.random.uniform(-1, 1, size=[2, 2, 3, 1]).astype(np.float16),
                               strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')

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
        signature_def_map = {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: serving_signature}

        builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
        builder.add_meta_graph_and_variables(sess, tags, signature_def_map=signature_def_map,
                                             strip_default_attrs=True)
        builder.save()

    model_feed_dict = {
        signature_constants.REGRESS_INPUTS: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16)
    }

    tf.neuron.saved_model.compile(model_dir, new_model_dir, model_feed_dict=model_feed_dict)

    pred_ref = tf.contrib.predictor.from_saved_model(model_dir)
    pred_tonga = tf.contrib.predictor.from_saved_model(new_model_dir)
    _assert_compiler_success(pred_tonga.graph)
    if 'NEURON_RTD_ADDRESS' in os.environ:
        result_ref = pred_ref(model_feed_dict)
        result_tonga = pred_tonga(model_feed_dict)
        for name in result_ref.keys():
            np.testing.assert_allclose(result_tonga[name], result_ref[name], rtol=1e-2, atol=1e-3)

def test_convert_to_inference_model_classify_api():
    np.random.seed(_RANDOM_SEED)
    model_dir = './original_saved_model_classify'
    new_model_dir = './kaena_saved_model_classify'
    tags = [tf.saved_model.tag_constants.SERVING]
    with tf.Session(graph=tf.Graph()) as sess:
        input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
        conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                               strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
        conv2d1 = tf.nn.conv2d(conv2d0, np.random.uniform(-1, 1, size=[2, 2, 3, 1]).astype(np.float16),
                               strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
        conv2d2 = tf.nn.conv2d(conv2d0, np.random.uniform(-1, 1, size=[2, 2, 3, 1]).astype(np.float16),
                               strides=[1, 1, 1, 1], padding='VALID', name='conv2d2')

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
    tf.neuron.saved_model.compile(model_dir, new_model_dir, model_feed_dict=model_feed_dict)

    pred_ref = tf.contrib.predictor.from_saved_model(model_dir)
    pred_tonga = tf.contrib.predictor.from_saved_model(new_model_dir)
    _assert_compiler_success(pred_tonga.graph)
    if 'NEURON_RTD_ADDRESS' in os.environ:
        result_ref = pred_ref(model_feed_dict)
        result_tonga = pred_tonga(model_feed_dict)
        for name in result_ref.keys():
            np.testing.assert_allclose(result_tonga[name], result_ref[name], rtol=1e-2, atol=1e-3)


def test_saved_model_cli_convert_kaena():
    np.random.seed(_RANDOM_SEED)
    model_dir = './original_saved_model2'
    new_model_dir_b1 = './saved_model_cli_convert_kaena_b1'
    new_model_dir_b2 = './saved_model_cli_convert_kaena_b2'
    new_model_dir_b3 = './saved_model_cli_convert_kaena_b3'
    new_model_dir_b4 = './saved_model_cli_convert_kaena_b4'
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
    model_feed_dict_npz = 'saved_model_cli_convert_kaena_b4_model_feed_dict.npz'
    np.savez(model_feed_dict_npz, **model_feed_dict_b4)
    proc = subprocess.run([
        'saved_model_cli', 'convert', '--tag_set', 'serve',
        '--dir', model_dir, '--output_dir', new_model_dir_b4, 'neuron',
        '--inputs', 'x0={0}[x0];x1={0}[x1]'.format(model_feed_dict_npz)])
    assert proc.returncode == 0
    pred_ref = tf.contrib.predictor.from_saved_model(model_dir)
    pred_tonga_b1 = tf.contrib.predictor.from_saved_model(new_model_dir_b1)
    pred_tonga_b2 = tf.contrib.predictor.from_saved_model(new_model_dir_b2)
    pred_tonga_b3 = tf.contrib.predictor.from_saved_model(new_model_dir_b3)
    pred_tonga_b4 = tf.contrib.predictor.from_saved_model(new_model_dir_b4)
    assert len(pred_tonga_b1.graph.get_operations()) == 5
    assert pred_tonga_b1.graph.get_operations()[2].type == 'InferentiaOp'
    assert len(pred_tonga_b2.graph.get_operations()) == 5
    assert pred_tonga_b2.graph.get_operations()[2].type == 'InferentiaOp'
    assert len(pred_tonga_b3.graph.get_operations()) == 5
    assert pred_tonga_b3.graph.get_operations()[2].type == 'InferentiaOp'
    assert len(pred_tonga_b4.graph.get_operations()) == 5
    assert pred_tonga_b4.graph.get_operations()[2].type == 'InferentiaOp'
    if 'NEURON_RTD_ADDRESS' in os.environ:
        result_ref_b1 = pred_ref(model_feed_dict_b1)
        result_ref_b2 = pred_ref(model_feed_dict_b2)
        result_ref_b3 = pred_ref(model_feed_dict_b3)
        result_ref_b4 = pred_ref(model_feed_dict_b4)
        result_tonga_b1 = pred_tonga_b1(model_feed_dict_b1)
        result_tonga_b2 = pred_tonga_b2(model_feed_dict_b2)
        result_tonga_b3 = pred_tonga_b3(model_feed_dict_b3)
        result_tonga_b4 = pred_tonga_b4(model_feed_dict_b4)
        for name in result_tonga_b1.keys():
            np.testing.assert_allclose(result_tonga_b1[name], result_ref_b1[name], rtol=1e-2, atol=1e-3)
            np.testing.assert_allclose(result_tonga_b2[name], result_ref_b2[name], rtol=1e-2, atol=1e-3)
            np.testing.assert_allclose(result_tonga_b3[name], result_ref_b3[name], rtol=1e-2, atol=1e-3)
            np.testing.assert_allclose(result_tonga_b4[name], result_ref_b4[name], rtol=1e-2, atol=1e-3)
