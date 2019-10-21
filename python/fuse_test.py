"""
Copyright (C) 2019, Amazon.com. All Rights Reserved
"""
import os
import sys
import subprocess
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.neuron import fuse


_RANDOM_SEED = 15213


def network_body(input0, input1, kernel0, kernel1):
    conv2d0 = tf.nn.conv2d(input0, kernel0,
                           strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
    conv2d1 = tf.nn.conv2d(input1, kernel1,
                           strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
    add0 = tf.add(conv2d0, conv2d1, name='add0')
    relu0 = tf.nn.relu(add0, name='relu0')
    sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
    return relu0, sigmoid0


def network_cpu(input0, input1, kernel0, kernel1):
    return network_body(input0, input1, kernel0, kernel1)


@fuse
def network_tonga(input0, input1, kernel0, kernel1):
    return network_body(input0, input1, kernel0, kernel1)


def test_fuse_simple():
    np.random.seed(_RANDOM_SEED)
    kernel0 = np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16)
    kernel1 = np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16)
    feed_dict = {
        'input0:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        'input1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
    }

    with tf.Session(graph=tf.Graph()) as sess:
        input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
        input1 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input1')
        relu0, sigmoid0 = network_cpu(input0, input1, kernel0, kernel1)
        result_ref = sess.run([relu0, sigmoid0], feed_dict)

    if 'KAENA_KRTD_SERVER_ADDRESS' in os.environ:
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input1')
            relu0, sigmoid0 = network_tonga(input0, input1, kernel0, kernel1)
            result_tonga = sess.run([relu0, sigmoid0], feed_dict)
        for res_tonga, res_ref in zip(result_tonga, result_ref):
            np.testing.assert_allclose(res_tonga, res_ref, rtol=1e-2, atol=1e-3)


@pytest.mark.xfail(run=False, reason='Running this test together with others requires 2 TPBs')
def test_fuse_eager_execution():
    assert subprocess.run([
        sys.executable, '-c', 'from tensorflow.python.neuron.python import fuse_test;'
                              'fuse_test.actualtest_fuse_eager_execution()'
    ]).returncode == 0


def actualtest_fuse_eager_execution():
    np.random.seed(_RANDOM_SEED)
    tf.enable_eager_execution()
    kernel0 = np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16)
    kernel1 = np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16)
    input0 = tf.constant(np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16))
    input1 = tf.constant(np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16))
    result_ref = network_cpu(input0, input1, kernel0, kernel1)

    if 'KAENA_KRTD_SERVER_ADDRESS' in os.environ:
        result_tonga = network_tonga(input0, input1, kernel0, kernel1)
        for res_tonga, res_ref in zip(result_tonga, result_ref):
            np.testing.assert_allclose(res_tonga.numpy(), res_ref.numpy(), rtol=1e-2, atol=1e-3)
