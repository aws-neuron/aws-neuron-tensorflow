"""
Copyright (C) 2019, Amazon.com. All Rights Reserved
"""
import os
import sys
import subprocess
import unittest
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

@fuse
def network_neuron(input0, input1, kernel0, kernel1):
    return network_body(input0, input1, kernel0, kernel1)


class TestFuse(unittest.TestCase):

    def test_simple(self):
        relu0, sigmoid0 = self._body_fuse(1)
        assert relu0.shape.as_list() == [1, 2, 2, 3]
        assert sigmoid0.shape.as_list() == [1, 2, 2, 3]

    def test_dynamic_batch_size(self):
        relu0, sigmoid0 = self._body_fuse(None)
        assert relu0.shape.as_list() == [None, 2, 2, 3]
        assert sigmoid0.shape.as_list() == [None, 2, 2, 3]

    def _body_fuse(self, batch_size):
        np.random.seed(_RANDOM_SEED)
        kernel0 = np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16)
        kernel1 = np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16)
        feed_dict = {
            'input0:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            'input1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        }

        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [batch_size, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [batch_size, 2, 2, 3], name='input1')
            relu0, sigmoid0 = network_body(input0, input1, kernel0, kernel1)
            result_ref = sess.run([relu0, sigmoid0], feed_dict)

        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [batch_size, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [batch_size, 2, 2, 3], name='input1')
            fuser = fuse
            if batch_size is None:
                io_shapes = [[1, 2, 2, 3], [1, 2, 2, 3]]
                fuser = fuse(input_shapes=io_shapes, output_shapes=io_shapes, dynamic_batch_size=True)
            relu0, sigmoid0 = fuser(network_body)(input0, input1, kernel0, kernel1)
            if 'NEURON_RTD_ADDRESS' in os.environ:
                result_neuron = sess.run([relu0, sigmoid0], feed_dict)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)
            return relu0, sigmoid0

    def test_nested_input_output(test):
        def nested_input_output(list_list_tuple, tuple_list):
            an0 = list_list_tuple[0][0][0]
            an1 = list_list_tuple[0][0][1]
            bx0 = list_list_tuple[1][0][0]
            bx1 = list_list_tuple[1][0][1]
            out0 = an0 + bx1
            out1 = an1 + bx0
            by0 = list_list_tuple[1][1][0]
            by1 = list_list_tuple[1][1][1]
            out2 = by0 + by1
            bz0 = list_list_tuple[1][2][0]
            bz1 = list_list_tuple[1][2][1]
            out3 = bz0 + bz1
            tuple23 = out2, out3
            pa0 = tuple_list[0][0]
            qa0 = tuple_list[1][0]
            return out0, (out1, tuple23, pa0 + qa0)
        def get_inputs(func):
            return (
                [
                    [(func([1]), func([1]))],
                    [
                        (func([1]), func([1])),
                        (func([1]), func([1])),
                        (func([1]), func([1])),
                    ]
                ],
                ([func([1])], [func([1])],),
            )
        def get_placeholders():
            ph_list = []
            def ph(shape):
                placeholder = tf.placeholder(tf.float32, shape)
                ph_list.append(placeholder)
                return placeholder
            inputs = get_inputs(ph)
            return ph_list, inputs
        inputs_np = get_inputs(np.ones)
        with tf.Session(graph=tf.Graph()) as sess:
            ph_list, inputs = get_placeholders()
            outputs = nested_input_output(*inputs)
            feed_dict = {ph: idx * np.ones([1]) for idx, ph in enumerate(ph_list)}
            result_ref = sess.run(outputs, feed_dict)
        with tf.Session(graph=tf.Graph()) as sess:
            ph_list, inputs = get_placeholders()
            outputs = fuse(nested_input_output)(*inputs)
            feed_dict = {ph: idx * np.ones([1]) for idx, ph in enumerate(ph_list)}
            if 'NEURON_RTD_ADDRESS' in os.environ:
                result_neuron = sess.run(outputs, feed_dict)
                result_ref_unpacked = _unpack_recursive(result_ref)
                result_neuron_unpacked = _unpack_recursive(result_neuron)
                for res_neuron, res_ref in zip(result_ref_unpacked, result_neuron_unpacked):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2)

    @unittest.skipIf('--runxfail' not in sys.argv,
                     'Running this test together with others requires 2 neuron cores')
    def test_fuse_eager_execution(self):
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
    result_ref = network_body(input0, input1, kernel0, kernel1)

    if 'NEURON_RTD_ADDRESS' in os.environ:
        result_neuron = network_neuron(input0, input1, kernel0, kernel1)
        for res_neuron, res_ref in zip(result_neuron, result_ref):
            np.testing.assert_allclose(res_neuron.numpy(), res_ref.numpy(), rtol=1e-2, atol=1e-3)


def _unpack_recursive(outputs):
    while any(isinstance(out, (tuple, list)) for out in outputs):
        unpacked = []
        for out in outputs:
            func = unpacked.extend if isinstance(out, (tuple, list)) else unpacked.append
            func(out)
        outputs = unpacked
    return outputs
