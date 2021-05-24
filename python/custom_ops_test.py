import tensorflow as tf
import unittest
import numpy as np
from tensorflow.neuron.python.unittest_base import TestV2Only

def test_channels_first_simple(mode):
    neuron_result = None
    ksize = [3,3]
    strides = [1, 1]
    a = tf.range(75, dtype='float32')
   
    orig_input = tf.reshape(a, [1, 5, 5, 3])
    orig_input_transposed = tf.transpose(orig_input, perm=[0, 3, 1, 2])
    
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
    tfcpu_result_transposed = tf.transpose(tfcpu_result, perm=[0, 3, 1, 2])

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input_transposed, ksize, strides, 'VALID', data_format='NCHW')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input_transposed, ksize, strides, 'VALID', data_format='NCHW')

    np.testing.assert_allclose(neuron_result, tfcpu_result_transposed)

def test_channels_last_simple(mode):
    ksize = [3,3]
    strides = [1, 1]
    a = tf.range(75, dtype='float32')
    orig_input = tf.reshape(a, [1, 5, 5, 3])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')

    np.testing.assert_allclose(neuron_result, tfcpu_result) 

def test_channels_first_large(mode):
    ksize = [4, 7]
    strides = [1, 1]
    a = tf.range(150528, dtype='float32')
    orig_input = tf.reshape(a, [1, 224, 224, 3])
    orig_input_transposed = tf.transpose(orig_input, perm=[0, 3, 1, 2])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
    if mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
    tfcpu_result_transposed = tf.transpose(tfcpu_result, perm=[0, 3, 1, 2])

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input_transposed, ksize, strides, 'VALID', data_format='NCHW')
        if mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input_transposed, ksize, strides, 'VALID', data_format='NCHW')

    np.testing.assert_allclose(neuron_result, tfcpu_result_transposed)
    
def test_channels_last_large(mode):
    ksize = [4, 7]
    strides = [1, 1]
    a = tf.range(150528, dtype='float32')
    orig_input = tf.reshape(a, [1, 224, 224, 3])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')

    np.testing.assert_allclose(neuron_result, tfcpu_result)

def test_large_stride_and_batch_size_channels_first(mode):
    ksize = [5,5]
    strides = [4, 2]
    a = tf.range(1500, dtype='float32')
    orig_input = tf.reshape(a, [5, 10, 10, 3])
    orig_input_transposed = tf.transpose(orig_input, perm=[0, 3, 1, 2])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
    tfcpu_result_transposed = tf.transpose(tfcpu_result, perm=[0, 3, 1, 2])

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input_transposed, ksize, strides, 'VALID', data_format='NCHW')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input_transposed, ksize, strides, 'VALID', data_format='NCHW')

    np.testing.assert_allclose(neuron_result, tfcpu_result_transposed)

def test_large_stride_and_batch_size_channels_last(mode):
    ksize = [5,5]
    strides = [4, 2]
    a = tf.range(1500, dtype='float32')
    orig_input = tf.reshape(a, [5, 10, 10, 3])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input, ksize, strides, 'VALID', data_format='NHWC')

    np.testing.assert_allclose(neuron_result, tfcpu_result) 
    
def test_same_padding_channels_first(mode):
    ksize = [3,3]
    strides = [1, 1]
    a = tf.range(75, dtype='float32')
    orig_input = tf.reshape(a, [1, 5, 5, 3])
    orig_input_transposed = tf.transpose(orig_input, perm=[0, 3, 1, 2])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    tfcpu_result_transposed = tf.transpose(tfcpu_result, perm=[0, 3, 1, 2])

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')

    np.testing.assert_allclose(neuron_result, tfcpu_result_transposed) 


def test_same_padding_channels_last(mode):
    ksize = [3,3]
    strides = [1, 1]
    a = tf.range(75, dtype='float32')
    orig_input = tf.reshape(a, [1, 5, 5, 3])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')

    np.testing.assert_allclose(neuron_result, tfcpu_result) 
    
def test_same_padding_weird_kernel_and_stride_channels_first(mode):
    ksize = [5,1]
    strides = [2, 3]
    a = tf.range(75, dtype='float32')
    orig_input = tf.reshape(a, [1, 5, 5, 3])
    orig_input_transposed = tf.transpose(orig_input, perm=[0, 3, 1, 2])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    tfcpu_result_transposed = tf.transpose(tfcpu_result, perm=[0, 3, 1, 2])

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')

    np.testing.assert_allclose(neuron_result, tfcpu_result_transposed) 

    
def test_same_padding_weird_kernel_and_stride_channels_last(mode):
    ksize = [5,1]
    strides = [2, 3]
    a = tf.range(75, dtype='float32')
    orig_input = tf.reshape(a, [1, 5, 5, 3])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
        if mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    np.testing.assert_allclose(neuron_result, tfcpu_result) 

def test_same_padding_even_kernel_channels_first(mode):
    ksize = [2,4]
    strides = [1, 1]
    a = tf.range(75, dtype='float32')
    orig_input  = tf.reshape(a, [1, 5, 5, 3])
    orig_input_transposed = tf.transpose(orig_input, perm=[0, 3, 1, 2])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    tfcpu_result_transposed = tf.transpose(tfcpu_result, perm=[0, 3, 1, 2])

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')

    np.testing.assert_allclose(neuron_result, tfcpu_result_transposed) 

def test_same_padding_even_kernel_channels_last(mode):
    ksize = [2,4]
    strides = [1, 1]
    a = tf.range(75, dtype='float32')
    orig_input  = tf.reshape(a, [1, 5, 5, 3])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')

    np.testing.assert_allclose(neuron_result, tfcpu_result) 

def test_same_padding_even_kernel_stride_channels_first(mode):
    ksize = [4,6]
    strides = [3, 4]
    a = tf.range(300, dtype='float32')
    orig_input = tf.reshape(a, [1, 10, 10, 3])
    orig_input_transposed = tf.transpose(orig_input, perm=[0, 3, 1, 2])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    if mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    tfcpu_result_transposed = tf.transpose(tfcpu_result, perm=[0, 3, 1, 2])

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')

    np.testing.assert_allclose(neuron_result, tfcpu_result_transposed) 

def test_same_padding_even_kernel_stride_channels_last(mode):
    ksize = [4,6]
    strides = [3, 4]
    a = tf.range(300, dtype='float32')
    orig_input = tf.reshape(a, [1, 10, 10, 3])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    tfcpu_result_transposed = tf.transpose(tfcpu_result, perm=[0, 3, 1, 2])

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')

    np.testing.assert_allclose(neuron_result, tfcpu_result) 

def test_larger_kernel_channels_first(mode):
    ksize = [11,12]
    strides = [3, 4]
    a = tf.range(300, dtype='float32')
    orig_input = tf.reshape(a, [1, 10, 10, 3])
    orig_input_transposed = tf.transpose(orig_input, perm=[0, 3, 1, 2])
    
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    tfcpu_result_transposed = tf.transpose(tfcpu_result, perm=[0, 3, 1, 2])

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')

    np.testing.assert_allclose(neuron_result, tfcpu_result_transposed) 

def test_larger_kernel_channels_last(mode):
    ksize = [11,12]
    strides = [3, 4]
    a = tf.range(300, dtype='float32')
    orig_input = tf.reshape(a, [1, 10, 10, 3])
    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')

    np.testing.assert_allclose(neuron_result, tfcpu_result) 

def test_larger_stride_same_padding_channels_first(mode):
    ksize = [6,6]
    strides = [10, 10]
    a = tf.range(75, dtype='float32')
    orig_input = tf.reshape(a, [1, 5, 5, 3])
    orig_input_transposed = tf.transpose(orig_input, perm=[0, 3, 1, 2])

    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    
    tfcpu_result_transposed = tf.transpose(tfcpu_result, perm=[0, 3, 1, 2])

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input_transposed, ksize, strides, 'SAME', data_format='NCHW')

    np.testing.assert_allclose(neuron_result, tfcpu_result_transposed) 

def test_larger_stride_same_padding_channels_last(mode):
    ksize = [6,6]
    strides = [10, 10]
    a = tf.range(75, dtype='float32')
    orig_input = tf.reshape(a, [1, 5, 5, 3])

    if mode == 'avgpool':
        tfcpu_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
    elif mode == 'maxpool':
        tfcpu_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')

    with tf.device('/device:AWS_NEURON:0'):
        if mode == 'avgpool':
            neuron_result = tf.nn.avg_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')
        elif mode == 'maxpool':
            neuron_result = tf.nn.max_pool(orig_input, ksize, strides, 'SAME', data_format='NHWC')

    np.testing.assert_allclose(neuron_result, tfcpu_result) 

class TestAvgPool(TestV2Only):
    def test_channels_first_simple_avg_pool(self):
        test_channels_first_simple('avgpool')
        

    def test_channels_last_simple_avg_pool(self):
        test_channels_last_simple('avgpool')


    def test_channels_first_large_avg_pool(self):
        test_channels_first_large('avgpool')
       
    def test_channels_last_large_avg_pool(self):
        test_channels_last_large('avgpool')

    def test_large_stride_and_batch_size_channels_first_avg_pool(self):
        test_large_stride_and_batch_size_channels_first('avgpool')

    def test_large_stride_and_batch_size_channels_last_avg_pool(self):
        test_large_stride_and_batch_size_channels_last('avgpool')
        
    def test_same_padding_channels_first_avg_pool(self):
        test_same_padding_channels_first('avgpool')


    def test_same_padding_channels_last_avg_pool(self):
        test_same_padding_channels_last('avgpool')
        
    def test_same_padding_weird_kernel_and_stride_channels_first_avg_pool(self):
        test_same_padding_weird_kernel_and_stride_channels_first('avgpool')

        
    def test_same_padding_weird_kernel_and_stride_channels_last_avg_pool(self):
        test_same_padding_weird_kernel_and_stride_channels_last('avgpool')

    def test_same_padding_even_kernel_channels_first_avg_pool(self):
        test_same_padding_even_kernel_channels_first('avgpool')

    def test_same_padding_even_kernel_channels_last_avg_pool(self):
        test_same_padding_even_kernel_channels_last('avgpool')

    def test_same_padding_even_kernel_stride_channels_first_avg_pool(self):
        test_same_padding_even_kernel_stride_channels_first('avgpool')

    def test_same_padding_even_kernel_stride_channels_last_avg_pool(self):
        test_same_padding_even_kernel_stride_channels_last('avgpool')

    def test_larger_kernel_channels_first_avg_pool(self):
        test_larger_kernel_channels_first('avgpool')

    def test_larger_kernel_channels_last_avg_pool(self):
        test_larger_kernel_channels_last('avgpool')

    def test_larger_stride_same_padding_channels_first_avg_pool(self):
        test_larger_stride_same_padding_channels_first('avgpool')

    def test_larger_stride_same_padding_channels_last_avg_pool(self):
        test_larger_stride_same_padding_channels_last('avgpool')


class TestMaxPool(TestV2Only):
    def test_channels_first_simple_max_pool(self):
        test_channels_first_simple('maxpool')
        

    def test_channels_last_simple_max_pool(self):
        test_channels_last_simple('maxpool')


    def test_channels_first_large_max_pool(self):
        test_channels_first_large('maxpool')
       
    def test_channels_last_large_max_pool(self):
        test_channels_last_large('maxpool')

    def test_large_stride_and_batch_size_channels_first_max_pool(self):
        test_large_stride_and_batch_size_channels_first('maxpool')

    def test_large_stride_and_batch_size_channels_last_max_pool(self):
        test_large_stride_and_batch_size_channels_last('maxpool')
        
    def test_same_padding_channels_first_max_pool(self):
        test_same_padding_channels_first('maxpool')


    def test_same_padding_channels_last_max_pool(self):
        test_same_padding_channels_last('maxpool')
        
    def test_same_padding_weird_kernel_and_stride_channels_first_max_pool(self):
        test_same_padding_weird_kernel_and_stride_channels_first('maxpool')

        
    def test_same_padding_weird_kernel_and_stride_channels_last_max_pool(self):
        test_same_padding_weird_kernel_and_stride_channels_last('maxpool')

    def test_same_padding_even_kernel_channels_first_max_pool(self):
        test_same_padding_even_kernel_channels_first('maxpool')

    def test_same_padding_even_kernel_channels_last_max_pool(self):
        test_same_padding_even_kernel_channels_last('maxpool')

    def test_same_padding_even_kernel_stride_channels_first_max_pool(self):
        test_same_padding_even_kernel_stride_channels_first('maxpool')

    def test_same_padding_even_kernel_stride_channels_last_max_pool(self):
        test_same_padding_even_kernel_stride_channels_last('maxpool')

    def test_larger_kernel_channels_first_max_pool(self):
        test_larger_kernel_channels_first('maxpool')

    def test_larger_kernel_channels_last_max_pool(self):
        test_larger_kernel_channels_last('maxpool')

    def test_larger_stride_same_padding_channels_first_max_pool(self):
        test_larger_stride_same_padding_channels_first('maxpool')

    def test_larger_stride_same_padding_channels_last_max_pool(self):
        test_larger_stride_same_padding_channels_last('maxpool')