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
import tensorflow as tf
from tensorflow.compat import v1 as tfv1
from tensorflow.neuron.python import graph_def_util as gdu
from tensorflow.neuron.python.custom_call import CustomCallLowering
from tensorflow.neuron.python.neuron_cc_hlo import graph_def_to_hlo, tf2xla_pb2
import numpy as np
from tensorflow.neuron.python.unittest_base import TestV2Only
from tensorflow.neuron.python.ops.gen_neuron_op import check_runtime_op


def run_channels_first_simple(mode):
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

def run_channels_last_simple(mode):
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

def run_channels_first_large(mode):
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
    
def run_channels_last_large(mode):
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

def run_large_stride_and_batch_size_channels_first(mode):
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

def run_large_stride_and_batch_size_channels_last(mode):
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
    
def run_same_padding_channels_first(mode):
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


def run_same_padding_channels_last(mode):
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
    
def run_same_padding_weird_kernel_and_stride_channels_first(mode):
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

    
def run_same_padding_weird_kernel_and_stride_channels_last(mode):
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

def run_same_padding_even_kernel_channels_first(mode):
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

def run_same_padding_even_kernel_channels_last(mode):
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

def run_same_padding_even_kernel_stride_channels_first(mode):
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

def run_same_padding_even_kernel_stride_channels_last(mode):
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

def run_larger_kernel_channels_first(mode):
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

def run_larger_kernel_channels_last(mode):
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

def run_larger_stride_same_padding_channels_first(mode):
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

def run_larger_stride_same_padding_channels_last(mode):
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
        run_channels_first_simple('avgpool')
        

    def test_channels_last_simple_avg_pool(self):
        run_channels_last_simple('avgpool')

    def test_channels_first_large_avg_pool(self):
        run_channels_first_large('avgpool')
       
    def test_channels_last_large_avg_pool(self):
        run_channels_last_large('avgpool')

    def test_large_stride_and_batch_size_channels_first_avg_pool(self):
        run_large_stride_and_batch_size_channels_first('avgpool')

    def test_large_stride_and_batch_size_channels_last_avg_pool(self):
        run_large_stride_and_batch_size_channels_last('avgpool')
        
    def test_same_padding_channels_first_avg_pool(self):
        run_same_padding_channels_first('avgpool')


    def test_same_padding_channels_last_avg_pool(self):
        run_same_padding_channels_last('avgpool')
        
    def test_same_padding_weird_kernel_and_stride_channels_first_avg_pool(self):
        run_same_padding_weird_kernel_and_stride_channels_first('avgpool')

        
    def test_same_padding_weird_kernel_and_stride_channels_last_avg_pool(self):
        run_same_padding_weird_kernel_and_stride_channels_last('avgpool')

    def test_same_padding_even_kernel_channels_first_avg_pool(self):
        run_same_padding_even_kernel_channels_first('avgpool')

    def test_same_padding_even_kernel_channels_last_avg_pool(self):
        run_same_padding_even_kernel_channels_last('avgpool')

    def test_same_padding_even_kernel_stride_channels_first_avg_pool(self):
        run_same_padding_even_kernel_stride_channels_first('avgpool')

    def test_same_padding_even_kernel_stride_channels_last_avg_pool(self):
        run_same_padding_even_kernel_stride_channels_last('avgpool')

    def test_larger_kernel_channels_first_avg_pool(self):
        run_larger_kernel_channels_first('avgpool')

    def test_larger_kernel_channels_last_avg_pool(self):
        run_larger_kernel_channels_last('avgpool')

    def test_larger_stride_same_padding_channels_first_avg_pool(self):
        run_larger_stride_same_padding_channels_first('avgpool')

    def test_larger_stride_same_padding_channels_last_avg_pool(self):
        run_larger_stride_same_padding_channels_last('avgpool')


class TestMaxPool(TestV2Only):
    def test_channels_first_simple_max_pool(self):
        run_channels_first_simple('maxpool')
        

    def test_channels_last_simple_max_pool(self):
        run_channels_last_simple('maxpool')


    def test_channels_first_large_max_pool(self):
        run_channels_first_large('maxpool')
       
    def test_channels_last_large_max_pool(self):
        run_channels_last_large('maxpool')

    def test_large_stride_and_batch_size_channels_first_max_pool(self):
        run_large_stride_and_batch_size_channels_first('maxpool')

    def test_large_stride_and_batch_size_channels_last_max_pool(self):
        run_large_stride_and_batch_size_channels_last('maxpool')
        
    def test_same_padding_channels_first_max_pool(self):
        run_same_padding_channels_first('maxpool')


    def test_same_padding_channels_last_max_pool(self):
        run_same_padding_channels_last('maxpool')
        
    def test_same_padding_weird_kernel_and_stride_channels_first_max_pool(self):
        run_same_padding_weird_kernel_and_stride_channels_first('maxpool')

        
    def test_same_padding_weird_kernel_and_stride_channels_last_max_pool(self):
        run_same_padding_weird_kernel_and_stride_channels_last('maxpool')

    def test_same_padding_even_kernel_channels_first_max_pool(self):
        run_same_padding_even_kernel_channels_first('maxpool')

    def test_same_padding_even_kernel_channels_last_max_pool(self):
        run_same_padding_even_kernel_channels_last('maxpool')

    def test_same_padding_even_kernel_stride_channels_first_max_pool(self):
        run_same_padding_even_kernel_stride_channels_first('maxpool')

    def test_same_padding_even_kernel_stride_channels_last_max_pool(self):
        run_same_padding_even_kernel_stride_channels_last('maxpool')

    def test_larger_kernel_channels_first_max_pool(self):
        run_larger_kernel_channels_first('maxpool')

    def test_larger_kernel_channels_last_max_pool(self):
        run_larger_kernel_channels_last('maxpool')

    def test_larger_stride_same_padding_channels_first_max_pool(self):
        run_larger_stride_same_padding_channels_first('maxpool')

    def test_larger_stride_same_padding_channels_last_max_pool(self):
        run_larger_stride_same_padding_channels_last('maxpool')


class TestCustomOp(TestV2Only):
    def test_checkruntime(self):
        print(dir(test))

    def test_aws_neuron_erf(self):
        self._tf2hlo_test_unary_op('AwsNeuronErf')

    def test_erf_custom_call_lowering(self):
        self._custom_call_lowering_test_unary_op(tfv1.math.erf, 'AwsNeuronErf')

    def test_aws_neuron_softplus(self):
        self._tf2hlo_test_unary_op('AwsNeuronSoftplus')

    def test_softplus_custom_call_lowering(self):
        self._custom_call_lowering_test_unary_op(tfv1.math.softplus, 'AwsNeuronSoftplus')

    def _tf2hlo_test_unary_op(self, call_target):
        graph = tfv1.Graph()
        with graph.as_default():
            ph = tfv1.placeholder(tfv1.float32, [1, 1])
            mid = tfv1.identity(ph)
            out = tfv1.identity(mid)
        graph_def = graph.as_graph_def()
        node_def = graph_def.node[1]
        node_def.op = '_AwsNeuronCustomOp'
        node_def.attr.clear()
        node_def.attr['custom_call_target'].s = call_target.encode()
        node_def.attr['input_dtypes'].list.type.append(ph.dtype.as_datatype_enum)
        node_def.attr['output_dtypes'].list.type.append(out.dtype.as_datatype_enum)
        shape = node_def.attr['output_shapes'].list.shape.add()
        for size in ph.shape:
            shape.dim.add().size = size

        tf2xla_config = tf2xla_pb2.Config()
        inp0 = tf2xla_config.feed.add()
        inp0.id.node_name = ph.op.name
        inp0.id.output_index = 0
        inp0.shape.CopyFrom(ph.shape.as_proto())
        inp0.type = ph.dtype.as_datatype_enum
        out0 = tf2xla_config.fetch.add()
        out0.id.node_name = out.op.name
        out0.id.output_index = 0
        out0.shape.CopyFrom(out.shape.as_proto())
        out0.type = out.dtype.as_datatype_enum

        hlo_module = graph_def_to_hlo(graph_def, tf2xla_config)
        inst = hlo_module.computations[0].instructions[2]
        self.assertEqual(inst.opcode, 'custom-call')
        self.assertEqual(inst.custom_call_target, call_target)
        self.assertIn(inst.backend_config, {'', b''})

    def _custom_call_lowering_test_unary_op(self, func, call_target):
        graph = tfv1.Graph()
        with graph.as_default():
            ph = tfv1.placeholder(tfv1.float32, [1, 1])
            mid = func(ph)
            out = tfv1.identity(mid)
        graph_def = graph.as_graph_def()
        node_def = graph_def.node[1]
        node_def.attr[gdu.kNeuronInferredShapes].list.shape.add().CopyFrom(mid.shape.as_proto())
        custom_call_lowering = CustomCallLowering()
        graph_def = custom_call_lowering.lower(graph_def)
        node_def = graph_def.node[1]
        self.assertEqual(node_def.op, '_AwsNeuronCustomOp')
        self.assertEqual(node_def.attr['custom_call_target'].s, call_target.encode())
        self.assertEqual(node_def.attr['input_dtypes'].list.type, [ph.dtype.as_datatype_enum])
        self.assertEqual(node_def.attr['output_dtypes'].list.type, [out.dtype.as_datatype_enum])
        self.assertEqual(len(node_def.attr['output_shapes'].list.shape), 1)
        self.assertEqual(node_def.attr['output_shapes'].list.shape[0], out.shape.as_proto())
        graph_def = custom_call_lowering.restore(graph_def)
        node_def = graph_def.node[1]
        node_def.attr.pop(gdu.kNeuronInferredShapes)
        self.assertEqual(graph_def.node[1], mid.op.node_def)
