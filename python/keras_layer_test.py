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
from distutils.version import LooseVersion
import inspect
import itertools
import unittest
import tensorflow as tf
import tensorflow.neuron as tfn
from tensorflow.neuron.python.unittest_base import TestV2Only, RemoveTestSession


# Generators take mandatory arguments input_shapes and input_dtypes.
# Other arguments will be treated as keyword arguments to the generated Keras layer

class ProductGenerator(dict):
    """
    Generate tests on cartesian product of all arguments
    e. g.,
    test_0 is made of input_shapes[0] + input_dtypes[0] + kwargs[0] ...
    test_1 is made of input_shapes[0] + input_dtypes[0] + kwargs[1] ...
    test_2 is made of input_shapes[0] + input_dtypes[1] + kwargs[1] ...
    ...
    test_7 is made of input_shapes[1] + input_dtypes[1] + kwargs[1] ...
    etc.
    """

class ZipLongestGenerator(dict):
    """
    Generate tests on a "zipped" iterator of all arguments
    e. g.,
    test_0 is made of input_shapes[0] + input_dtypes[0] + kwargs[0] ...
    test_1 is made of input_shapes[1] + input_dtypes[1] + kwargs[1] ...
    etc.
    This mode is useful for cases where a blind cartesian product of arguments
    may lead to incompatible arguments. For example, the "Concatenate" layer
    with input_shapes=[(1, 5, 3), (1, 8, 3)] can only work with axis=1.
    """


class KerasLayerGenerator(RemoveTestSession):
    """
    Test generator engine metaclass
    """

    def __new__(mcs, name, bases, dct):

        def legalize(string):
            return ''.join(ch for ch in string if ch.isalpha() or ch.isnumeric() or ch in '_.-')

        def get_input(shape, dtype):
            if isinstance(dtype, tuple):
                return tuple(get_input(sp, dt) for sp, dt in zip(shape, dtype))
            if isinstance(shape[0], int):
                return random(shape, dtype)
            else:
                return [random(sp, dtype) for sp in shape]

        def random(shape, dtype=tf.float32):
            rand_dtype = dtype
            if dtype.is_floating:
                min_val, max_val = -0.1, 0.1
            elif dtype.is_integer:
                min_val, max_val = 0, 100
            elif dtype.is_bool:
                min_val, max_val = -0.1, 0.1
                rand_dtype = tf.float32
            output = tf.random.uniform(shape, min_val, max_val, dtype=rand_dtype)
            if dtype.is_bool:
                output = output > 0.0
            return output

        def gen_test(input_shape, input_dtype, layer_type, layer_kwargs):
            def test(self):
                example_inputs = get_input(input_shape, input_dtype)
                layer = layer_type(**layer_kwargs)
                if layer_kwargs.get('data_format', None) == 'channels_first':
                    example_inputs_channels_last = example_inputs
                    perm_last_to_first = list(range(example_inputs.shape.ndims))
                    perm_last_to_first.insert(1, perm_last_to_first.pop())
                    example_inputs = tf.transpose(example_inputs_channels_last, perm_last_to_first)
                inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
                try:
                    output = layer(*inputs)
                except (tf.errors.UnimplementedError, tf.errors.InvalidArgumentError):
                    layer_kwargs.update(data_format='channels_last')
                    layer_channels_last = layer_type(**layer_kwargs)
                    # run a dummy inference to initialize and assign weights
                    layer_channels_last(example_inputs_channels_last)
                    for dst, src in zip(layer_channels_last.weights, layer.weights):
                        dst.assign(src)
                    # re-run inference after assigning weights
                    output_channels_last = layer_channels_last(example_inputs_channels_last)
                    output = tf.transpose(output_channels_last, perm_last_to_first)
                print('layer config:', layer.get_config())
                print('output shape:', output.shape)
                layer_neuron = tfn.trace(layer, example_inputs)
                # assert everything is on Neuron
                graph = layer_neuron.aws_neuron_function.python_function.graph
                op_type_set = {op.type for op in graph.get_operations()}
                assert op_type_set == {'Placeholder', 'IdentityN', 'NeuronOp'}
                output_neuron = layer_neuron(*inputs)
                self.assertAllClose(output_neuron, output, rtol=1e-2, atol=1e-2)
            return test

        empty_default = inspect.signature(lambda x: None).parameters['x'].default
        name_format = 'test_{}-case{}-{}-[{}]'.format
        gen_method_map = {
            ProductGenerator: itertools.product,
            ZipLongestGenerator: itertools.zip_longest,
        }
        skip_layer_names = not_implemented_layer_names()
        for layer_name, layer_gen in get_layer_generators().items():
            if layer_gen is None:
                continue
            layer_type = getattr(tf.keras.layers, layer_name)
            params = inspect.signature(layer_type.__init__).parameters
            allowed_layer_kwargs = layer_gen.copy()
            input_shapes = allowed_layer_kwargs.pop('input_shapes')
            input_dtypes = allowed_layer_kwargs.pop('input_dtypes')
            skipper = allowed_layer_kwargs.pop('skipper', lambda kwargs: False)
            for key, value in params.items():
                value_default = value.default
                if value_default is not empty_default and key not in allowed_layer_kwargs:
                    allowed_args = [value_default]
                    if type(value_default) is bool:
                        allowed_args.append(not value_default)
                    allowed_layer_kwargs[key] = allowed_args
                data_format_candidates = ['channels_last', 'channels_first']
                if key == 'data_format' and value_default in {None, *data_format_candidates}:
                    allowed_layer_kwargs[key] = data_format_candidates
            keys = list(allowed_layer_kwargs.keys())
            allowed_args = [allowed_layer_kwargs[key] for key in keys]
            gen_method = gen_method_map[type(layer_gen)]
            iterator = gen_method(input_shapes, input_dtypes, *allowed_args)
            for idx, (in_shape, in_dtype, *args) in enumerate(iterator):
                if isinstance(in_dtype, tuple):
                    in_dtype_repr = ','.join(dtype.name for dtype in in_dtype)
                else:
                    in_dtype_repr = in_dtype.name
                args_str = ','.join(legalize(value) for value in str(args).split(','))
                test_name = name_format(layer_name, idx, in_dtype_repr, args_str)
                layer_kwargs = dict(zip(keys, args))
                if skipper(layer_kwargs):
                    continue
                test_func = gen_test(in_shape, in_dtype, layer_type, layer_kwargs)
                if layer_name in skip_layer_names:
                    test_func = unittest.skip('Not implemented')(test_func)
                dct[test_name] = test_func
        return RemoveTestSession.__new__(mcs, name, bases, dct)


def get_layer_generators():
    """
    All Keras layer information is defined here.
    """
    float_types = [tf.float32, tf.float16]
    activations = [
        'elu', 'exponential', 'hard_sigmoid', 'relu',
        'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh',
    ]

    def skip_strides_and_dilation_rate(layer_kwargs):
        return layer_kwargs['strides'] > 1 and layer_kwargs['dilation_rate'] > 1

    def skip_pool_3d(layer_kwargs):
        # 'same' triggers 'Assertion 'tensortensorWaveop->gSrcAXNumJSON() == tensortensorWaveop->gSrcBXNumJSON()' failed: When tensors for TensorTensor are in the same buffer, X count must be equal'
        if layer_kwargs['padding'] == 'same':
            return True
        # 3 triggers 'SG-ERR: Runtime exception <Non-output memory location with no reader {arg0.1_transpose_10-t46_i7}@SB<64,3840>(8x512)#Internal DebugInfo: <arg0.1_transpose_10-t46_i7||float32||UNDEF||[8, 128, 1]>>'
        return layer_kwargs['pool_size'] == 3 and layer_kwargs['data_format'] == 'channels_last'

    # some reusable generators
    reduce_gen = ProductGenerator(
        input_shapes=[[(1, 3, 32), (1, 3, 32), (1, 3, 32)]],
        input_dtypes=float_types,
    )
    attention_gen = ProductGenerator(
        input_shapes=[([(1, 8, 32), (1, 16, 32), (1, 16, 32)], [(1, 8), (1, 16)])],
        input_dtypes=[(tf.float32, tf.bool), (tf.float16, tf.bool)],
        causal=[False, True],
    )
    pooling_gen_common = dict(
        input_dtypes=[tf.float32],
        pool_size=[2, 3],
        strides=[1, 2],
        padding=['valid', 'same'],
    )
    pool1d_gen = ProductGenerator(
        input_shapes=[(1, 28, 16)],
        **pooling_gen_common,
    )
    pool2d_gen = ProductGenerator(
        input_shapes=[(1, 14, 14, 16)],
        **pooling_gen_common,
    )
    pool3d_gen = ProductGenerator(
        input_shapes=[(1, 10, 10, 10, 8)],
        **pooling_gen_common,
        skipper=skip_pool_3d,
    )
    normalization_gen = ProductGenerator(
        input_shapes=[(1, 8, 8, 32)],
        input_dtypes=[tf.float16],
        axis=[-1, 1],
    )
    conv1d_gen = ProductGenerator(
        input_shapes=[(1, 28, 3), (1, 20, 32)],
        input_dtypes=float_types,
        filters=[16],
        kernel_size=[1, 3],
        strides=[1, 2],
    )
    conv2d_gen = ProductGenerator(
        input_shapes=[(1, 28, 28, 3), (1, 9, 9, 32)],
        input_dtypes=[tf.float32],
        filters=[16],
        kernel_size=[(1, 1), (2, 2), (3, 3)],
        strides=[1, 2],
        padding=['valid', 'same'],
        dilation_rate=[1, 2],
        use_bias=[False],
        skipper=skip_strides_and_dilation_rate,
    )
    conv3d_gen = ProductGenerator(
        input_shapes=[(1, 8, 8, 8, 4)],
        input_dtypes=[tf.float32],
        filters=[16],
        kernel_size=[(1, 1, 1), (3, 3, 3)],
        strides=[1, 2],
        padding=['valid'],
        use_bias=[False],
    )
    global_pooling_1d_gen = ProductGenerator(
        input_shapes=[(1, 16, 8)],
        input_dtypes=float_types,
    )
    global_pooling_2d_gen = ProductGenerator(
        input_shapes=[(1, 16, 16, 8)],
        input_dtypes=float_types,
    )
    global_pooling_3d_gen = ProductGenerator(
        input_shapes=[(1, 6, 6, 6, 6)],
        input_dtypes=float_types,
    )

    # define all generators here
    layer_generators = dict(
        AbstractRNNCell=None,
        Activation=ProductGenerator(
            input_shapes=[(1, 20, 20, 32)],
            input_dtypes=[tf.float32],
            activation=activations,
        ),
        ActivityRegularization=None,
        Add=reduce_gen,
        AdditiveAttention=attention_gen,
        AlphaDropout=None,
        Attention=attention_gen,
        Average=reduce_gen,
        AveragePooling1D=ProductGenerator(
            input_shapes=[(1, 28, 16)],
            input_dtypes=[tf.float32],
            pool_size=[2, 3],
            strides=[1, 2],
            padding=['valid'],
        ),
        AveragePooling2D=ProductGenerator(
            input_shapes=[(1, 14, 14, 16)],
            input_dtypes=[tf.float32],
            pool_size=[2, 3],
            strides=[1, 2],
            padding=['valid'],  # 'same' triggers 'neuroncc/starfish/penguin/ir/Value.py:40: AssertionError'
        ),
        AveragePooling3D=pool3d_gen,
        BatchNormalization=normalization_gen,
        Bidirectional=None,
        Concatenate=ZipLongestGenerator(
            input_shapes=[[(1, 2, 2, 3), (1, 2, 2, 3), (1, 2, 2, 3)], [(1, 5, 3), (1, 8, 3)]],
            input_dtypes=[tf.float32, tf.float16],
            axis=[-1, 1],
        ),
        Conv1D=conv1d_gen,
        Conv2D=conv2d_gen,
        Conv2DTranspose=conv2d_gen,
        Conv3D=conv3d_gen,
        Conv3DTranspose=conv3d_gen,
        ConvLSTM2D=None,
        Cropping1D=ProductGenerator(
            input_shapes=[(1, 32, 16)],
            input_dtypes=float_types,
            cropping=[1],
        ),
        Cropping2D=ProductGenerator(
            input_shapes=[(1, 32, 32, 16)],
            input_dtypes=float_types,
            cropping=[1, ((1, 2), (3, 4))],
        ),
        Cropping3D=ProductGenerator(
            input_shapes=[(1, 10, 10, 10, 8)],
            input_dtypes=float_types,
            cropping=[1, ((1, 2), (2, 1), (1, 3))],
        ),
        Dense=ProductGenerator(
            input_shapes=[(1, 64)],
            input_dtypes=float_types,
            units=[32],
        ),
        DenseFeatures=None,
        DepthwiseConv2D=ProductGenerator(
            input_shapes=[(1, 28, 28, 3), (1, 20, 20, 32)],
            input_dtypes=[tf.float32],
            kernel_size=[(1, 1), (3, 3)],
            strides=[1, 2],
            depth_multiplier=[1, 2],
            use_bias=[False],
        ),
        Dot=ZipLongestGenerator(
            input_shapes=[[(1, 5, 2), (1, 2, 5)], [(5, 2), (5, 2)]],
            input_dtypes=[tf.float32, tf.float16],
            axes=[(1, 2), 1],
        ),
        Dropout=None,
        ELU=ProductGenerator(
            input_shapes=[(1, 8, 4)],
            input_dtypes=float_types,
            alpha=[1.0, 0.5],
        ),
        Embedding=ProductGenerator(
            input_shapes=[(2, 3)],
            input_dtypes=[tf.int32],
            input_dim=[1000],
            output_dim=[64],
            input_length=[10],
        ),
        Flatten=ProductGenerator(
            input_shapes=[(1, 2, 3)],
            input_dtypes=float_types,
        ),
        GRU=None,
        GRUCell=None,
        GaussianDropout=None,
        GaussianNoise=None,
        GlobalAveragePooling1D=global_pooling_1d_gen,
        GlobalAveragePooling2D=global_pooling_2d_gen,
        GlobalAveragePooling3D=global_pooling_3d_gen,
        GlobalMaxPool1D=global_pooling_1d_gen,
        GlobalMaxPool2D=global_pooling_2d_gen,
        GlobalMaxPool3D=global_pooling_3d_gen,
        InputLayer=None,
        InputSpec=None,
        LSTM=None,
        LSTMCell=None,
        Lambda=None,
        Layer=None,
        LayerNormalization=normalization_gen,
        LeakyReLU=ProductGenerator(
            input_shapes=[(1, 8, 8, 6)],
            input_dtypes=float_types,
        ),
        LocallyConnected1D=ProductGenerator(
            input_shapes=[(1, 8, 6)],
            input_dtypes=float_types,
            filters=[4],
            kernel_size=[1, 3],
            strides=[1, 2],
        ),
        LocallyConnected2D=ProductGenerator(
            input_shapes=[(1, 8, 8, 6)],
            input_dtypes=[tf.float32],
            filters=[4],
            kernel_size=[(1, 1), (3, 3)],
            strides=[1, 2],
        ),
        Masking=ProductGenerator(
            input_shapes=[(1, 8, 32)],
            input_dtypes=float_types,
            mask_value=[0.0, 0.5],
        ),
        MaxPool1D=pool1d_gen,
        MaxPool2D=pool2d_gen,
        MaxPool3D=pool3d_gen,
        Maximum=reduce_gen,
        Minimum=reduce_gen,
        Multiply=reduce_gen,
        PReLU=ProductGenerator(
            input_shapes=[(1, 8, 8, 6)],
            input_dtypes=float_types,
            shared_axes=[None, [1, 2]]
        ),
        Permute=ProductGenerator(
            input_shapes=[(1, 8, 10, 6)],
            input_dtypes=float_types,
            dims=[(1, 3, 2), (3, 1, 2)]
        ),
        RNN=None,
        ReLU=ProductGenerator(
            input_shapes=[(1, 8, 10, 6)],
            input_dtypes=float_types,
            max_value=[2.0],
            negative_slope=[0.0, 0.1],
            threshold=[0.0, 3.0],
        ),
        RepeatVector=ProductGenerator(
            input_shapes=[(1, 8)],
            input_dtypes=float_types,
            n=[3],
        ),
        Reshape=ZipLongestGenerator(
            input_shapes=[(1, 8), (1, 8, 6)],
            input_dtypes=[tf.float32, tf.float16],
            target_shape=[(2, 4), (6, 8)],
        ),
        SeparableConv1D=ProductGenerator(
            input_shapes=[(1, 20, 32)],
            input_dtypes=[tf.float16],
            filters=[16],
            kernel_size=[3],
            strides=[1, 2],
            depth_multiplier=[1, 2],
            use_bias=[False],
        ),
        SeparableConv2D=ProductGenerator(
            input_shapes=[(1, 20, 20, 32)],
            input_dtypes=[tf.float32],
            filters=[16],
            kernel_size=[(3, 3)],
            strides=[1, 2],
            depth_multiplier=[1, 2],
            use_bias=[False],
        ),
        SimpleRNN=None,
        SimpleRNNCell=None,
        Softmax=ProductGenerator(
            input_shapes=[(1, 20, 20, 32)],
            input_dtypes=[tf.float32],
            axis=[-1, 1],
        ),
        SpatialDropout1D=None,
        SpatialDropout2D=None,
        SpatialDropout3D=None,
        StackedRNNCells=None,
        Subtract=ProductGenerator(
            input_shapes=[[(1, 3, 32), (1, 3, 32)]],
            input_dtypes=float_types,
        ),
        ThresholdedReLU=ProductGenerator(
            input_shapes=[(1, 8, 10, 6)],
            input_dtypes=float_types,
            theta=[1.0, 0.5],
        ),
        TimeDistributed=None,
        UpSampling1D=ProductGenerator(
            input_shapes=[(1, 20, 32)],
            input_dtypes=[tf.float32],
            size=[1, 2],
        ),
        UpSampling2D=ProductGenerator(
            input_shapes=[(1, 20, 20, 32)],
            input_dtypes=[tf.float32],
            size=[(2, 1), (2, 2)],
            interpolation=['nearest', 'bilinear'],
        ),
        UpSampling3D=ProductGenerator(
            input_shapes=[(1, 8, 8, 8, 8)],
            input_dtypes=[tf.float32],
            size=[(2, 2, 2)],
        ),
        Wrapper=None,
        ZeroPadding1D=ProductGenerator(
            input_shapes=[(1, 8, 6)],
            input_dtypes=float_types,
            padding=[1, 2],
        ),
        ZeroPadding2D=ProductGenerator(
            input_shapes=[(1, 8, 8, 8)],
            input_dtypes=float_types,
            padding=[1, 2],
        ),
        ZeroPadding3D=ProductGenerator(
            input_shapes=[(1, 8, 8, 8, 6)],
            input_dtypes=float_types,
            padding=[1, 2],
        ),
    )

    # update with some newly introduced layers/activations
    if LooseVersion(tf.__version__) >= LooseVersion('2.2.0'):
        activations.append('swish')
    if LooseVersion(tf.__version__) >= LooseVersion('2.3.0'):
        layer_generators.update(
            Conv1DTranspose=conv1d_gen,
        )
    if LooseVersion(tf.__version__) >= LooseVersion('2.4.0'):
        activations.append('gelu')
        layer_generators.update(
            MultiHeadAttention=ProductGenerator(
                input_shapes=[((1, 8, 16), (1, 4, 16))],
                input_dtypes=[(tf.float32, tf.float32), (tf.float16, tf.float16)],
                num_heads=[2],
                key_dim=[2],
            ),
        )

    # sort all layer types alphabetically
    return dict(sorted(layer_generators.items()))


def not_implemented_layer_names():
    layer_names = {
        'Embedding',
        'UpSampling2D',
    }
    return layer_names


class TestKerasLayer(TestV2Only, metaclass=KerasLayerGenerator):
    """
    Placeholder class for generated tests
    """
