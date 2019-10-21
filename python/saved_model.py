"""
Copyright (C) 2019, Amazon.com. All Rights Reserved
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import json
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.saved_model import saved_model as tf_saved_model
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.neuron.python.graph_util import inference_graph_from_session


@tf_export('neuron.saved_model.simple_save')
def simple_save(sess, export_dir, inputs, outputs, batch_size=1, **kwargs):
    """Convenience function to build a `SavedModel` suitable for serving.
    Args:
        sess: The TensorFlow session from which to save the meta graph and variables.
        export_dir: The path to which the `SavedModel` will be stored.
        inputs: dict mapping string input names to tensors. These are added
            to the `SignatureDef` as the inputs.
        outputs:  dict mapping string output names to tensors. These are added
            to the `SignatureDef` as the outputs.
        batch_size: (Optional) Batch size used in inference.
    Note: This function shares all arguments with `inference_graph_from_session`.
    """
    _check_export_dir(export_dir)
    # if `feed_dict` is not given, try to guess a `shape_feed_dict` from `batch_size`
    if 'shape_feed_dict' not in kwargs and 'feed_dict' not in kwargs:
        shape_feed_dict = _infer_input_shapes(inputs.values(), batch_size)
        kwargs.update(shape_feed_dict=shape_feed_dict)
    infer_graph = inference_graph_from_session(
        sess, input_tensors=inputs.values(), output_tensors=outputs.values(),
        **kwargs)

    # load inference graph into a session and export as a SavedModel
    with session.Session(graph=infer_graph) as sess:
        inputs = {key: sess.graph.get_tensor_by_name(ts.name) for key, ts in inputs.items()}
        outputs = {key: sess.graph.get_tensor_by_name(ts.name) for key, ts in outputs.items()}
        tf_saved_model.simple_save(sess, export_dir, inputs, outputs)


def _infer_input_shapes(input_tensors, batch_size):
    """Infer/guess the shape of the inputs using batch_size.
    Args:
        input_tensors: Iterable of input tensors
        batch_size: Positive integer; batch size used to infer input tensor shapes
    Returns:
        shape_feed_dict: A dictionary with name of the tensor as the key and its
            shape as a list as the value corresponding to it.
    Raises: ValueError if input tensor shapes are not inferrable only using batch size
    """
    shape_feed_dict = {}
    for tensor in input_tensors:
        shape = tensor.shape
        if shape.rank is None:
            raise ValueError('input tensor {} must have known rank'.format(tensor.name))
        if not shape.is_fully_defined():
            shape_proto = shape.as_proto()
            shape_proto.dim[0].size = batch_size
            shape = TensorShape(shape_proto)
        if not shape.is_fully_defined():
            raise ValueError('batch_size is not sufficient to determine the'
                             ' shape of input tensor {}'.format(tensor))
        shape_feed_dict[tensor.name] = shape.as_list()
    return shape_feed_dict


@tf_export('neuron.saved_model.compile')
def convert_to_inference_model(model_dir, new_model_dir, batch_size=1,
                               model_shape_feed_dict=None, model_feed_dict=None,
                               tags=None, signature_def_key=None,
                               **kwargs):
    """Convert a `SavedModel` to a Neuron-optimized `SavedModel`.
    Args:
        model_dir: The path of the original `SavedModel`.
        new_model_dir: The path to which the Neuron-optimized `SavedModel` will be stored.
        batch_size: (Optional) Positive integer representing batch size used in inference.
            Defaults to 1.
        model_shape_feed_dict: (Optional) Dictionary {str: list} used for inferring
            tensor shapes. Keys should match model input names and values are lists
            of positive integers representing model input tensor shapes.
        model_feed_dict: (Optional) Dictionary {str: numpy.array} used for inference.
            Useful for inferring tensor shapes. Keys should match model input names
            and values are numpy arrays that can be fed as inputs to the `SavedModel`.
        tags: (Optional) Iterable of strings to identify the required `MetaGraphDef`.
            These should correspond to the tags used when saving the variables using
            the `SavedModel` `save()` API. Default is to use the first `tag_set` available
            in the `SavedModel`.
        signature_def_key: (Optional) Iterable of strings specifying the `signature_def`s
            to use. Default is to use all `signature_def`s corresponding to `tags`.
        minimum_segment_size: Integer; minimum number of ops in an `InferentiaOp` used by
            `whitelist_partition`.
        no_fuse_ops: None or iterable of strings (unordered) representing names of ops
            that are forcibly placed on CPU.
        compiler_args: List of strings representing neuron-cc compiler arguments. Note that
            these arguments apply to all subgraphs generated by whitelist partitioning.
        compiler_workdir: Str representing work directory of the neuron-cc compiler.
    """
    _check_export_dir(new_model_dir)
    kwargs = kwargs.copy()
    if tags is None:
        tags = ['serve']  # todo: maybe read this tag from saved_model.pb
    elif type(tags) is str:
        tags = [tags]
    with session.Session(graph=ops.Graph()) as sess:
        meta_graph = tf_saved_model.loader.load(sess, tags, model_dir)
        signature_def_map = meta_graph.signature_def
        inputs = {}
        outputs = {}
        if signature_def_key is not None:
            if type(signature_def_key) is str:
                signature_def_key = [signature_def_key]
            signature_def_key = set(signature_def_key)
            signature_def_map = {key: value for key, value in signature_def_map.items()
                                            if key in signature_def_key}
        for value in signature_def_map.values():
            inputs.update({key: sess.graph.get_tensor_by_name(val.name)
                           for key, val in value.inputs.items()})
            outputs.update({key: sess.graph.get_tensor_by_name(val.name)
                            for key, val in value.outputs.items()})

        if model_feed_dict is not None:
            feed_dict = {inputs[key]: value for key, value in model_feed_dict.items()}
            kwargs.update(feed_dict=feed_dict)
        else:
            if model_shape_feed_dict is not None:
                shape_feed_dict = {inputs[key]: value
                                   for key, value in model_shape_feed_dict.items()}
            else:
                shape_feed_dict = _infer_input_shapes(inputs.values(), batch_size)
            if 'shape_feed_dict' in kwargs:
                shape_feed_dict.update(kwargs['shape_feed_dict'])
            kwargs.update(shape_feed_dict=shape_feed_dict)

        # get inference graph
        infer_graph = inference_graph_from_session(
            sess, input_tensors=inputs.values(), output_tensors=outputs.values(),
            **kwargs)

    # load inference graph into a session and export as a SavedModel
    with session.Session(graph=infer_graph) as sess:
        builder = tf_saved_model.builder.SavedModelBuilder(new_model_dir)
        for sig_def_map in signature_def_map.values():
            for tensor in sig_def_map.inputs.values():
                infer_tensor = infer_graph.get_tensor_by_name(tensor.name)
                tensor.tensor_shape.CopyFrom(infer_tensor.shape.as_proto())
            for tensor in sig_def_map.outputs.values():
                infer_tensor = infer_graph.get_tensor_by_name(tensor.name)
                tensor.tensor_shape.CopyFrom(infer_tensor.shape.as_proto())
        builder.add_meta_graph_and_variables(sess, tags, signature_def_map=signature_def_map,
                                             strip_default_attrs=True)
        builder.save()
        verbosity = logging.get_verbosity()
        logging.set_verbosity(logging.INFO)
        logging.info('Successfully converted {} to {}'.format(model_dir, new_model_dir))
        logging.set_verbosity(verbosity)


def _check_export_dir(export_dir):
    if os.path.exists(export_dir):
        raise AssertionError("Export directory already exists. Please specify a different "
                             "export directory: {}".format(export_dir))


def convert_to_inference_model_cli(args):
    if args.inputs or args.input_exprs or args.input_examples:
        from tensorflow.python.tools.saved_model_cli import load_inputs_from_input_arg_string
        model_feed_dict = load_inputs_from_input_arg_string(
            args.inputs, args.input_exprs, args.input_examples)
        kwargs = dict(model_feed_dict=model_feed_dict)
    elif args.input_shape_dict:
        kwargs = dict(model_shape_feed_dict=json.loads(args.input_shape_dict))
    else:
        kwargs = dict(batch_size=args.batch_size)
    convert_to_inference_model(
        args.dir, args.output_dir, tags=args.tag_set.split(','),
        signature_def_key=args.signature_def,
        compiler_workdir=args.compiler_workdir,
        minimum_segment_size=args.minimum_segment_size, **kwargs)


def register_convert_parser(convert_subparsers):
    parser = convert_subparsers.add_parser(
        'neuron',
        description='Convert the SavedModel with Tensorflow-Neuron integration',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--signature_def',
        type=str,
        default=None,
        metavar='SIGNATURE_DEF_KEY',
        help='key of SignatureDef to run')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='optimal size for the input batch')
    parser.add_argument(
        '--minimum_segment_size',
        type=int,
        default=2,
        help=('the minimum number of nodes required for a subgraph to be replaced'
              'in a InferentiaOp node'))
    parser.add_argument(
        '--input_shape_dict',
        default=None,
        help='Serialized dictionary for inputs names and shapes (JSON).')
    parser.add_argument('--compiler_workdir', help='path to compiler workdir')
    msg = ('Loading inputs from files, in the format of \'<input_key>=<filename>,'
           ' or \'<input_key>=<filename>[<variable_name>]\', separated by \';\'.'
           ' The file format can only be from .npy, .npz or pickle.')
    parser.add_argument('--inputs', type=str, default='', help=msg)
    msg = ('Specifying inputs by python expressions, in the format of'
           ' "<input_key>=\'<python expression>\'", separated by \';\'. '
           'numpy module is available as \'np\'. '
           'Will override duplicate input keys from --inputs option.')
    parser.add_argument('--input_exprs', type=str, default='', help=msg)
    msg = (
        'Specifying tf.Example inputs as list of dictionaries. For example: '
        '<input_key>=[{feature0:value_list,feature1:value_list}]. Use ";" to '
        'separate input keys. Will override duplicate input keys from --inputs '
        'and --input_exprs option.')
    parser.add_argument('--input_examples', type=str, default='', help=msg)
    parser.set_defaults(func=convert_to_inference_model_cli)
