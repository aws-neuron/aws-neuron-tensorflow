"""
Copyright (C) 2019, Amazon.com. All Rights Reserved
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import json
import numpy
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import ops
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.saved_model import saved_model as tf_saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.profiler import model_analyzer, option_builder
from tensorflow.python.client import timeline
from tensorflow.neuron.python.graph_util import inference_graph_from_session
from tensorflow.neuron.python.graph_util import logging_show_info
from tensorflow.neuron.python.graph_util import compiled_graph_op_counts
from tensorflow.neuron.python.graph_util import register_neuron_op


@deprecated(None, 'Please refer to AWS documentation on Neuron integrated TensorFlow 2.0.')
@tf_export('neuron.saved_model.simple_save')
def simple_save(session, export_dir, inputs, outputs, legacy_init_op=None, batch_size=1, **kwargs):
    """Convenience function to build a `SavedModel` suitable for serving.
    Args:
        session: The TensorFlow session from which to save the meta graph and variables.
        export_dir: The path to which the `SavedModel` will be stored.
        inputs: dict mapping string input names to tensors. These are added
            to the `SignatureDef` as the inputs.
        outputs:  dict mapping string output names to tensors. These are added
            to the `SignatureDef` as the outputs.
        batch_size: (Optional) Batch size used in inference.
    Note: This function sends all unknown arguments to `tf.neuron.graph_util.inference_graph_from_session`.
    """
    register_neuron_op()
    _check_export_dir(export_dir)
    # if `feed_dict` is not given, try to guess a `shape_feed_dict` from `batch_size`
    if 'shape_feed_dict' not in kwargs and 'feed_dict' not in kwargs:
        shape_feed_dict = _infer_input_shapes(inputs.values(), batch_size)
        kwargs.update(shape_feed_dict=shape_feed_dict)
    infer_graph = inference_graph_from_session.__wrapped__(
        session, input_tensors=inputs.values(), output_tensors=outputs.values(),
        **kwargs)

    # load inference graph into a session and export as a SavedModel
    with tf_session.Session(graph=infer_graph) as sess:
        inputs = {key: sess.graph.get_tensor_by_name(ts.name) for key, ts in inputs.items()}
        outputs = {key: sess.graph.get_tensor_by_name(ts.name) for key, ts in outputs.items()}
        tf_saved_model.simple_save.__wrapped__(sess, export_dir, inputs, outputs,
                                               legacy_init_op=legacy_init_op)


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
                               tags=None, signature_def_key=None, strip_default_attrs=False,
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
        signature_def_key: (Optional) String specifying the `signature_def` to use. Default is
            to use 'serving_default' or the first `signature_def` corresponding to `tags`.
        strip_default_attrs: Boolean. If `True`, default-valued attributes will be
            removed from the NodeDefs. For a detailed guide, see
            [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
        minimum_segment_size: Integer; minimum number of ops in an `NeuronOp` used by
            `whitelist_partition`.
        no_fuse_ops: None or iterable of strings (unordered) representing names of ops
            that are forcibly placed on CPU.
        compiler_args: List of strings representing neuron-cc compiler arguments. Note that
            these arguments apply to all subgraphs generated by whitelist partitioning.
        compiler_workdir: Str representing work directory of the neuron-cc compiler.

    Returns:
        Dictionary with operator counts before/after optimization, etc.

    Note: This function sends all unknown arguments to `tf.neuron.graph_util.inference_graph_from_session`.
    """
    register_neuron_op()
    _check_export_dir(new_model_dir)
    kwargs = kwargs.copy()
    tags = _normalize_tags(tags, model_dir)
    with tf_session.Session(graph=ops.Graph()) as sess:
        meta_graph = tf_saved_model.loader.load.__wrapped__(sess, tags, model_dir)
        signature_def_key, signature_def = _get_signature_def(meta_graph, signature_def_key)
        input_tensors = {sess.graph.get_tensor_by_name(ts.name)
                         for ts in signature_def.inputs.values()}
        output_tensors = {sess.graph.get_tensor_by_name(ts.name)
                          for ts in signature_def.outputs.values()}
        inputs = {}
        for key, value in signature_def.inputs.items():
            if key not in inputs:
                inputs[key] = sess.graph.get_tensor_by_name(value.name)
        if model_feed_dict is not None:
            feed_dict = {inputs[key]: value for key, value in model_feed_dict.items()}
            kwargs.update(feed_dict=feed_dict)
        else:
            if model_shape_feed_dict is not None:
                kwargs.update(shape_feed_dict={
                    inputs[key]: value for key, value in model_shape_feed_dict.items()})
            else:
                if 'shape_feed_dict' not in kwargs and 'feed_dict' not in kwargs:
                    kwargs.update(shape_feed_dict=_infer_input_shapes(inputs.values(), batch_size))

        # get inference graph
        infer_graph = inference_graph_from_session.__wrapped__(
            sess, input_tensors=input_tensors, output_tensors=output_tensors,
            **kwargs)

    # load inference graph into a session and export as a SavedModel
    with tf_session.Session(graph=infer_graph) as sess:
        builder = tf_saved_model.builder.SavedModelBuilder(new_model_dir)
        signature_def_map = {signature_def_key: signature_def}
        for tensor in signature_def.inputs.values():
            infer_tensor = infer_graph.get_tensor_by_name(tensor.name)
            tensor.tensor_shape.CopyFrom(infer_tensor.shape.as_proto())
        for tensor in signature_def.outputs.values():
            infer_tensor = infer_graph.get_tensor_by_name(tensor.name)
            tensor.tensor_shape.CopyFrom(infer_tensor.shape.as_proto())
        builder.add_meta_graph_and_variables(sess, tags, signature_def_map=signature_def_map,
                                             strip_default_attrs=strip_default_attrs)
        builder.save()
    num_ops_tfn, num_ops_on_neuron = compiled_graph_op_counts(infer_graph)
    on_neuron_ratio = float(num_ops_on_neuron) / num_ops_tfn if num_ops_tfn != 0 else 0.0
    converted_msg = '{} to {}'.format(model_dir, new_model_dir)
    if on_neuron_ratio == 0.0:
        ops_msg = 'no operator'
        unless_msg = ''
    else:
        ops_msg = 'only a small portion of operators'
        unless_msg = ' (well, unless there are too many training operators in your SavedModel)'
    warning_msg = (
        'but {} will be running on AWS machine learning accelerators. This is probably '
        'not what you want{}. Please refer to https://github.com/aws/aws-neuron-sdk '
        'for current limitations of the AWS Neuron SDK. We are actively improving '
        '(and hiring)!'.format(ops_msg, unless_msg))
    if on_neuron_ratio > 0.3:
        with logging_show_info():
            logging.info('Successfully converted {}'.format(converted_msg))
    elif on_neuron_ratio == 0.0:
        logging.warning('Converted {} {}'.format(converted_msg, warning_msg))
    else:
        logging.warning('Converted {} {}'.format(converted_msg, warning_msg))
    return dict(OnNeuronRatio=on_neuron_ratio)
compile = convert_to_inference_model


def _check_export_dir(export_dir):
    if file_io.file_exists(export_dir):
        raise AssertionError("Export directory already exists. Please specify a different "
                             "export directory: {}".format(export_dir))


def _normalize_tags(tags, model_dir):
    if tags is None:
        default_tag_set = {tag_constants.SERVING}
        # default to 'serving_default' first and then the tags in the first meta_graph
        model = parse_saved_model(model_dir)
        for meta_graph in model.meta_graphs:
            if set(meta_graph.meta_info_def.tags) == default_tag_set:
                tags = [tag_constants.SERVING]
        if tags is None:
            tags = model.meta_graphs[0].meta_info_def.tags
        if len(model.meta_graphs) > 1:
            tags_list = [mg.meta_info_def.tags for mg in model.meta_graphs]
            logging.warning('SavedModel {} contains more than one tag-set {}; '
                            'using {} as the default'.format(model_dir, tags_list, tags))
    elif type(tags) is str:
        tags = tags.split(',')
    return tags


def _get_signature_def(meta_graph, signature_def_key):
    signature_def_map = meta_graph.signature_def
    if signature_def_key is None:  # default to 'serving_default' first
        if signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY in signature_def_map:
            signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        else:  # default to the first one found
            signature_def_key = list(signature_def_map.keys())[0]
            logging.warning('Using non-default signature_def_key {}'.format(signature_def_key))
    return signature_def_key, signature_def_map[signature_def_key]


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
              'in a NeuronOp node'))
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


if sys.argv[0].endswith('saved_model_cli'):
    def convert_add_subparsers(*args, **kwargs):
        parser = add_subparsers(*args, **kwargs)
        if kwargs.get('title', None) == 'conversion methods':
            register_convert_parser(parser)
        return parser
    if argparse.ArgumentParser.add_subparsers is not convert_add_subparsers:
        add_subparsers = argparse.ArgumentParser.add_subparsers
        argparse.ArgumentParser.add_subparsers = convert_add_subparsers


@tf_export('neuron.saved_model.profile')
def profile(model_dir, timeline_json=None, batch_size=1, model_shape_feed_dict=None,
            model_feed_dict=None, tags=None, signature_def_key=None, config=None,
            num_warmup_runs=1, op_log=None, cmd='scope', options=None):
    """Run tensorflow profiler on a `SavedModel`.

    Args:
        model_dir: The path of the `SavedModel`.
        timeline_json: The path to which a 'timeline' json tracing will be saved.
            This json can be visualized using chrome://tracing.
        batch_size: Positive integer representing batch size used in inference.
            Defaults to 1.
        model_shape_feed_dict: Dictionary {str: list} used for creating input data
            from tensor shapes. Keys should match model input names and values are lists
            of positive integers representing model input tensor shapes.
        model_feed_dict: Dictionary {str: numpy.array} used for inference.
            Useful for inferring tensor shapes. Keys should match model input names
            and values are numpy arrays that can be fed as inputs to the `SavedModel`.
        tags: Iterable of strings to identify the required `MetaGraphDef`.
            These should correspond to the tags used when saving the variables using
            the `SavedModel` `save()` API. Default is to use the first `tag_set` available
            in the `SavedModel`.
        signature_def_key: String specifying the `signature_def` to use. Default is
            to use 'serving_default' or the first `signature_def` corresponding to `tags`.
        num_warmup_runs: int representing number of warmup inference runs.
            Possibly useful with XLA where re-compilation can happen.
        op_log: tensorflow.tfprof.OpLogProto proto. User can assign "types" to
            graph nodes with op_log. "types" allow user to flexibly group and
            account profiles using options['accounted_type_regexes'].
        cmd: string. Either 'op', 'scope', 'graph' or 'code'.
            'op' view organizes profile using operation type. (e.g. MatMul)
            'scope' view organizes profile using graph node name scope.
            'graph' view organizes profile using graph node inputs/outputs.
            'code' view organizes profile using Python call stack.
        options: A dict of options. See core/profiler/g3doc/options.md.

    Returns:
        If cmd is 'scope' or 'graph', returns GraphNodeProto proto.
        If cmd is 'op' or 'code', returns MultiGraphNodeProto proto.
        Side effect: stdout/file/timeline.json depending on options['output']
    """
    tags = _normalize_tags(tags, model_dir)
    with tf_session.Session(graph=ops.Graph(), config=config) as sess:
        meta_graph = tf_saved_model.loader.load(sess, tags, model_dir)
        _, signature_def = _get_signature_def(meta_graph, signature_def_key)
        inputs = {name: sess.graph.get_tensor_by_name(ts.name)
                  for name, ts in signature_def.inputs.items()}
        outputs = {name: sess.graph.get_tensor_by_name(ts.name)
                   for name, ts in signature_def.outputs.items()}
        if model_feed_dict is None:
            model_feed_dict = {}
            for name, tensor in inputs.items():
                if model_shape_feed_dict is None:
                    shape = tensor.shape.as_list()
                    shape[0] = batch_size
                else:
                    shape = json.loads(model_shape_feed_dict)[name]
                model_feed_dict[name] = numpy.zeros(shape, dtype=tensor.dtype.as_numpy_dtype)
        feed_dict = {tensor: model_feed_dict[name] for name, tensor in inputs.items()}

        # warm up run
        for _ in range(num_warmup_runs):
            sess.run(outputs, feed_dict=feed_dict)
        run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
        run_metadata = config_pb2.RunMetadata()

        # profiling run
        sess.run(outputs, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

        # write out timeline json if requested
        if timeline_json is not None:
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(timeline_json, 'w') as f:
                f.write(chrome_trace)

        # run profiling
        if options is None:
            options = option_builder.ProfileOptionBuilder.time_and_memory()
        return model_analyzer.profile(sess.graph, run_metadata, op_log=op_log, cmd=cmd, options=options)
