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

import sys
import os
import argparse
import json
from tempfile import TemporaryDirectory
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import ops
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.saved_model import saved_model as tf_saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.profiler import model_analyzer, option_builder
from tensorflow.python.client import timeline
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.ops.variable_scope import get_variable
from tensorflow.python.ops.init_ops import Zeros as zeros_initializer
from tensorflow.python.training.saver import Saver
from tensorflow.python.framework import importer
from tensorflow.core.framework import graph_pb2
from tensorflow.neuron.python import graph_def_util as gdu
from tensorflow.neuron.python import utils
from tensorflow.neuron.python.graph_util import inference_graph_from_session


@deprecated(None, 'Please refer to AWS documentation on Neuron integrated TensorFlow 2.0.')
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


def _infer_input_shapes(input_tensors, batch_size, signature_def):
    """Infer/guess the shape of the inputs using batch_size.
    Args:
        input_tensors: Iterable of input tensors
        batch_size: Positive integer; batch size used to infer input tensor shapes
    Returns:
        shape_feed_dict: A dictionary with name of the tensor as the key and its
            shape as a list as the value corresponding to it.
    Raises: ValueError if input tensor shapes are not inferrable only using batch size
    """
    signature_shapes = {tp.name: tp.tensor_shape for tp in signature_def.inputs.values()}
    shape_feed_dict = {}
    for tensor in input_tensors:
        shape = tensor.shape
        if not shape.is_fully_defined():
            if shape.rank is not None:
                shape_proto = shape.as_proto()
            else:
                shape_proto = signature_shapes.get(tensor.name, None)
            if shape_proto is None or shape_proto.unknown_rank:
                raise ValueError('input tensor {} must have known rank'.format(tensor.name))
            if shape_proto.dim:
                shape_proto.dim[0].size = batch_size
            shape = TensorShape(shape_proto)
        if not shape.is_fully_defined():
            raise ValueError('batch_size is not sufficient to determine the'
                             ' shape of input tensor {}'.format(tensor))
        shape_feed_dict[tensor.name] = shape.as_list()
    return shape_feed_dict


def convert_to_inference_model(model_dir, new_model_dir, batch_size=1,
                               model_shape_feed_dict=None, model_feed_dict=None,
                               tags=None, signature_def_key=None, strip_default_attrs=False,
                               config_proto=None, constant_size_to_exclude=1024, 
                               convert_constants_to_variables=False, compiler_workdir=None, **kwargs):
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
    if config_proto is None:
        config_proto = config_pb2.ConfigProto(allow_soft_placement=True)
    _check_export_dir(new_model_dir)
    kwargs = kwargs.copy()
    tags = _normalize_tags(tags, model_dir)
    with tf_session.Session(graph=ops.Graph(), config=config_proto) as sess:
        meta_graph = tf_saved_model.loader.load.__wrapped__(sess, tags, model_dir)
        _check_for_compatible_tf_version(model_dir, sess)
        signature_def_key, signature_def = _get_signature_def(meta_graph, signature_def_key)
        input_tensors = {sess.graph.get_tensor_by_name(ts.name)
                         for ts in signature_def.inputs.values()}
        output_tensors = {sess.graph.get_tensor_by_name(ts.name)
                          for ts in signature_def.outputs.values()}
        saved_model_main_op = meta_graph.collection_def['saved_model_main_op'].node_list.value
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
                    kwargs.update(shape_feed_dict=_infer_input_shapes(inputs.values(), batch_size, signature_def))

        # get inference graph
        infer_graph = inference_graph_from_session.__wrapped__(
            sess, input_tensors=input_tensors, output_tensors=output_tensors,
            signature_def=signature_def,
            protected_op_names=saved_model_main_op, compiler_workdir=compiler_workdir, **kwargs)

        if convert_constants_to_variables:
            if compiler_workdir is None:
                temp_dir = TemporaryDirectory()
                compiler_workdir = temp_dir.name
            infer_graph = convert_constant_to_variables(
                    sess, 
                    infer_graph,
                    compiler_workdir=compiler_workdir,
                    constant_size_to_exclude=constant_size_to_exclude,
                )
    # load inference graph into a session and export as a SavedModel
    with tf_session.Session(graph=infer_graph, config=config_proto) as sess:
        # After adding variables in the graph, need to initialize the variables before saving them
        for op in infer_graph.get_operations():
            if "init" in op.name and op.type == "NoOp":
                sess.run(op)

        builder = tf_saved_model.builder.SavedModelBuilder(new_model_dir)
        signature_def_map = {signature_def_key: signature_def}
        for tensor in signature_def.inputs.values():
            infer_tensor = infer_graph.get_tensor_by_name(tensor.name)
            tensor.tensor_shape.CopyFrom(infer_tensor.shape.as_proto())
        for tensor in signature_def.outputs.values():
            infer_tensor = infer_graph.get_tensor_by_name(tensor.name)
            tensor.tensor_shape.CopyFrom(infer_tensor.shape.as_proto())
        saved_model_main_op = [sess.graph.get_operation_by_name(name) for name in saved_model_main_op]
        main_op = saved_model_main_op[0] if saved_model_main_op else None
        builder.add_meta_graph_and_variables(sess, tags, signature_def_map=signature_def_map,
                                             strip_default_attrs=strip_default_attrs,
                                             main_op=main_op)
        builder.save()
    num_ops_tfn, num_ops_on_neuron = gdu.compiled_graph_op_counts(infer_graph.as_graph_def())
    on_neuron_ratio = float(num_ops_on_neuron) / num_ops_tfn if num_ops_tfn != 0 else 0.0
    utils.model_conversion_report(model_dir, new_model_dir, on_neuron_ratio)
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


def set_core_binding(model_dir, index_list):
    saved_model_pb, neuron_node_list = _saved_model_pb_neuron_nodes(model_dir)
    default_model_config = [-1, -1, -1, 10, -1]
    for node, device_index in zip(neuron_node_list, index_list):
        len_model_config = len(node.attr['model_config'].list.i)
        if len_model_config < 5:
            model_config = default_model_config.copy()
            model_config[4] = device_index
            node.attr['model_config'].list.i[len_model_config:] = model_config[len_model_config:]
        else:
            node.attr['model_config'].list.i[4] = device_index
    with gfile.Open(os.path.join(model_dir, 'saved_model.pb'), 'wb') as f:
        f.write(saved_model_pb.SerializeToString())


def inspect_core_binding(model_dir):
    saved_model_pb, neuron_node_list = _saved_model_pb_neuron_nodes(model_dir)
    with utils.logging_show_info():
        for node in neuron_node_list:
            if len(node.attr['model_config'].list.i) > 4:
                device_index = node.attr['model_config'].list.i[4]
            else:
                device_index = -1
            if device_index < 0:
                logging.info('subgraph {} does not bind to a NeuronCore Group'.format(node.name))
            else:
                logging.info('subgraph {} binds to NeuronCore Group index {}'
                             .format(node.name, device_index))


def _saved_model_pb_neuron_nodes(model_dir):
    saved_model_pb = saved_model_pb2.SavedModel()
    with gfile.Open(os.path.join(model_dir, 'saved_model.pb'), 'rb') as f:
        saved_model_pb.ParseFromString(f.read())
    neuron_node_list = []
    for meta_graph in saved_model_pb.meta_graphs:
        for node in meta_graph.graph_def.node:
            if node.op == gdu.tNeuronOp:
                neuron_node_list.append(node)
    return saved_model_pb, neuron_node_list


def profile(model_dir, model_feed_dict=None, timeline_json=None,
            tags=None, signature_def_key=None, config=None,
            num_warmup_runs=1, op_log=None, cmd='scope', options=None):
    """Run tensorflow profiler on a `SavedModel`.

    Args:
        model_dir: The path of the `SavedModel`.
        timeline_json: The path to which a 'timeline' json tracing will be saved.
            This json can be visualized using chrome://tracing.
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

def _check_for_compatible_tf_version(model_dir, sess):
    #this function checks for a StatefulPartitionedCall
    #operator, which is not supported in TF1.15.x
    #Therefore if we find this operator we know that
    #the model is using TF2.x which is unsupported
    for op in sess.graph.get_operations():
        if op.type == 'StatefulPartitionedCall':
            raise NotImplementedError('Model {} is of type tensorflow2.x. '
                                        'As of now, tensorflow-neuron only supports '
                                        'models saved in tensorflow1.15.x. Please '
                                        'save your model with tensorflow1.15.x '
                                        'and try again.'.format(model_dir))
                                        

def convert_constant_to_variables(
    sess,
    compiled_graph,
    compiler_workdir,
    constant_size_to_exclude=1024,
):
    # This function is used to replace the constants in the graph with the variables.
    # This is done specifically for the large constants in the graph.

    checkpoint_dir = os.path.join(compiler_workdir, "checkpoint_dir")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, "og.ckpt")

    _saver = Saver()
    _saver.save(sess, checkpoint_dir)

    with tf_session.Session(graph=compiled_graph) as session:
        _variables = {}
        op_names = []
        tensor_names = []
        # Getting all the Const nodes and their values. These const nodes will be removed from the graph
        for op in session.graph.get_operations():
            if "Const" in op.type:
                for tensor in op.values():
                    op_names.append(op.name)
                    tensor_names.append(tensor.name)
        
        tensor_values = session.run(tensor_names)
        for name, value in zip(op_names, tensor_values):
            total_elements = 1
            for x in value.shape:
                total_elements *= x
            if total_elements > constant_size_to_exclude:
                _variables[name] = value

        new_nodes = []
        old_nodes_names = []
        # Creating the graph def by removing all the Const nodes
        for node in session.graph.as_graph_def().node:
            if node.name in _variables:
                old_nodes_names.append(node.name)
                continue
            else:
                new_nodes.append(node)

        mod_graph_def = graph_pb2.GraphDef()
        mod_graph_def.node.extend(new_nodes)

    # Creating a graph from the modified graph def. This graph def has all the nodes from the frozen graph, except the const nodes
    graph = ops.Graph()
    with tf_session.Session(graph=graph) as session:
        init_vars = {}
        for var, value in _variables.items():
            _variables[var] = ops.convert_to_tensor(get_variable(
                name="{}-imported".format(var),
                shape=value.shape,
                initializer=zeros_initializer(),
            ))
            init_vars[var] = "{}-imported".format(var)
        checkpoint_utils.init_from_checkpoint(checkpoint_dir, init_vars)

        session.run(global_variables_initializer())
        importer.import_graph_def(mod_graph_def, name="", input_map=_variables)

    return graph
