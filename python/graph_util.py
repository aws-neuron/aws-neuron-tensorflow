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
import signal
import argparse
import time
import tempfile
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import subprocess
import shlex
import collections
from copy import deepcopy
import itertools
from distutils import spawn
from contextlib import contextmanager
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import importer
from tensorflow.python.framework import graph_util_impl as tf_graph_util
from tensorflow.python.framework.tensor_shape import TensorShape, dimension_value
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.core.framework import attr_value_pb2, graph_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import meta_graph
from tensorflow_neuron.python import graph_def_util as gdu
from tensorflow_neuron.python import meta_graph_util as mgu
from tensorflow_neuron.python import neuron_cc as ncc
from tensorflow_neuron.python import utils

tNeuronOp = 'NeuronOp'

@deprecated(None, 'Please refer to AWS documentation on Neuron integrated TensorFlow 2.0.')
def inference_graph_from_session(
        sess=None, input_tensors=None, output_tensors=None, signature_def=None,
        shape_feed_dict=None, feed_dict=None, dynamic_batch_size=False,
        protected_op_names=None,
        supported_op_types=None, no_fuse_ops=None, force_fuse_ops=None, minimum_segment_size=None,
        grappler=False, max_num_compilers=None,
        compiler_args=None, compiler_workdir=None, compiler_timeout=None, compiler_recovery=True,
        compiler_verbose=None, amp_pass=False):
    """Constructs an inference graph from a tensorflow session.

    Generally decomposes into 5 passes:
        1. Convert all variables to constants, `Assign`s to `Identity`s.
        2. Whitelist-based graph partitioning, each subgraph (wrapped in an `NeuronOp`)
            will contain only operations whose types match the types listed in `supported_op_types`.
        3. Shape inference to find shapes for input/output tensors of `NeuronOp` subgraphs.
        4. Call neuron-cc compiler on each `NeuronOp`.
        5. Restore `NeuronOp`s that are failed to compile into their original form.

    Args:
        sess: Active TensorFlow session.
        input_tensors: None or iterable of strings/tensors (unordered). Strings should be
            tensor names. Setting this argument can help when inference starts from some
            arbitrary tensors that are not placeholder tensors.
        output_tensors: None or iterable of strings/tensors (unordered). Strings should be
            tensor names.
        signature_def: None or a `SignatureDef` protobuf message marking graph inputs and outputs.
        shape_feed_dict: Dict `{str: shape}` used by `shape_inference`.
        feed_dict: Dict `{str: numpy.ndarray}` used by `shape_inference_with_inputs`.
            Optional. If both `shape_feed_dict` and `feed_dict` are unspecified, no shape
            inference will be performed. If only `shape_feed_dict` is specified, will perform
            `shape_inference` only. As long as `feed_dict` is specified, will perform
            `shape_inference` first and then `shape_inference_with_inputs`.
        dynamic_batch_size: Bool that represents whether the inference graph will support
            dynamic batch sizes during inference.
        supported_op_types: Iterable of strings (unordered) representing compilable op names.
        no_fuse_ops: None or iterable of strings (unordered) representing names of ops
            that are forcibly placed on CPU.
        force_fuse_ops: None or iterable of strings (unordered) representing names of ops
            that are forcibly sent to the neuron-cc compiler.
        minimum_segment_size: Integer; minimum number of ops in an `NeuronOp` used by
            `whitelist_partition`.
        max_num_compilers: Integer representing maximum allowed compiler processes.
        compiler_args: List of strings representing compiler arguments. Note that these
            arguments will be applied to all subgraphs generated by whitelist partitioning.
        compiler_workdir: Str representing work directory of the neuron-cc compiler.
        compiler_timeout: Integer representing maximum allowed runtime for the neuron-cc compiler.
        compiler_recovery: Bool representing whether to recovery from neuron-cc compiler failure.

    Returns:
        A `Graph` object that is optimized for running inference on Inferentia.

    Note:
        `input_tensors`, `shape_feed_dict`, and `feed_dict` can all set input tensors, and so
        the latter one will always override the former one.
    """
    if 'NEURON_CC_FLAGS' in os.environ:
        parser = argparse.ArgumentParser()
        parser.add_argument('--must-compile', action='store_true')
        parser.add_argument('--workdir', dest='dump_prefix', default=None)
        parser.add_argument('--verbose', type=int, default=None)
        tf_neuron_args, neuron_cc_args = parser.parse_known_args(shlex.split(os.environ['NEURON_CC_FLAGS']))
        if tf_neuron_args.verbose is not None:
            compiler_verbose = tf_neuron_args.verbose
        if tf_neuron_args.must_compile:
            compiler_recovery = False
            if compiler_verbose is None:
                compiler_verbose = 1
            logging.warning('Enabling must-compile according to NEURON_CC_FLAGS environment variable; '
                            'neuron-cc failures will be thrown as exceptions')
        if tf_neuron_args.dump_prefix is not None:
            compiler_workdir = tf_neuron_args.dump_prefix
        if neuron_cc_args:
            if compiler_args is None:
                compiler_args = neuron_cc_args
            else:
                compiler_args.extend(neuron_cc_args)
    if sess is None:
        sess = ops.get_default_session()
    if feed_dict is not None:
        feed_dict = {getattr(ts, 'name', ts): value for ts, value in feed_dict.items()}
    if shape_feed_dict is not None:
        shape_feed_dict = {getattr(ts, 'name', ts): value for ts, value in shape_feed_dict.items()}
    if signature_def is None:
        # build a SignatureDef from input/output tensors
        if input_tensors is None:
            if feed_dict is not None:
                input_names = feed_dict.keys()
            elif shape_feed_dict is not None:
                input_names = shape_feed_dict.keys()
            else:
                input_names = [op.outputs[0].name for op in sess.graph.get_operations()
                                                  if op.type == 'Placeholder']
        else:
            input_names = [getattr(ts, 'name', ts) for ts in input_tensors]
        input_tensors = [sess.graph.get_tensor_by_name(name) for name in input_names]
        if output_tensors is None:
            output_ops = [op for op in sess.graph.get_operations()
                             if all(not ts.consumers() for ts in op.outputs)]
            output_names = [ts.name for op in output_ops for ts in op.outputs]
        else:
            output_names = [getattr(ts, 'name', ts) for ts in output_tensors]
        output_tensors = [sess.graph.get_tensor_by_name(name) for name in output_names]
        signature_def = mgu.build_signature_def(input_tensors, output_tensors)

    # convert variables to constants
    if protected_op_names is None:
        protected_op_names = set()
    protected_op_names = set(protected_op_names)
    io_infos = itertools.chain(signature_def.inputs.values(), signature_def.outputs.values())
    protected_op_names.update(sess.graph.get_tensor_by_name(info.name).op.name for info in io_infos)
    if feed_dict is not None:
        protected_op_names.update(sess.graph.get_tensor_by_name(name).op.name
                                  for name in feed_dict.keys())
        if shape_feed_dict is None:
            key_dict = {key: key for key in feed_dict}
            evaluated_feed_dict = sess.run(key_dict, feed_dict)
            shape_feed_dict = {key: value.shape for key, value in evaluated_feed_dict.items()}
    if shape_feed_dict is not None:
        protected_op_names.update(sess.graph.get_tensor_by_name(name).op.name
                                  for name in shape_feed_dict.keys())

    if grappler:
        with sess.graph.as_default():
            rewriter_config = rewriter_config_pb2.RewriterConfig()
            opt_config = config_pb2.ConfigProto()
            opt_config.graph_options.rewrite_options.CopyFrom(rewriter_config)
            train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
            train_op.extend(sess.graph.get_tensor_by_name(name) for name in output_names)
            grappler_metagraph = meta_graph.create_meta_graph_def(graph=sess.graph)
            graph_def = tf_optimizer.OptimizeGraph(opt_config, grappler_metagraph)
    else:
        graph_def = sess.graph.as_graph_def(add_shapes=True)

    # convert all variables to constants
    with replace_extract_sub_graph():
        graph_def = tf_graph_util.convert_variables_to_constants.__wrapped__(
            sess, graph_def, list(protected_op_names))
    for node in graph_def.node:
        if node.op == 'RefEnter':
            node.op = 'Enter'
        elif node.op == 'RefExit':
            node.op = 'Exit'
    original_graph_def = graph_def

    # setup op exclusions
    no_fuse_ops = set() if no_fuse_ops is None else set(no_fuse_ops)
    control_op_names = [node.name for node in graph_def.node if _has_control_input(node)]

    # exclude ops with control outputs
    no_fuse_ops.update(control_op_names)

    # exclude ops that are attached to string tensors
    name_to_consumers = {}
    for node in graph_def.node:
        for inp in node.input:
            input_node_name = inp[:inp.index(':')] if ':' in inp else inp
            if input_node_name not in name_to_consumers:
                name_to_consumers[input_node_name] = set()
            name_to_consumers[input_node_name].add(node.name)
    for node in graph_def.node:
        if 'T' in node.attr:
            if node.attr['T'].type == dtypes.string.as_datatype_enum:
                no_fuse_ops.add(node.name)
                no_fuse_ops.update(name_to_consumers.get(node.name, []))
        if 'dtype' in node.attr:
            if node.attr['dtype'].type == dtypes.string.as_datatype_enum:
                no_fuse_ops.add(node.name)
                no_fuse_ops.update(name_to_consumers.get(node.name, []))

    # normalize operators
    graph_def = gdu.normalize_operators(graph_def)

    # initialize inferred shapes
    graph_def = gdu.encode_inferred_shapes(graph_def, shape_feed_dict)

    # Adding the auto-mixed precision pass
    if amp_pass:
        graph_def = amp_optimization(graph_def, signature_def)

    # fuse ops into `NeuronOp`'s and determine tensors that require shapes
    part_graph_def = whitelist_partition(
        graph_def, signature_def, supported_op_types=supported_op_types,
        no_fuse_ops=no_fuse_ops, force_fuse_ops=force_fuse_ops,
        minimum_segment_size=minimum_segment_size)

    # perform an inference to find tensor shapes as a last resort
    # todo: change to hard_shape_inference == True
    if feed_dict is not None:
        part_graph_def = gdu.shape_inference_with_inputs(part_graph_def, sess, feed_dict)

    # call compiler for each `NeuronOp`
    args_dict = {}
    if compiler_args is not None:
        args_dict = {node.name: compiler_args for node in gdu.get_neuron_nodes(part_graph_def)}
    compiled_graph_def = compile_subgraphs(
        part_graph_def, workdir=compiler_workdir,
        args_dict=args_dict, timeout=compiler_timeout, max_num_compilers=max_num_compilers,
        verbose=compiler_verbose)

    if dynamic_batch_size:
        compiled_graph_def = mark_batch_axis(compiled_graph_def)

    if compiler_recovery:
        compiled_graph_def = gdu.restore_compiler_failures(compiled_graph_def, original_graph_def)
        compiled_graph_def = gdu.run_graph_def_pass_in_subgraphs(compiled_graph_def, gdu.erase_large_constants)
        compiled_graph_def = nchw_to_nhwc(compiled_graph_def)

    # try to enable dynamic batch size if possible
    if not dynamic_batch_size:
        compiled_graph_def, dynamic_batch_size = set_dynamic_batch_size(compiled_graph_def)

    # rename NeuronOp's for better visualization
    compiled_graph_def = gdu.prefix_node_names(compiled_graph_def)

    # raise exception if NeuronOp is still uncompiled after fallback pass
    uncompiled_node_names = []
    for node in gdu.get_neuron_nodes(compiled_graph_def):
        if not node.attr['executable'].s:
            uncompiled_node_names.append(node.name)
    if uncompiled_node_names:
        raise ValueError('The following subgraphs failed to compile: {}'.format(uncompiled_node_names))

    # execution plan analysis
    compiled_graph_def = gdu.set_execution_plan(compiled_graph_def)

    # return a new graph
    compiled_graph = _graph_def_to_graph(compiled_graph_def)

    # statistics on number of operations
    num_ops_original = len(sess.graph.get_operations())
    num_ops_tfn, num_ops_on_neuron = gdu.compiled_graph_op_counts(compiled_graph_def)
    with utils.logging_show_info():
        logging.info('Number of operations in TensorFlow session: {}'.format(num_ops_original))
        logging.info('Number of operations after tf.neuron optimizations: {}'.format(num_ops_tfn))
        logging.info('Number of operations placed on Neuron runtime: {}'.format(num_ops_on_neuron))
    if ncc.find_neuron_cc() is None:
        logging.warning('***************************************************************')
        logging.warning('')
        logging.warning('  neuron-cc is not found.')
        logging.warning('')
        logging.warning('  To fully optimize TensorFlow model with AWS Neuron, please')
        logging.warning('')
        logging.warning('  install the neuron-cc compiler by "pip install neuron-cc".')
        logging.warning('')
        logging.warning('***************************************************************')
    return compiled_graph


# for test package compatibility
def compiled_graph_op_counts(compiled_graph):
    neuron_ops = [op for op in _neuron_ops(compiled_graph)]
    num_ops_on_neuron = 0
    for op in neuron_ops:
        if op.get_attr('executable'):
            subgraph_def = gdu.get_subgraph_def(op.node_def)
            num_ops_on_neuron += len(subgraph_def.node) - len(op.get_attr('input_names'))
    num_ops_tfn = len(compiled_graph.get_operations()) + num_ops_on_neuron - len(neuron_ops)
    return max(num_ops_tfn, 0), max(num_ops_on_neuron, 0)


def _graph_def_to_graph(graph_def):
    graph = ops.Graph()
    with graph.as_default():
        importer.import_graph_def(graph_def, name='')
    return graph


def _neuron_ops(graph):
    return (op for op in graph.get_operations() if op.type == gdu.tNeuronOp)


def _has_control_input(node):
    return any(inp.startswith('^') for inp in node.input)


@contextmanager
def replace_extract_sub_graph():
    extract_sub_graph = tf_graph_util.extract_sub_graph
    tf_graph_util.extract_sub_graph = tf_graph_util.extract_sub_graph.__wrapped__
    try:
        yield
    finally:
        tf_graph_util.extract_sub_graph = extract_sub_graph


def shape_inference(graph_def, shape_feed_dict, output_tensors):
    shape_feed_dict = {getattr(key, 'name', key): TensorShape(value).as_list()
                       for key, value in shape_feed_dict.items()}
    graph_def = gdu.encode_inferred_shapes(graph_def, shape_feed_dict)
    opt_config = config_pb2.ConfigProto()
    rewriter_config = opt_config.graph_options.rewrite_options
    rewriter_config.meta_optimizer_iterations = 1
    rewriter_config.min_graph_nodes = 2

    # configure shape inference
    rewriter_config.optimizers.append('aws_neuron_static_shape_inference')

    # create meta_graph_def and run grappler passes
    meta_graph_def = meta_graph_pb2.MetaGraphDef(graph_def=graph_def)
    value = meta_graph_def.collection_def[ops.GraphKeys.TRAIN_OP].node_list.value
    value.extend(getattr(key, 'name', key) for key in shape_feed_dict)
    value.extend(getattr(ts, 'name', ts) for ts in output_tensors)
    graph_def = tf_optimizer.OptimizeGraph(opt_config, meta_graph_def)
    return graph_def

def tag_multicore(graph_def, num_cores=1):
    """Adds an attribute to graph def to tag for multicore utilization
       API here is necessary for OpenPose tutorial

    Args:
        graph_def: input `GraphDef` proto.
        num_multicores: number of multicores to tag this graph as

    Returns:
        A `GraphDef` proto with the new attribute added
    """
    for node in graph_def.node:
        if node.op == tNeuronOp:
            node.attr['_automatic_multicore'].i = num_cores

    return graph_def

def amp_optimization(graph_def, signature_def):
    """Runs a AutoMixedPrecision pass to insert fp16/bf16 cast nodes at appropriate places.

    Args:
        graph_def: input `GraphDef` proto.
        signature_def: a `SignatureDef` protobuf message marking graph inputs and outputs.

    Returns:
        A `GraphDef` proto with cast operators inserted at appropriate places.
    """
    opt_config = config_pb2.ConfigProto()
    rewriter_config = opt_config.graph_options.rewrite_options
    rewriter_config.meta_optimizer_iterations = 1
    rewriter_config.min_graph_nodes = 2
    rewriter_config.optimizers.append('auto_mixed_precision_neuron')
    # Need to add constant fold since TVM has issues getting cast operators applied after constants.
    # Hence, we need to fold them to get cast const ops.
    rewriter_config.optimizers.append('constfold')

    # Setting the devidde to CPU before AMP pass
    for node in graph_def.node:
        node.device = "/device:CPU:0"

    # create meta_graph_def and run grappler passes
    meta_graph_def = meta_graph_pb2.MetaGraphDef(graph_def=graph_def)
    meta_graph_def.signature_def['serving_default'].CopyFrom(signature_def)
    graph_def = tf_optimizer.OptimizeGraph(opt_config, meta_graph_def)

    # Resetting the device to empty
    for node in graph_def.node:
        node.device = ''

    return graph_def

def conv2d_padding_optimization(graph_def, signature_def):
    """Runs a pass to split a Conv2D operation with SAME to one with VALID and a Pad operation

    Args:
        graph_def: input `GraphDef` proto.
        signature_def: a `SignatureDef` protobuf message marking graph inputs and outputs.

    Returns:
        A `GraphDef` proto with cast operators inserted at appropriate places.
    """

    opt_config = config_pb2.ConfigProto()
    rewriter_config = opt_config.graph_options.rewrite_options
    rewriter_config.meta_optimizer_iterations = 1
    rewriter_config.min_graph_nodes = 1
    rewriter_config.optimizers.append('aws_neuron_split_conv2d_same_padding')

    # create meta_graph_def and run grappler passes
    meta_graph_def = meta_graph_pb2.MetaGraphDef(graph_def=graph_def)
    meta_graph_def.signature_def['serving_default'].CopyFrom(signature_def)
    graph_def = tf_optimizer.OptimizeGraph(opt_config, meta_graph_def)

    return graph_def

def whitelist_partition(graph_def, signature_def,
                        supported_op_types=None, no_fuse_ops=None, force_fuse_ops=None,
                        minimum_segment_size=None):
    """Partitions a `GraphDef` proto according to a TensorFlow op whitelist and
    fuses each whitelisted subgraph into an `NeuronOp`.

    Args:
        graph_def: input `GraphDef` proto.
        signature_def: a `SignatureDef` protobuf message marking graph inputs and outputs.
        supported_op_types: None or iterable of strings (unordered) representing
            whitelisted op type names.
        no_fuse_ops: None or iterable of strings (unordered) representing
            names of ops that will stay unfused.
        force_fuse_ops: None or iterable of strings (unordered) representing
            names of ops that will be forcibly fused into `NeuronOp`.
        minimum_segment_size: int; minimum number of ops in an `NeuronOp`.

    Returns:
        A `GraphDef` proto with whitelisted subgraphs fused as `NeuronOp`s.
    """
    original_graph_def = graph_def
    if supported_op_types is None:
        neuron_cc = ncc.find_neuron_cc()
        if neuron_cc is None:
            return graph_def
        else:
            command = [neuron_cc, 'list-operators', '--framework', 'TENSORFLOW']
            try:
                supported_op_types = {op_type.strip() for op_type in subprocess.check_output(command).decode()[:-1].split('\n')}
            except subprocess.CalledProcessError:
                logging.warning('neuron-cc is not behaving correctly. Please check neuron-cc '
                                'installation, or reinstall by "pip install --force neuron-cc".')
                return graph_def
            supported_op_types.discard('Placeholder')
            supported_op_types.discard('IdentityN')
            supported_op_types.add('SquaredDifference')
    if no_fuse_ops is None:
        no_fuse_ops = []
    if force_fuse_ops is None:
        force_fuse_ops = []
    if minimum_segment_size is None:
        num_ops = len([node for node in graph_def.node if node.op != 'Placeholder'])
        minimum_segment_size = min(2, max(1, num_ops))
    opt_config = config_pb2.ConfigProto()
    rewriter_config = opt_config.graph_options.rewrite_options
    rewriter_config.meta_optimizer_iterations = 1
    rewriter_config.min_graph_nodes = 2
    rewriter_config.optimizers.append('aws_neuron_static_shape_inference')

    # configure operator fusion
    fuser_config = rewriter_config.custom_optimizers.add()
    fuser_config.name = 'aws_neuron_fuse_supported_operators'
    param_map = fuser_config.parameter_map
    param_map['minimum_segment_size'].i = minimum_segment_size
    param_map['supported_op_types'].list.s.extend(compat.as_bytes(item) for item in supported_op_types)
    param_map['no_fuse_ops'].list.s.extend(compat.as_bytes(getattr(item, 'name', item)) for item in no_fuse_ops)
    param_map['force_fuse_ops'].list.s.extend(compat.as_bytes(getattr(item, 'name', item)) for item in force_fuse_ops)

    # create meta_graph_def and run grappler passes
    meta_graph_def = meta_graph_pb2.MetaGraphDef(graph_def=graph_def)
    meta_graph_def.signature_def['serving_default'].CopyFrom(signature_def)
    mgu.setup_opt_config_node_v1(meta_graph_def, minimum_segment_size, supported_op_types, no_fuse_ops, force_fuse_ops)
    graph_def = tf_optimizer.OptimizeGraph(opt_config, meta_graph_def)

    # add subgraph's control input to `NeuronOp`'s control input
    original_node_with_control_inputs = gdu.get_node_with_control_inputs(original_graph_def)
    post_part_node_names = {node.name for node in graph_def.node}
    for node in gdu.get_neuron_nodes(graph_def):
        for sg_node in gdu.get_subgraph_def(node).node:
            if sg_node.name in original_node_with_control_inputs:
                for inp in original_node_with_control_inputs[sg_node.name]:
                    if inp.lstrip('^') in post_part_node_names and inp not in node.input:
                        node.input.append(inp)
    return graph_def


def _get_subgraph(node, volatile=False):
    return _graph_def_to_graph(gdu.get_subgraph_def(node, volatile=volatile))


def compile_subgraphs(graph_def,
                      workdir=None, args_dict=None, timeout=None, max_num_compilers=None,
                      verbose=None):
    """Compile `NeuronOp`s in a `GraphDef` proto.

    Args:
        graph_def: Input `GraphDef` proto that contains `NeuronOp`s.
        workdir: None or path-like representing the working directory used by the compiler;
            if None, will use `tempfile` to create a temporary workdir for each subgraph,
            else will create and use 'workdir/op_name' for each subgraph.
        args_dict: Dict `{str: list}` that maps NeuronOp names to compiler arguments;
            compiler arguments should be a list of strings, as used in `subprocess.run`.
        timeout: Integer representing timeout limit for the compiler. Default: 18000.
        max_num_compilers: Integer representing maximum allowed compiler processes.
            Default is number of cpu cores.

    Returns:
        A `GraphDef` proto with `NeuronOp`s already compiled.
    """
    if all(node.op != gdu.tNeuronOp for node in graph_def.node):
        return graph_def
    subgraph_compilers = {}
    if workdir is None:
        workdir_obj = tempfile.TemporaryDirectory()
        workdir_base = workdir_obj.name
    else:
        workdir_base = os.path.abspath(workdir)
    if timeout is None:
        timeout = 18000
    Compiler = collections.namedtuple('Compiler', 'command verbose workdir_path subgraph_info')
    _neuron_cc_input_name = 'graph_def.pb'
    _neuron_executable_name = 'graph_def.neff'
    neuron_cc = ncc.find_neuron_cc()
    if neuron_cc is None:
        return graph_def
    subgraph_info_format = '{{subgraph {} with input tensors {}, output tensors {}}}'.format
    neuron_nodes = gdu.get_neuron_nodes(graph_def)
    if not neuron_nodes:
        return graph_def
    command = []
    for node in neuron_nodes:
        if len(node.attr['input_names'].list.s) == 0 or len(node.attr['output_names'].list.s) == 0:
            continue
        subgraph_info = subgraph_info_format(node.name, *_io_tensor_info(node))
        io_config_json = _io_config(node)
        if io_config_json is None:
            logging.warning('Not fusing subgraph {}: --io-config error'.format(subgraph_info))
            continue
        if any(not TensorShape(shape).is_fully_defined() for shape in node.attr['output_shapes'].list.shape):
            logging.warning('Cannot infer output tensor shapes for subgraph {}'.format(node.name))
            continue
        subgraph_def = gdu.get_subgraph_def(node)
        for sgn in subgraph_def.node:
            inferred_shapes = sgn.attr.pop(gdu.kNeuronInferredShapes, None)
            if gdu.kOutputShapes not in sgn.attr and inferred_shapes is not None:
                sgn.attr[gdu.kOutputShapes].CopyFrom(inferred_shapes)
        workdir_path = os.path.join(workdir_base, node.name)
        os.makedirs(workdir_path, exist_ok=True)
        input_path = os.path.join(workdir_path, _neuron_cc_input_name)
        with open(input_path, 'wb') as f:
            f.write(subgraph_def.SerializeToString())
        command = [neuron_cc, 'compile', input_path, '--framework', 'TENSORFLOW',
                   '--pipeline', 'compile', 'SaveTemps',
                   '--output', os.path.join(workdir_path, _neuron_executable_name)]
        command.extend(['--io-config', io_config_json])
        if args_dict is not None:
            extend_args = args_dict.get(node.name, [])
            if isinstance(extend_args, (str, bytes)):
                extend_args = [extend_args]
            command.extend(extend_args)
        if verbose is not None:
            command.extend(['--verbose', str(verbose)])
        subgraph_compilers[node.name] = Compiler(command, verbose, workdir_path, subgraph_info)

    # try progress bar mode first
    try_progress_bar_mode = len(subgraph_compilers) == 1 and verbose is None and workdir is None
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=int, default=None)
    verbose_args, _ = parser.parse_known_args(command)
    progress_bar_mode_done = False
    if try_progress_bar_mode and (verbose_args.verbose is None or verbose_args.verbose == 35):
        node_name = next(iter(subgraph_compilers))
        command = subgraph_compilers[node_name].command.copy()
        command.extend(['--verbose=35'])
        _, _, workdir_path, subgraph_info = subgraph_compilers[node_name]
        info_string = 'fusing subgraph {} with neuron-cc'.format(subgraph_info)
        with utils.logging_show_info():
            logging.info(info_string)
        neuron_cc_stderr_txt_path = os.path.join(workdir_path, 'neuron-cc-stderr.txt')
        try:
            # note: cannot use subprocess.PIPE due to https://bugs.python.org/issue30154
            with open(neuron_cc_stderr_txt_path, 'w') as f:
                proc = subprocess.run(command, cwd=workdir_path, timeout=timeout, stderr=f)
        except subprocess.TimeoutExpired as e:
            logging.warning(e)
            progress_bar_mode_done = True
            subgraph_compilers[node_name] = None
        else:
            if proc.returncode == 0:
                progress_bar_mode_done = True
            else:
                with open(neuron_cc_stderr_txt_path, 'r') as f:
                    neuron_cc_stderr = f.read()
                if not neuron_cc_stderr.endswith('IndexError: list index out of range\n'):
                    # neuron-cc recognized progress bar mode but crashed for other reasons
                    logging.warning(neuron_cc_stderr)
                    progress_bar_mode_done = True
                    subgraph_compilers[node_name] = None
    if not progress_bar_mode_done:
        if max_num_compilers is None:
            num_cpu = multiprocessing.cpu_count()
            try:
                with open('/proc/meminfo') as f:
                    for line in f:
                        if 'MemAvailable' in line:
                            available_mem_in_kb = int(line.split()[1])
                            break
                num_mem_gb = int(available_mem_in_kb / 4e6)  # 4 GB memory for each neuron-cc process
                max_num_compilers = max(1, min(num_cpu, num_mem_gb))
            except:
                max_num_compilers = num_cpu
        with ThreadPoolExecutor(max_workers=max_num_compilers) as executor:
            compiler_returns = {
                node_name: executor.submit(
                    _fork_compiler, subgraph_compilers, node_name, timeout, try_progress_bar_mode)
                for node_name in subgraph_compilers.keys()
            }
            compiler_returns = {key: value.result() for key, value in compiler_returns.items()}
        for node_name in subgraph_compilers.keys():
            if not compiler_returns[node_name]:
                subgraph_compilers[node_name] = None

    # fill NeuronOp properties
    for node in gdu.get_neuron_nodes(graph_def):
        node.attr['input_batch_axis'].list.i[:] = [-1 for _ in node.attr['input_names'].list.s]
        node.attr['output_batch_axis'].list.i[:] = [-1 for _ in node.attr['output_names'].list.s]
        if subgraph_compilers.get(node.name, None) is None:
            continue
        workdir_path = subgraph_compilers[node.name].workdir_path
        executable_path = os.path.join(workdir_path, _neuron_executable_name)
        with open(executable_path, 'rb') as f:
            node.attr['executable'].s = f.read()
    return graph_def


def _io_tensor_info(node):
    input_names = node.attr['input_names'].list.s
    input_dtypes = node.attr['input_dtypes'].list.type
    input_shapes = node.attr['input_shapes'].list.shape
    input_tensors_info = []
    tensor_format = "<tf.Tensor '{}' shape={} dtype={}>".format
    for name, dtype_enum, shape_proto in zip(input_names, input_dtypes, input_shapes):
        name = name.decode()
        dtype = dtypes.as_dtype(dtype_enum)
        shape = TensorShape(shape_proto)
        shape = '<unknown>' if shape.rank is None else tuple(shape.as_list())
        input_tensors_info.append(tensor_format(name, shape, dtype.name))
    output_names = node.attr['output_names'].list.s
    output_dtypes = node.attr['output_dtypes'].list.type
    output_shapes = node.attr['output_shapes'].list.shape
    output_tensors_info = []
    for name, dtype_enum, shape_proto in zip(output_names, output_dtypes, output_shapes):
        name = name.decode()
        dtype = dtypes.as_dtype(dtype_enum)
        shape = TensorShape(shape_proto)
        shape = '<unknown>' if shape.rank is None else tuple(shape.as_list())
        output_tensors_info.append(tensor_format(name, shape, dtype.name))
    return input_tensors_info, output_tensors_info


def _fork_compiler(subgraph_compilers, node_name, timeout, try_progress_bar_mode):
    compiler = subgraph_compilers[node_name]
    if compiler is None:
        return None
    command, verbose, workdir_path, subgraph_info = compiler
    logfile = os.path.join(workdir_path, 'graph_def.neuron-cc.log')
    info_string = 'fusing subgraph {} with neuron-cc'.format(subgraph_info)
    if not verbose:
        info_string = '{}; you may check progress by inspecting file {}'.format(info_string, logfile)
    if not try_progress_bar_mode:
        with utils.logging_show_info():
            logging.info(info_string)
    if verbose:
        proc = subprocess.Popen(command, cwd=workdir_path)
        returncode = _wait_compiler(proc, timeout)
    else:
        with open(logfile, 'w') as logfd:
            proc = subprocess.Popen(command, cwd=workdir_path, stdout=logfd, stderr=logfd)
            returncode = _wait_compiler(proc, timeout)
    io_config_index = command.index('--io-config')
    with open(os.path.join(workdir_path, 'graph_def.io-config.json'), 'w') as f:
        f.write(command[io_config_index+1])
    if returncode != 0:
        logging.warning("Failed to fuse subgraph {} with '{}'".format(subgraph_info, subprocess.list2cmdline(command)))
        return None
    return True


def _wait_compiler(proc, timeout):
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        try:
            proc.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGKILL)
        return None
    return proc.returncode


def mark_batch_axis(compiled_graph_def):
    for node in gdu.get_neuron_nodes(compiled_graph_def):
        subgraph = _get_subgraph(node)
        node.attr['input_batch_axis'].list.i[:] = _batch_axis(node, subgraph, 'input_names')
        node.attr['output_batch_axis'].list.i[:] = _batch_axis(node, subgraph, 'output_names')
    return compiled_graph_def


def _batch_axis(node, subgraph, names_key):
    return [_one_batch_axis(subgraph, name) for name in node.attr[names_key].list.s]


def _one_batch_axis(subgraph, name):
    shape = subgraph.get_tensor_by_name(name.decode()).shape
    if shape.rank is None:
        return 0
    return 0 if len(shape) > 0 and dimension_value(shape[0]) is None else -1


def _io_config(node):
    inputs = {}
    input_names = node.attr['input_names'].list.s
    input_dtypes = node.attr['input_dtypes'].list.type
    input_shapes = node.attr['input_shapes'].list.shape
    for name, dtype_enum, shape_proto in zip(input_names, input_dtypes, input_shapes):
        name = name.decode()
        dtype = dtypes.as_dtype(dtype_enum)
        shape = TensorShape(shape_proto)
        if not shape.is_fully_defined():
            logging.warning('subgraph {} input tensor {} has undetermined shape {}'.format(node.name, name, shape))
            return None
        inputs[name] = [shape.as_list(), dtype.name]
    outputs = [name.decode() for name in node.attr['output_names'].list.s]
    return json.dumps({'inputs': inputs, 'outputs': outputs})


def nchw_to_nhwc(graph_def):
    """Convert data formats of all Conv2D/MaxPool/AvgPool ops to NCHW and insert transposes
    """
    func_map = {
        'Conv2D': nn_ops.conv2d,
        'MaxPool': nn_ops.max_pool,
        'AvgPool': nn_ops.avg_pool,
    }
    if all(node.op not in func_map for node in graph_def.node):
        return graph_def
    remove_node_names = set()
    node_rename_map = {}
    graph = _graph_def_to_graph(graph_def)
    perm_to_nhwc = [0, 2, 3, 1]

    def get_nhwc_attr(name):
        attribute = op.get_attr(name)
        if isinstance(attribute, list) and len(attribute) == 4:
            return [attribute[idx] for idx in perm_to_nhwc]
        else:
            return attribute

    with graph.as_default():
        for op in graph.get_operations():
            if op.type in func_map and op.get_attr('data_format') == b'NCHW':
                if op.type == 'Conv2D':
                    padding = op.get_attr('padding')
                    if padding == b'EXPLICIT':
                        explicit = op.get_attr('explicit_paddings')
                        padding = [explicit[2*idx:2*idx+2] for idx in perm_to_nhwc]
                    kwargs = dict(filters=op.inputs[1], dilations=get_nhwc_attr('dilations'),
                                  padding=padding, strides=get_nhwc_attr('strides'))
                elif op.type in {'MaxPool', 'AvgPool'}:
                    kwargs = dict(ksize=get_nhwc_attr('ksize'), padding=op.get_attr('padding'),
                                  strides=get_nhwc_attr('strides'))
                else:
                    continue
                input_nchw = op.inputs[0]
                with ops.name_scope(op.name):
                    input_nhwc = array_ops.transpose(input_nchw, perm_to_nhwc)
                    tensor_nhwc = func_map[op.type](input_nhwc, **kwargs)
                    tensor_nchw = array_ops.transpose(tensor_nhwc, [0, 3, 1, 2])
                remove_node_names.add(op.name)
                node_rename_map[tensor_nchw.op.name] = op.name
    temp_graph_def = graph.as_graph_def()
    graph_def = graph_pb2.GraphDef()
    graph_def.node.MergeFrom(
        node for node in temp_graph_def.node if node.name not in remove_node_names)
    for node in graph_def.node:
        if node.name in node_rename_map:
            node.name = node_rename_map[node.name]
    return graph_def


def set_dynamic_batch_size(compiled_graph_def):
    dbs = DynamicBatchSizeHelper()
    subgraph_enable_map = {}
    for node in gdu.get_neuron_nodes(compiled_graph_def):
        subgraph = _get_subgraph(node, volatile=True)
        input_names = [name.decode() for name in node.attr['input_names'].list.s]
        output_names = [name.decode() for name in node.attr['output_names'].list.s]
        tensor_dynamic_map = {}
        for name in input_names:
            shape = subgraph.get_tensor_by_name(name).shape
            tensor_dynamic_map[name] = shape.rank is None or (len(shape) > 0 and shape.as_list()[0] is None)
        for op in subgraph.get_operations():
            inputs, outputs = dbs.dynamic_inputs_outputs(op)
            if all(tensor_dynamic_map.get(ts.name, False) for ts in inputs):
                tensor_dynamic_map.update((ts.name, True) for ts in outputs)
        subgraph_enable_map[node.name] = all(tensor_dynamic_map.get(name, False) for name in input_names + output_names)
    dynamic_batch_size = subgraph_enable_map and all(
        subgraph_enable_map.get(node.name, False) for node in gdu.get_neuron_nodes(compiled_graph_def))
    if dynamic_batch_size:
        for node in gdu.get_neuron_nodes(compiled_graph_def):
            subgraph = _get_subgraph(node, volatile=True)
            node.attr['input_batch_axis'].list.i[:] = _batch_axis(node, subgraph, 'input_names')
            node.attr['output_batch_axis'].list.i[:] = _batch_axis(node, subgraph, 'output_names')
    return compiled_graph_def, dynamic_batch_size


class DynamicBatchSizeHelper:

    unary_ops = {
        'Bitcast', 'Identity', 'Abs', 'Acos', 'Acosh', 'Asin', 'Asinh', 'Atan', 'Atan2',
        'Atanh', 'BesselI0e', 'BesselI1e', 'Cast', 'Ceil', 'Cos', 'Cosh', 'Digamma',
        'Erf', 'Erfc', 'Exp', 'Expm1', 'Floor', 'FloorDiv', 'FloorMod', 'Inv',
        'IsFinite', 'IsInf', 'IsNan', 'Lgamma', 'Log', 'Log1p', 'Mod', 'Neg', 'Pow',
        'Reciprocal', 'Rint', 'Round', 'Rsqrt', 'Sigmoid', 'Sign', 'Sin', 'Sinh', 'Sqrt',
        'Square', 'Tan', 'Tanh', 'Elu', 'Relu', 'Relu6', 'Selu', 'Softplus', 'Softsign',
        'LogSoftmax', 'Softmax',
    }
    binary_broadcast_ops = {
        'Add', 'AddV2', 'Div', 'DivNoNan', 'Equal', 'Greater', 'GreaterEqual',
        'Less', 'LessEqual', 'LogicalAnd', 'LogicalNot', 'LogicalOr', 'Maximum', 'Minimum',
        'Mul', 'MulNoNan', 'NotEqual', 'RealDiv', 'SquaredDifference', 'Subtract',
        'TruncateDiv', 'TruncateMod', 'Xdivy', 'Xlogy',
    }
    reduce_axis_ops = {
        'ArgMax', 'ArgMin', 'EuclideanNorm', 'Max', 'Mean', 'Min', 'Prod', 'Sum',
    }
    pseudo_unary_ops = {
        'Pad', 'PadV2', 'ClipByValue', 'AvgPool', 'AvgPool3D', 'BiasAdd',
        'Conv2D', 'Conv3D', 'DepthwiseConv2dNative', 'Dilation2D',
        'FractionalAvgPool', 'FractionalMaxPool', 'FusedBatchNorm', 'FusedBatchNormV2', 'FusedBatchNormV3',
        'FusedPadConv2D', 'FusedResizeAndPadConv2D', 'MaxPool', 'MaxPoolV2', 'MaxPool3D',
    }

    def dynamic_inputs_outputs(self, op):
        if op.type in DynamicBatchSizeHelper.unary_ops:
            return list(op.inputs), op.outputs
        elif op.type in DynamicBatchSizeHelper.binary_broadcast_ops:
            shape0, shape1 = [ts.shape for ts in op.inputs]
            if shape0.rank is None or shape1.rank is None:
                return [], []
            if shape0.rank > shape1.rank:
                return [op.inputs[0]], op.outputs
            elif shape0.rank < shape1.rank:
                return [op.inputs[1]], op.outputs
            else:  # same rank
                inputs = []
                if len(shape0) > 0 and shape0.as_list()[0] is None:
                    inputs.append(op.inputs[0])
                if len(shape1) > 0 and shape0.as_list()[0] is None:
                    inputs.append(op.inputs[1])
                return inputs, op.outputs
        elif op.type in DynamicBatchSizeHelper.reduce_axis_ops:
            axis_op = op.inputs[-1].op
            if axis_op.type == 'Const':
                axis_list = _get_int32_values(axis_op)
                if axis_list and 0 not in axis_list:
                    return list(op.inputs[:-1]), op.outputs
        elif op.type in DynamicBatchSizeHelper.pseudo_unary_ops:
            return list(op.inputs[:1]), op.outputs
        elif op.type in {'Concat', 'ConcatV2'}:
            axis_op = op.inputs[-1].op
            if axis_op.type == 'Const':
                axis_list = _get_int32_values(axis_op)
                if any(axis < 0 for axis in axis_list):
                    rank = op.inputs[0].shape.rank
                    if rank is None:
                        return [], []
                    axis_list = [axis if axis >= 0 else (axis + rank) for axis in axis_list]
                if axis_list and 0 not in axis_list:
                    return list(op.inputs[:-1]), op.outputs
        elif op.type == 'ExpandDims':
            pass
        elif op.type == 'Stack':
            pass
        elif op.type in {'BatchMatMul', 'BatchMatMulV2'}:
            pass
        elif op.type == 'Cumprod':
            pass
        elif op.type == 'Cumsum':
            pass
        elif op.type == 'MatMul':
            if not op.node_def.attr['transpose_a'].b:
                return list(op.inputs[:1]), op.outputs
        elif op.type == 'Slice':
            pass
        elif op.type == 'StridedSlice':
            pass
        elif op.type == 'Shape':
            pass
        elif op.type == 'Reshape':
            pass
        elif op.type == 'Squeeze':
            pass
        elif op.type == 'Transpose':
            pass
        elif op.type == 'Unstack':
            pass
        return [], []


def _get_int32_values(const_op):
    tensor_proto = const_op.node_def.attr['value'].tensor
    dtype = dtypes.as_dtype(tensor_proto.dtype)
    if dtype is not dtypes.int32:
        return []
    content = tensor_proto.tensor_content
    if content:
        int_val = []
        for start in range(0, len(content), dtype.size):
            encoded_int = content[start:start+dtype.size]
            int_val.append(int.from_bytes(encoded_int, byteorder=sys.byteorder, signed=True))
        return int_val
    else:
        return tensor_proto.int_val
