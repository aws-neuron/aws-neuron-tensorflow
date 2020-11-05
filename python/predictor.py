# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
import os
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops
from tensorflow.python.client import session
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python.saved_model import saved_model as tf_saved_model
from tensorflow.python.saved_model.utils import build_tensor_info
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_utils

from tensorflow.neuron.python.graph_util import inference_graph_from_session
from tensorflow.neuron.python.saved_model_util import get_io_names_from_signature_def


DEFAULT_TAGS = 'serve'


@deprecated(None, 'Please refer to AWS documentation on Neuron integrated TensorFlow 2.0.')
@tf_export('neuron.predictor.from_saved_model')
def from_saved_model(export_dir, signature_def_key=None, signature_def=None,
                     input_names=None, output_names=None, tags=None, graph=None, config=None):
  """Constructs a `Predictor` from a `SavedModel` on disk.

    Args:
      export_dir: a path to a directory containing a `SavedModel`.
      signature_def_key: Optional string specifying the signature to use. If
        `None`, then `DEFAULT_SERVING_SIGNATURE_DEF_KEY` is used. Only one of
      `signature_def_key` and `signature_def`
      signature_def: A `SignatureDef` proto specifying the inputs and outputs
        for prediction. Only one of `signature_def_key` and `signature_def`
        should be specified.
        input_names: A dictionary mapping strings to `Tensor`s in the `SavedModel`
          that represent the input. The keys can be any string of the user's
          choosing.
        output_names: A dictionary mapping strings to `Tensor`s in the
          `SavedModel` that represent the output. The keys can be any string of
          the user's choosing.
      tags: Optional. Tags that will be used to retrieve the correct
        `SignatureDef`. Defaults to `DEFAULT_TAGS`.
      graph: Optional. The Tensorflow `graph` in which prediction should be
        done.
      config: `ConfigProto` proto used to configure the session.

    Returns:
      An initialized `Predictor`.

    Raises:
      ValueError: More than one of `signature_def_key` and `signature_def` is
        specified.
  """
  # todo: split NeuronPredictor constructor into two cases for SavedModel/GraphDef
  return NeuronPredictor(model_dir=export_dir,
                         signature_def_key=signature_def_key,
                         signature_def=signature_def,
                         input_names=input_names,
                         output_names=output_names,
                         tags=tags,
                         graph=graph,
                         config=config)


@deprecated(None, 'Please refer to AWS documentation on Neuron integrated TensorFlow 2.0.')
@tf_export('neuron.predictor.from_graph_def')
def from_graph_def(export_dir, signature_def_key=None, signature_def=None,
                   input_names=None, output_names=None, tags=None, graph=None, config=None):
  """Constructs a `Predictor` from a serialized `GraphDef` protocol buffer
     (a.k.a serialized frozen graph) on disk.

    Args:
      export_dir: a path to a directory containing a serialized `GraphDef` protocol buffer.
      signature_def_key: Optional string specifying the signature to use. If
        `None`, then `DEFAULT_SERVING_SIGNATURE_DEF_KEY` is used. Only one of
      `signature_def_key` and `signature_def`
      signature_def: A `SignatureDef` proto specifying the inputs and outputs
        for prediction. Only one of `signature_def_key` and `signature_def`
        should be specified.
        input_names: A dictionary mapping strings to `Tensor`s in the `GraphDef`
          that represent the input. The keys can be any string of the user's
          choosing.
        output_names: A dictionary mapping strings to `Tensor`s in the
          `GraphDef` that represent the output. The keys can be any string of
          the user's choosing.
      tags: Optional. Tags that will be used to retrieve the correct
        `SignatureDef`. Defaults to `DEFAULT_TAGS`.
      graph: Optional. The Tensorflow `graph` in which prediction should be
        done.
      config: `ConfigProto` proto used to configure the session.

    Returns:
      An initialized `Predictor`.

    Raises:
      ValueError: `input_names` or `output_names` are not given, or more than one
        of `signature_def_key` and `signature_def` is specified.
  """
  # todo: split NeuronPredictor constructor into two cases for SavedModel/GraphDef
  if input_names is None:
    raise ValueError('input_names must be provided for running inference on frozen graph')
  if output_names is None:
    raise ValueError('output_names must be provided for running inference on frozen graph')
  return NeuronPredictor(model_dir=export_dir,
                         signature_def_key=signature_def_key,
                         signature_def=signature_def,
                         input_names=input_names,
                         output_names=output_names,
                         tags=tags,
                         graph=graph,
                         config=config)


class NeuronPredictor(object):
  """Predictor class to predict Neuron TensorFlow models.

  The `NeuronPredictor` object wraps a saved model which is specified by a `model_dir`,
  which loads the model, freezes it, converts it to Neuron graph(s), and executes the
  workflow for prediction.

  The `model_dir` must be specified and point to a valid saved model directory.

  The `config` argument can be passed `RunConfig` object containing information
  about the execution environment. It is passed on to the `model_fn`, if the
  `model_fn` has a parameter named "config" (and input functions in the same
  manner). If the `config` parameter is not passed, it is instantiated by the
  `Predictor`. Not passing config means that defaults useful for local execution
  are used. `Predictor` makes config available to the model (for instance, to
  allow specialization based on the number of workers available), and also uses
  some of its fields to control internals, especially regarding checkpointing.

  The `params` argument contains hyperparameters. It is passed to the
  `model_fn`, if the `model_fn` has a parameter named "params", and to the input
  functions in the same manner. `Predictor` only passes params along, it does
  not inspect it. The structure of `params` is therefore entirely up to the
  developer.

  None of `Predictor`'s methods can be overridden in subclasses (its
  constructor enforces this). Subclasses should use `model_fn` to configure
  the base class, and may add methods implementing specialized functionality.

  @compatibility(eager)
  Predictors are not compatible with eager execution.
  @end_compatibility
  """

  def __del__(self):
    """ Further expansion of destructor needed, currently explicitly closing the
    internal session. Not too sure if its explicitly needed"""
    if self._internal_sess is not None:
      self._internal_sess.close()

  @property
  def graph(self):
    return self._graph

  @property
  def session(self):
    return self._session

  @property
  def feed_tensors(self):
    return self._feed_tensors

  @property
  def fetch_tensors(self):
    return self._fetch_tensors

  def compile(self, model_feed_dict=None, model_shape_feed_dict=None, args=None, workdir=None, **kwargs):
    feed_dict = None if model_feed_dict is None else self.return_feed_dict(model_feed_dict)
    shape_feed_dict = None if model_shape_feed_dict is None else self.return_feed_dict(model_shape_feed_dict)
    extra_kwargs = dict(feed_dict=feed_dict, shape_feed_dict=shape_feed_dict,
                        compiler_args=args, compiler_workdir=workdir)
    extra_kwargs.update(kwargs)
    sess = self._internal_sess
    graph = inference_graph_from_session.__wrapped__(sess, input_tensors=self.input_names,
                                         output_tensors=self.output_names, **extra_kwargs)
    self._feed_tensors = {name: graph.get_tensor_by_name(ts.name)
                          for name, ts in self._feed_tensors.items()}
    self._fetch_tensors = {name: graph.get_tensor_by_name(ts.name)
                          for name, ts in self._fetch_tensors.items()}
    sess.close()
    # to allow for multiple predicts on different outputs
    self._internal_sess = session.Session(graph=graph, config=self._config)
    self._graph  = self._internal_sess.graph

  def export_saved_model(self, export_dir):
    # load inference graph into a session and export as a SavedModel
    with session.Session(graph=self._graph) as sess:
        builder = tf_saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(sess,
                                             self._tags,
                                             signature_def_map=self._signature_def_map,
                                             strip_default_attrs=True)
        builder.save()

  def __repr__(self):
    return '{} with feed tensors {} and fetch_tensors {}'.format(
        type(self).__name__, self._feed_tensors, self._fetch_tensors)


  def __init__(self,
               model_dir,
               signature_def_key=None,
               signature_def=None,
               input_names=None,
               output_names=None,
               tags=None,
               graph=None,
               config=None):
    """Constructs an `Predictor` instance.

    Args:
      model_dir: Model to load into EI graph.
      signature_def_key: Optional string specifying the signature to use. If
        `None`, then `DEFAULT_SERVING_SIGNATURE_DEF_KEY` is used. Only one of
        `signature_def_key` and `signature_def` should be specified.
      signature_def: A `SignatureDef` proto specifying the inputs and outputs
        for prediction. Only one of `signature_def_key` and `signature_def`
        should be specified.
      input_names: A dictionary mapping strings to `Tensor`s in the `SavedModel`
        that represent the input. The keys can be any string of the user's
        choosing.
      output_names: A dictionary mapping strings to `Tensor`s in the
        `SavedModel` that represent the output. The keys can be any string of
        the user's choosing.
      tags: Optional. List or comma separated list of tags that will be used to retrieve
        the correct `SignatureDef`. Defaults to `DEFAULT_TAGS`.
      config: Configuration object.

    Raises:
      RuntimeError: If eager execution is enabled.
      ValueError: parameters of `model_fn` don't match `params`.
      ValueError: if this is called via a subclass and if that class overrides
        a member of `Predictor`.
    """
    self._internal_sess = None
    self._config = config

    if tags is None:
      tags = [DEFAULT_TAGS]
    else:
      if isinstance(tags, str):
        tags = tags.split(',')
    set_of_tags = set()
    self._tags = []
    for tag in tags:
      if tag not in set_of_tags:
        self._tags.append(tag)
        set_of_tags.add(tag)

    # Get input_names and output_names , if required.
    input_names, output_names = get_io_names_from_signature_def(model_dir,
                                                                signature_def_key,
                                                                signature_def,
                                                                input_names,
                                                                output_names,
                                                                ','.join(tags))
    self.input_names = list(input_names.values())
    self.output_names = list(output_names.values())

    self._graph = graph or ops.Graph()

    self._internal_sess = session.Session(config=config, graph=self._graph)

    is_saved_model = model_dir is not None and os.path.isdir(model_dir)

    if is_saved_model:
      loader.load(self._internal_sess, tags, model_dir)
    elif model_dir is not None: # for frozen model
      with gfile.FastGFile(model_dir, 'rb') as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())
        with self._graph.as_default():
          importer.import_graph_def(graph_def, name="")
    else:
      raise Exception("Must supply a valid saved model or frozen graph")

    self._feed_tensors = {k: self._internal_sess.graph.get_tensor_by_name(self.make_tensor_name(v))
                          for k, v in input_names.items()}
    self._fetch_tensors = {k: self._internal_sess.graph.get_tensor_by_name(self.make_tensor_name(v))
                           for k, v in output_names.items()}

    # save signature def map for export
    if is_saved_model:
        metagraph_def = saved_model_utils.get_meta_graph_def(model_dir, ','.join(tags))
        self._signature_def_map = metagraph_def.signature_def
    # construct default signature def map for export
    else:
        inputs = {k: build_tensor_info(v) for k, v in self._feed_tensors.items()}
        outputs = {k: build_tensor_info(v) for k, v in self._fetch_tensors.items()}
        self._signature_def_map = {
            tf_saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf_saved_model.signature_def_utils.build_signature_def(
                   inputs, outputs
                )
            }

  def print_graph_ops(self, g):
    for op in g.get_operations():
      for inp in op.inputs:
        print ("  Input {}".format(inp))
        print('- {0:20s} "{1}" ({2} outputs) {3}'.format(op.type, op.name, len(op.outputs), op.outputs))

  def make_tensor_name(self, tensor_name):
    if ":" in tensor_name:
      return tensor_name
    return tensor_name+":0"

  def return_feed_dict(self, feats):
    feed_dict = {}
    feature_index = 0
    for feed_name, value in feats.items():
      if feed_name not in self._feed_tensors.keys():
        raise ValueError("Incorrect input name: {0}. Does not match the input names specified in Predictor or signature def:  {1}".format(feed_name, self._feed_tensors.keys()))
      in_tensor_name = self._feed_tensors[feed_name]

      if isinstance(in_tensor_name, ops.Tensor):
        in_tensor_name = in_tensor_name.name

      in_tensor_name = self.make_tensor_name(in_tensor_name)
      feed_dict[in_tensor_name] = value

    return feed_dict


  def __call__(self, model_feed_dict):
    """Yields predictions for given features.

    Args:
      feed_fn: A function that returns dictionary of input_names:tensor_values, which
      will be used for feed_dict.

    Yields:
      Evaluated values of `predictions` tensors.

    Raises:
      ValueError: If input_names provided in predict doesn't match Predictor/signature def input names.

    """
    result = None
    # random_seed.set_random_seed(self._config.tf_random_seed)
    # sboshin: For multiple inputs, We will first check if the tensor is specified, if it isn't we will assume :0
    # Also The input function will return a list of features, that will be used as inputs. I believe the input function
    # should return a feed dict of tensor/op_names and their respective tensors
    feed_dict = self.return_feed_dict(model_feed_dict)
    result = self._internal_sess.run(self._fetch_tensors, feed_dict=feed_dict)
    return result
