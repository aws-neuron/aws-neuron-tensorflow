"""EIA Utilities for Saved Models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_utils

DEFAULT_TAGS = 'serve'

def _get_signature_def(signature_def_key, export_dir, tags):
    """Construct a `SignatureDef` proto."""
    if signature_def_key is None:
        print("Using DEFAULT_SERVING_SIGNATURE_DEF_KEY .....")
    signature_def_key = (
        signature_def_key or
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    metagraph_def = saved_model_utils.get_meta_graph_def(export_dir, tags)

    try:
        signature_def = metagraph_def.signature_def[signature_def_key]
    except KeyError as e:
        formatted_key = _DEFAULT_INPUT_ALTERNATIVE_FORMAT.format(
            signature_def_key)
        try:
            signature_def = metagraph_def.signature_def[formatted_key]
        except KeyError:
            raise ValueError(
                'Got signature_def_key "{}". Available signatures are {}. '
                'Original error:\n{}'.format(
                    signature_def_key, list(metagraph_def.signature_def), e))
            logging.warning('Could not find signature def "%s". '
                            'Using "%s" instead', signature_def_key, formatted_key)
    return signature_def

def _check_signature_arguments(signature_def_key,
                               signature_def,
                               input_names,
                               output_names):
  """Validates signature arguments Predictor."""
  signature_def_key_specified = signature_def_key is not None
  signature_def_specified = signature_def is not None
  input_names_specified = input_names is not None
  output_names_specified = output_names is not None
  if input_names_specified != output_names_specified:
    raise ValueError(
        'input_names and output_names must both be specified or both be '
        'unspecified.'
    )

  if (signature_def_key_specified + signature_def_specified +
      input_names_specified > 1):
    raise ValueError(
        'You must specify at most one of signature_def_key OR signature_def OR'
        '(input_names AND output_names).'
    )

def get_io_names_from_signature_def( model_dir,
                                                signature_def_key=None, 
                                                signature_def=None,
                                                input_names=None, 
                                                output_names=None,       
                                                tags=None):
    _check_signature_arguments(
    signature_def_key, signature_def, input_names, output_names)
    tags = tags or DEFAULT_TAGS
    if os.path.isdir(model_dir):
        # if saved model is provided.
        # signature def is only for saved model.
        if input_names is None:
            if signature_def is None:
                signature_def = _get_signature_def(signature_def_key, model_dir, tags)
            input_names = {k: v.name for k, v in signature_def.inputs.items()}
            output_names = {k: v.name for k, v in signature_def.outputs.items()}
    return input_names, output_names





