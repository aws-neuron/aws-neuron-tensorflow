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
import collections
import os
import argparse
import shlex
from contextlib import contextmanager
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.grappler import tf_optimizer


@contextmanager
def logging_show_info():
    verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.INFO)
    try:
        yield
    finally:
        logging.set_verbosity(verbosity)


def most_popular_namescope(all_node_names):
    all_splitted = [name.split('/') for name in all_node_names]
    max_level = max(len(splitted) for splitted in all_splitted)
    most_popular_namescope = []
    max_popularity = 0
    for lvl in range(max_level):
        names = [splitted[lvl] for splitted in all_splitted if lvl < len(splitted)]
        (scope, popularity), = collections.Counter(names).most_common(1)
        if popularity >= max_popularity:
            most_popular_namescope.append(scope)
            max_popularity = popularity
        else:
            break
    return '/'.join(most_popular_namescope)


def model_conversion_report(model_dir, new_model_dir, on_neuron_ratio):
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
    else:
        logging.warning('Converted {} {}'.format(converted_msg, warning_msg))


def parse_neuron_cc_flags(args=None, flag_set=None):
    if args is None:
        neuron_cc_flags = os.environ.get('NEURON_CC_FLAGS', '')
        args = shlex.split(neuron_cc_flags)
    parser = argparse.ArgumentParser()

    def maybe_add_argument(flag, *args, **kwargs):
        if flag_set is None or flag in flag_set:
            parser.add_argument(flag, *args, **kwargs)

    maybe_add_argument('--workdir', dest='dump_prefix', default=None)
    verbose_choices = ['debug', 'info', 'warning', 'error', 'critical']
    verbose_choices.extend([key.upper() for key in verbose_choices])
    maybe_add_argument('--verbose', choices=verbose_choices, default=None)
    maybe_add_argument('--dynamic-batch-size', action='store_true')
    maybe_add_argument('--fp32-cast', default='matmult-fp16')
    maybe_add_argument('--neuroncore-pipeline-cores', default=None)
    maybe_add_argument('--reduce-neff-size', action='store_true')
    tfn_args, compiler_args = parser.parse_known_args(args)
    verbose = getattr(tfn_args, 'verbose', None)
    if verbose is None:
        # set between WARN and ERROR to trigger neuron-cc progress bar
        tfn_args.verbose = tfn_args.log_level = 35
    else:
        verbose_map = {
            'debug': 'DEBUG',
            'info': 'INFO',
            'warning': 'WARN',
            'error': 'ERROR',
            'critical': 'FATAL',
        }
        tfn_args.log_level = getattr(logging, verbose_map.get(verbose.lower(), 'WARN'))
    return tfn_args, compiler_args

def _assert_compiler_success_func(wfunc):
    assert any(op.type == 'NeuronOp' for op in wfunc.graph.get_operations())

@contextmanager
def change_grappler_logging_level_according_to_cc_flags():
    tfn_args, _ = parse_neuron_cc_flags()
    verbose = tfn_args.log_level < logging.WARN
    original_optimize_graph = tf_optimizer.OptimizeGraph

    def optimize_graph(*args, **kwargs):
        if len(args) > 2:
            args = args[:2]
        kwargs.pop('verbose', None)
        return original_optimize_graph(*args, **kwargs, verbose=verbose)

    tf_optimizer.OptimizeGraph = optimize_graph
    try:
        yield
    finally:
        tf_optimizer.OptimizeGraph = original_optimize_graph


def decorate_methods_with(decorator):

    def decorate_methods(cls):
        for key, value in vars(cls).items():
            if callable(value) and key != '__init__':
                setattr(cls, key, decorator(value))
        return cls

    return decorate_methods
