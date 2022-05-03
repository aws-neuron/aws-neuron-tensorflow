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
# Note: this file becomes tensorflow/neuron/__init__.py and tensorflow_core/neuron/__init__.py
# in the plugin package

import sys as _sys
from distutils.version import LooseVersion as _LooseVersion
from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader

def _forward_module(old_name):
    existing_name = old_name.replace('tensorflow.neuron', 'tensorflow_neuron')
    _module = _LazyLoader(old_name.split('.')[-1], globals(), existing_name)
    return _sys.modules.setdefault(old_name, _module)

_forward_module('tensorflow.neuron.python')

from tensorflow_neuron import __version__
from tensorflow_neuron.python import graph_util
if _LooseVersion(__version__) < _LooseVersion('2.0.0'):
    from tensorflow_neuron.python import saved_model
else:
    from tensorflow_neuron.python import saved_model_v2 as saved_model
    from tensorflow_neuron.python import _trace
    from tensorflow_neuron.python._trace import trace
from tensorflow_neuron.python import predictor
from tensorflow_neuron.python.fuse import fuse
from tensorflow_neuron.python.performance import measure_performance
