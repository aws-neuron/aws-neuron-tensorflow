# Copyright 2020 AWS Neuron. All Rights Reserved.
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

import sys as _sys
import os as _os
_package_install_dir = _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))
_sys.path.append(_os.path.join(_package_install_dir, 'tensorflow-plugins'))
from tensorflow.neuron.python import graph_util
_sys.path.pop()  # assume graph_util does not hack _sys.path (which is true as of writing)
from tensorflow.neuron.python import saved_model
from tensorflow.neuron.python import predictor
from tensorflow.neuron.python.fuse import fuse
from tensorflow.neuron.python.performance import measure_performance
from tensorflow.neuron.ops.gen_neuron_op import neuron_op
