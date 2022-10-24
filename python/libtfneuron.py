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
import os
from contextlib import contextmanager
from ctypes import cdll, c_char_p, string_at
from tensorflow.python.framework import errors


class LibTfNeuron:

    def __init__(self):
        tfn_path = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(tfn_path, 'libtfneuron.so')
        self.lib = None
        if os.path.isfile(path):
            self.lib = cdll.LoadLibrary(path)

    @property
    def available(self):
        return self.lib is not None

    def NeuronOptimize(self, s_graph_def):
        return self.run(self.lib.NeuronOptimizeGraphDef, s_graph_def)

    def NeuronConvert(self, s_graph_def):
        return self.run(self.lib.NeuronConvertGraphDef, s_graph_def)

    def NeuronTf2Xla(self, s_graph_def, s_config):
        return self.run(self.lib.NeuronTf2Xla, s_graph_def, s_config)

    def NeuronVerifyHlo(self, s_hlo_module):
        return self.run(self.lib.NeuronVerifyHlo, s_hlo_module)

    def run(self, func, *s_inputs):
        lib = self.lib
        with self.serialized() as serialized:
            c_s_inputs = []
            for s_in in s_inputs:
                c_s_inputs.append(c_char_p(s_in))
                c_s_inputs.append(len(s_in))
            func(serialized, *c_s_inputs)
            code = lib.NeuronSerializedStatusCode(serialized)
            if code != errors.OK:
                message = lib.NeuronSerializedStatusMessage(serialized)
                message = string_at(message)
                exception_type = errors.exception_type_from_error_code(code)
                raise exception_type(None, None, message.decode())
            s_output = lib.NeuronSerializedData(serialized)
            s_output_size = lib.NeuronSerializedSize(serialized)
            s_output = string_at(s_output, s_output_size)
        return s_output

    @contextmanager
    def serialized(self):
        serialized = self.lib.NewNeuronSerialized()
        try:
            yield serialized
        finally:
            self.lib.DeleteNeuronSerialized(serialized)


libtfneuron = LibTfNeuron()
