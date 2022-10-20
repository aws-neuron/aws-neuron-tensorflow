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

    def run(self, func, s_graph_def):
        lib = self.lib
        with self.serialized() as serialized:
            c_s_graph_def = c_char_p(s_graph_def)
            func(serialized, c_s_graph_def, len(s_graph_def))
            code = lib.NeuronSerializedStatusCode(serialized)
            if code != errors.OK:
                message = lib.NeuronSerializedStatusMessage(serialized)
                message = string_at(message)
                exception_type = errors.exception_type_from_error_code(code)
                raise exception_type(None, None, message.decode())
            s_graph_def = lib.NeuronSerializedData(serialized)
            s_graph_def_size = lib.NeuronSerializedSize(serialized)
            s_graph_def = string_at(s_graph_def, s_graph_def_size)
        return s_graph_def

    @contextmanager
    def serialized(self):
        serialized = self.lib.NewNeuronSerialized()
        try:
            yield serialized
        finally:
            self.lib.DeleteNeuronSerialized(serialized)


libtfneuron = LibTfNeuron()
