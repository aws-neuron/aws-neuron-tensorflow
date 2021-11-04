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
import shutil
import subprocess
import unittest
import numpy as np
import tensorflow as tf
import tensorflow.neuron as tfn
from tensorflow.neuron.python.unittest_base import TestV1Only


class TestOpRegister(TestV1Only):

    export_dir_test = './simple_save_op_register'

    def test_simple(self):
        sess = tf.Session()
        ph = tf.placeholder(tf.float32, [1,1])
        output = tf.matmul(ph, np.ones([1,1]).astype(np.float32))
        output += 1.0
        shutil.rmtree(self.export_dir_test, ignore_errors=True)
        tfn.saved_model.simple_save(sess, self.export_dir_test, {ph.name: ph}, {output.name: output})

    def test_clean_process(self):
        subprocess.check_call([sys.executable, __file__, 'TestOpRegister.test_simple'])

    @unittest.skip('unsupported in libmode')
    def test_neuron_op_runtime_import(self):
        self.test_simple()
        subprocess.check_call([sys.executable, '-c', _content_neuron_op_runtime_import.format(self.export_dir_test)])


_content_neuron_op_runtime_import = '''
import os
import numpy as np
import tensorflow as tf
pred = tf.contrib.predictor.from_saved_model("{}")
if "NEURON_TF_COMPILE_ONLY" not in os.environ:
    feeds = {{name: np.zeros(tensor.shape, tensor.dtype.as_numpy_dtype) for name, tensor in pred.feed_tensors.items()}}
    print(pred(feeds))
'''


if __name__ == '__main__':
    unittest.main()
