
import sys
import shutil
import subprocess
import unittest
import numpy as np
import tensorflow as tf
import tensorflow.neuron as tfn


class TestOpRegister(unittest.TestCase):

    def test_simple(self):
        sess = tf.Session()
        ph = tf.placeholder(tf.float32, [1,1])
        output = tf.matmul(ph, np.ones([1,1]).astype(np.float32))
        output += 1.0
        export_dir_test = './simple_save_op_register'
        shutil.rmtree(export_dir_test, ignore_errors=True)
        tfn.saved_model.simple_save(sess, export_dir_test, {ph.name: ph}, {output.name: output})

    def test_clean_process(self):
        subprocess.check_call([sys.executable, __file__, 'TestOpRegister.test_simple'])


if __name__ == '__main__':
    unittest.main()
