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
import unittest
import os
import shutil
import subprocess
import time
from concurrent import futures
from distutils import spawn
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.core.framework.tensor_pb2 import TensorProto
import tensorflow.neuron as tfn


def can_import_serving():
    try:
        from tensorflow_serving.apis import prediction_service_pb2_grpc
        return bool(find_model_server())
    except ImportError:
        return False


def find_model_server():
    return spawn.find_executable('tensorflow_model_server') or spawn.find_executable('tensorflow_model_server_neuron')


class TestMeasurePerformance(unittest.TestCase):

    def test_simple(self):
        export_dir_test = './simple_save_performance_measurement'
        with tf.Session(graph=tf.Graph()) as sess:
            size = 1024
            ph = tf.placeholder(tf.float32, [size, size])
            output = tf.matmul(ph, np.ones([size, size]).astype(np.float32))
            shutil.rmtree(export_dir_test, ignore_errors=True)
            tf.saved_model.simple_save(sess, export_dir_test, {ph.name: ph}, {output.name: output})

        def one_thread(pred, feed_dict):
            for _ in range(1000):
                pred(model_feed_dict)

        max_workers = 4
        pred_list = [tfn.predictor.from_saved_model(export_dir_test) for _ in range(max_workers)]
        model_feed_dict = {ph.name: np.random.rand(size, size)}
        with tfn.measure_performance():
            executor = futures.ThreadPoolExecutor(max_workers=max_workers)
            fut_list = []
            for pred in pred_list:
                fut = executor.submit(one_thread, pred, model_feed_dict)
                fut_list.append(fut)
            for fut in fut_list:
                fut.result()

    def test_func(self):
        export_dir_test = './simple_save_performance_measurement_func'
        with tf.Session(graph=tf.Graph()) as sess:
            size = 1024
            ph = tf.placeholder(tf.float32, [size, size])
            output = tf.matmul(ph, np.ones([size, size]).astype(np.float32))
            shutil.rmtree(export_dir_test, ignore_errors=True)
            tf.saved_model.simple_save(sess, export_dir_test, {ph.name: ph}, {output.name: output})

        def one_thread(pred, model_feed_dict):
            for _ in range(1000):
                pred(model_feed_dict)

        pred = tfn.predictor.from_saved_model(export_dir_test)
        model_feed_dict = {ph.name: np.random.rand(size, size)}
        with tfn.measure_performance(func=pred) as pred:
            one_thread(pred, model_feed_dict)


class TestMeasurePerformanceServing(unittest.TestCase):

    def setUp(self):
        self.export_dir_base = os.path.realpath('./simple_save_performance_measurement_serving')
        export_dir_test = os.path.join(self.export_dir_base, '1')
        with tf.Session(graph=tf.Graph()) as sess:
            self.size = 1000
            ph = tf.placeholder(tf.float32, [self.size, self.size])
            output = tf.matmul(ph, np.ones([self.size, self.size]).astype(np.float32))
            shutil.rmtree(export_dir_test, ignore_errors=True)
            tf.saved_model.simple_save(sess, export_dir_test, {'input': ph}, {'output': output})
        model_server_path = find_model_server()
        self.port = '8999'
        cmd = [find_model_server(), '--model_base_path={}'.format(self.export_dir_base), '--port={}'.format(self.port)]
        self.model_server_proc = subprocess.Popen(cmd)
        time.sleep(1)

    def tearDown(self):
        self.model_server_proc.terminate()
        self.model_server_proc.communicate()

    @unittest.skipIf(not can_import_serving(), 'not runnable without tf-serving')
    def test_serving(self):
        import grpc
        from tensorflow_serving.apis import predict_pb2
        from tensorflow_serving.apis import prediction_service_pb2_grpc
        channel = grpc.insecure_channel('localhost:{}'.format(self.port))
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        data = np.random.rand(self.size, self.size).astype(np.float32)
        tensor_proto = TensorProto()
        tensor_proto.dtype = tf.float32.as_datatype_enum
        for size in data.shape:
            tensor_proto.tensor_shape.dim.add().size = size
        tensor_proto.tensor_content = data.tobytes()
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'default'
        request.inputs['input'].CopyFrom(tensor_proto)
        with tfn.measure_performance():
            for _ in range(1000):
                result = stub.Predict(request).outputs


if __name__ == '__main__':
    unittest.main()
