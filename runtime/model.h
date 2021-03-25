/* Copyright Amazon Web Services and its Affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_NEURON_RUNTIME_MODEL_H_
#define TENSORFLOW_NEURON_RUNTIME_MODEL_H_

#include "engine.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace neuron {

class NeuronModel {
 public:
  NeuronModel();
  Status compute(OpKernelContext* ctx, const NodeDef& node_def,
                 const std::vector<const Tensor*>& input_tensors);
  ~NeuronModel();

 private:
  Status initialize(const NodeDef& node_def, const std::string& session_handle);
  tensorflow::mutex mutex_model_;
  NeuronEngine* neuron_engine_ = nullptr;
  uint32_t nn_id_ = NRT_INVALID_NN_ID;
  int64 estimated_cost_ = 0;
  ProfilerInterface profile_;
  thread::ThreadPool h2d_transfer_pool_;
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_MODEL_H_
