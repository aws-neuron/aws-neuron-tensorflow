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

#ifndef TENSORFLOW_NEURON_RUNTIME_DIRECT_DYNAMIC_BATCH_H_
#define TENSORFLOW_NEURON_RUNTIME_DIRECT_DYNAMIC_BATCH_H_

#include <vector>
#include "executable_info.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace neuron {

class NeuronBatchSharder {
 public:
  Status Setup(const NeuronExecutableInfo& info,
               const std::vector<TensorShape>& input_shapes);
  const std::vector<TensorShape>& GetClientOutputShapes() const;
  bool CanSkip() { return can_skip_; }
  Status ShardInputs(std::vector<std::vector<Tensor>>* sharded_inputs,
                     const std::vector<Tensor>& inputs);
  Status ShardOutputs(std::vector<std::vector<Tensor>>* sharded_outputs,
                      const std::vector<Tensor>& outputs);

 private:
  Status ShardTensors(std::vector<std::vector<Tensor>>* sharded_tensors,
                      const std::vector<Tensor>& tensors,
                      const std::vector<bool>& tensors_need_sharding);
  bool can_skip_ = true;
  int neff_batch_size_ = 1;
  int client_batch_size_ = 1;
  std::vector<TensorShape> client_output_shapes_;
  std::vector<bool> inputs_need_sharding_;
  std::vector<bool> outputs_need_sharding_;
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DIRECT_DYNAMIC_BATCH_H_
