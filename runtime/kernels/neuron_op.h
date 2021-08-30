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

#ifndef TENSORFLOW_NEURON_RUNTIME_KERNELS_NEURON_OP_H_
#define TENSORFLOW_NEURON_RUNTIME_KERNELS_NEURON_OP_H_

#include "../model.h"
#include "../direct/function.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace neuron {

class NeuronOp : public OpKernel {
 public:
  explicit NeuronOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  Status grpc_runtime_status_;
  NeuronModel model_;
  NeuronFunction function_;
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_KERNELS_NEURON_OP_H_
