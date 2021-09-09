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

#include "registration.h"
#include "../device.h"
#include "../direct/placer.h"

namespace tensorflow {
namespace neuron {

class CheckRuntimeOp : public OpKernel {
 public:
  explicit CheckRuntimeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
      OP_REQUIRES_OK(context, NeuronCorePlacer::Singleton().GetStatus());
  }

  bool IsExpensive() override { return false; }
};

NEURON_REGISTER_KERNEL_BUILDER("CheckRuntimeOp", DEVICE_CPU, CheckRuntimeOp);

}  // namespace neuron
}  // namespace tensorflow
