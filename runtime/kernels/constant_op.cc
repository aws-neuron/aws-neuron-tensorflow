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

namespace tensorflow {
namespace neuron {

class ConstantOp : public OpKernel {
 public:
  explicit ConstantOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), tensor_(ctx->output_type(0)) {
    const TensorProto* proto = nullptr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
    OP_REQUIRES_OK(ctx, ctx->device()->MakeTensorFromProto(
                            *proto, AllocatorAttributes(), &tensor_));
    OP_REQUIRES(
        ctx, ctx->output_type(0) == tensor_.dtype(),
        errors::InvalidArgument(
            "Type mismatch between value (", DataTypeString(tensor_.dtype()),
            ") and dtype (", DataTypeString(ctx->output_type(0)), ")"));
  }

  void Compute(OpKernelContext* ctx) override {
    ctx->set_output(0, tensor_);
    if (TF_PREDICT_FALSE(ctx->track_allocations())) {
      ctx->record_persistent_memory_allocation(tensor_.AllocatedBytes());
    }
  }

  bool IsExpensive() override { return false; }

 private:
  Tensor tensor_;
  TF_DISALLOW_COPY_AND_ASSIGN(ConstantOp);
};

NEURON_REGISTER_KERNEL_BUILDER("Const", DEVICE_NEURON, ConstantOp);

}  // namespace neuron
}  // namespace tensorflow
