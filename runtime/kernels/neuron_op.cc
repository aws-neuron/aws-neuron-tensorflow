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

#include "neuron_op.h"

namespace tensorflow {
namespace register_kernel {

static KernelDef *neuron_kernel(const std::string &type) {
    // Need to allocate from an existing one to workaround the bug that setting one attribute
    // can affect other attributes
    KernelDef *kernel = new KernelDef(GetRegisteredKernelsForOp("IdentityN").kernel(0));
    kernel->clear_op();
    kernel->clear_device_type();
    kernel->clear_constraint();
    kernel->clear_host_memory_arg();
    kernel->clear_label();
    kernel->clear_priority();
    kernel->set_op(GetRegisteredKernelsForOp(type).kernel_size() ? "_no_register" : type);
    kernel->set_device_type(DEVICE_CPU);
    return kernel;
}

class NeuronName {
public:
    const KernelDef *Build() {
        return neuron_kernel("NeuronOp");
    }
};

}  // namespace register_kernel

namespace neuron {

void NeuronOp::Compute(OpKernelContext *ctx) {
    std::vector<const Tensor*> input_tensors(ctx->num_inputs());
    for (auto idx = 0; idx < ctx->num_inputs(); ++idx) {
        input_tensors[idx] = &ctx->input(idx);
    }
    std::vector<Tensor> temp_tensors(ctx->num_inputs());
    if (def().attr().count("_input_shuffles")) {
        const auto &input_shuffles = def().attr().at("_input_shuffles").list();
        VLOG(1) << input_shuffles.tensor_size() << " input shuffle indices";
        for (auto idx = 0; idx < ctx->num_inputs(); ++idx) {
            uint64 start_timestamp = Env::Default()->NowMicros();
            const Tensor &src = ctx->input(idx);
            Tensor *dst = &temp_tensors[idx];
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(src.dtype(), src.shape(), dst));
            const TensorProto &shuffle = input_shuffles.tensor(idx);
            if (shuffle.int64_val_size() != src.NumElements()) {
                VLOG(1) << "skipping shuffle at input " << idx;
                continue;
            }
            OP_REQUIRES_OK(ctx, tensor_shuffle(dst, src, shuffle));
            input_tensors[idx] = dst;
            uint64 elapsed = Env::Default()->NowMicros() - start_timestamp;
            VLOG(1) << "input shuffle at " << idx << " took " << elapsed << " us";
        }
    }
    OP_REQUIRES_OK(ctx, model_.compute(ctx, def(), input_tensors));
}

// need to override kernel_builder as NeuronName to prevent multiple kernel registrations
#if TF_VERSION_LESS_THAN(2, 4)
REGISTER_KERNEL_BUILDER(NeuronName(), neuron::NeuronOp);
#else
REGISTER_KERNEL_BUILDER_IMPL_2("NeuronOp", register_kernel::NeuronName(), false, NeuronOp);
#endif

}  // namespace neuron
}  // namespace tensorflow
