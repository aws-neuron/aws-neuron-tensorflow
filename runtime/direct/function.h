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

#ifndef TENSORFLOW_NEURON_RUNTIME_DIRECT_FUNCTION_H_
#define TENSORFLOW_NEURON_RUNTIME_DIRECT_FUNCTION_H_

#include <memory>
#include <string>
#include "../macros.h"
#include "executable.h"
#include "executable_info.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace neuron {

class NeuronFunction {
 public:
  NeuronFunction();
  Status Run(OpKernelContext* ctx, const NodeDef& node_def);

 private:
  Status MaybeInit(const NodeDef& node_def, const std::string& session_handle);
  Status SetupInputsOutputs(OpKernelContext* ctx, const NodeDef& node_def,
                            std::vector<Tensor>* inputs,
                            std::vector<Tensor>* outputs);
  Status SetupInputs(OpKernelContext* ctx, const NodeDef& node_def,
                     std::vector<Tensor>* inputs);
  Status SetupOutputs(OpKernelContext* ctx, const NodeDef& node_def,
                      std::vector<Tensor>* outputs);
  tensorflow::mutex mu_;
  NeuronExecutableInfo info_;
  std::unique_ptr<NeuronDataParallelExecutable> exe_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronFunction);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DIRECT_FUNCTION_H_
