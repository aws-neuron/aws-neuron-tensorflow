/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_NEURON_CONVERT_TF2XLA_H_
#define TENSORFLOW_NEURON_CONVERT_TF2XLA_H_

#include <memory>

#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace neuron {
namespace convert {

xla::StatusOr<std::unique_ptr<xla::HloSnapshot>> GraphDefConfigToHloSnapshot(
    const GraphDef& graph_def, const tf2xla::Config& config);
Status VerifyHloModuleProto(const xla::HloModuleProto& hlo_module);

}  // namespace convert
}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_CONVERT_TF2XLA_H_
