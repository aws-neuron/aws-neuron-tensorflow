/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_NEURON_TF2XLA_COMPILE_H_
#define TENSORFLOW_NEURON_TF2XLA_COMPILE_H_

#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/neuron/tf2hlo/flags.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"

namespace tensorflow {
namespace neuron {

// CompileGraph compiles the graph_def into an object file containing a function
// that performs the graph operations.
//
// The XLA compilation options are specified in the flags.
Status CompileGraph(GraphDef graph_def, const tf2xla::Config& config,
                    const tensorflow::tfcompile::MainFlags& flags);

// The full compilation method, for reuse in a library setting.
Status Main(const tensorflow::tfcompile::MainFlags& flags);

Status ReadProtoFile(const string& fname, protobuf::Message* proto);

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_TF2XLA_COMPILE_H_
