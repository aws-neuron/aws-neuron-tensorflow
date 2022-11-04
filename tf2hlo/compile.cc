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

#include "tensorflow/neuron/tf2hlo/compile.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/neuron/grappler/convert/tf2xla.h"

namespace tensorflow {
namespace neuron {

Status CompileGraph(GraphDef graph_def, const tf2xla::Config& config,
                    const tensorflow::tfcompile::MainFlags& flags) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloSnapshot> module,
                      convert::GraphDefConfigToHloSnapshot(graph_def, config));

  if (!flags.out_session_module.empty()) {
    // Serialize the HloSnapshot deterministically so that all the outputs of a
    // tf_library genrule are deterministic.
    const size_t size = module->ByteSizeLong();
    auto serialized = absl::make_unique<char[]>(size);
    TF_RET_CHECK(
        SerializeToBufferDeterministic(*module, serialized.get(), size));
    TF_RETURN_IF_ERROR(
        WriteStringToFile(Env::Default(), flags.out_session_module,
                          absl::string_view(serialized.get(), size)));
  }
  return Status::OK();
}

Status ReadProtoFile(const string& fname, protobuf::Message* proto) {
  if (absl::EndsWith(fname, ".pbtxt")) {
    return ReadTextProto(Env::Default(), fname, proto);
  } else {
    return ReadBinaryProto(Env::Default(), fname, proto);
  }
}

Status Main(const tensorflow::tfcompile::MainFlags& flags) {
  // Process config.
  tf2xla::Config config;
  if (flags.config.empty()) {
    return errors::InvalidArgument("Must specify --config");
  }
  TF_RETURN_IF_ERROR(ReadProtoFile(flags.config, &config));
  TF_RETURN_IF_ERROR(ValidateConfig(config));

  // Read and initialize the graph.
  if (flags.graph.empty()) {
    return errors::InvalidArgument("Must specify --graph");
  }
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(ReadProtoFile(flags.graph, &graph_def));

  return CompileGraph(std::move(graph_def), config, flags);
}

}  // namespace neuron
}  // namespace tensorflow
