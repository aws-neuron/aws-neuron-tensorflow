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
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace neuron {

Status CompileGraph(GraphDef graph_def, const tf2xla::Config& config,
                    const tensorflow::tfcompile::MainFlags& flags) {
  // Converts the graph into an XLA computation.
  // TODO(toddw): Should we let the user pick the XLA cpu vs. gpu client?
  se::Platform* cpu_platform =
      se::MultiPlatformManager::PlatformWithName("Host").ValueOrDie();
  xla::CompileOnlyClient* client =
      xla::ClientLibrary::GetOrCreateCompileOnlyClient(cpu_platform)
          .ValueOrDie();
  xla::XlaComputation computation;
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToXla(std::move(graph_def), config, client, &computation));

  if (!flags.out_session_module.empty()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloSnapshot> module,
                        computation.Snapshot());
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

static Status ReadProtoFile(const string& fname, protobuf::Message* proto) {
  if (absl::EndsWith(fname, ".pbtxt")) {
    return ReadTextProto(Env::Default(), fname, proto);
  } else {
    return ReadBinaryProto(Env::Default(), fname, proto);
  }
}

// Replaces {{tag.type tag.name}} in the error message with tag_name.
// TODO(bixia): We currently only handlge tag.type == "node".
//
// In the error message, a graph node is represented as {{tag.type, tag.name}},
// to allow a Python debugger to insert source information about the graph node.
// For example, a Python add expression may be represented as
// {{node, x_y_sum}} = Add(x, y) in the error message. See routine interpolate
// in tensorflow/python/framework/error_interpolation.py for more detail.
static std::string InterpolateErrorMessage(std::string message) {
  // See _NAME_REGEX in tensorflow/python/framework/error_interpolation.py
  // Change "prefix {{node tag.name}} suffix" to "prefix tag.name suffix".
  static LazyRE2 pattern{"(.*){{node (.*)}}(.*)"};
  RE2::GlobalReplace(&message, *pattern, "\\1\\2\\3");

  return message;
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

  Status status = CompileGraph(std::move(graph_def), config, flags);
  if (!status.ok()) {
    return Status(status.code(),
                  InterpolateErrorMessage(status.error_message()));
  }
  return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow
