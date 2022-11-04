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
#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"

namespace tensorflow {
namespace neuron {
namespace convert {

xla::StatusOr<std::unique_ptr<xla::HloSnapshot>> GraphDefConfigToHloSnapshot(
    const GraphDef& graph_def, const tf2xla::Config& config) {
  // Converts the graph into an XLA computation.
  TF_ASSIGN_OR_RETURN(se::Platform* plat,
                      se::MultiPlatformManager::PlatformWithName("Host"));
  TF_ASSIGN_OR_RETURN(xla::CompileOnlyClient* client,
                      xla::ClientLibrary::GetOrCreateCompileOnlyClient(plat));
  xla::XlaComputation cpt;
  TF_RETURN_IF_ERROR(ConvertGraphDefToXla(graph_def, config, client, &cpt));
  return cpt.Snapshot();
}

Status VerifyHloModuleProto(const xla::HloModuleProto& hlo_module) {
  xla::ProgramShape program_shape(hlo_module.host_program_shape());
  xla::HloModuleConfig config(program_shape);
  return xla::CreateModuleFromProto(hlo_module, config).status();
}

namespace {

constexpr char tAwsNeuronCustomOp[] = "_AwsNeuronCustomOp";
constexpr char kCustomCallTarget[] = "custom_call_target";
constexpr char kBackendConfig[] = "backend_config";
constexpr char kOutputShapes[] = "output_shapes";

static void ToXlaShape(xla::Shape* xla_shape, const TensorShapeProto& shape) {
  for (const auto& dim : shape.dim()) {
    xla_shape->add_dimensions(dim.size());
  }
  xla::Layout* layout = xla_shape->mutable_layout();
  layout->set_format(xla::DENSE);
  for (int64_t m2m = shape.dim_size() - 1; m2m >= 0; --m2m) {
    layout->add_minor_to_major(m2m);
  }
}

class AwsNeuronCustomXlaOp : public XlaOpKernel {
 public:
  explicit AwsNeuronCustomXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) {
    VLOG(1) << ctx->num_inputs() << " inputs";
    VLOG(1) << ctx->num_outputs() << " outputs";
    std::vector<xla::XlaOp> operands;
    operands.reserve(ctx->num_inputs());
    for (int idx = 0; idx < ctx->num_inputs(); ++idx) {
      operands.push_back(ctx->Input(idx));
    }
    AttrValue_ListValue output_shapes = def().attr().at(kOutputShapes).list();
    xla::Shape xla_output_shape;
    if (1 == ctx->num_outputs()) {
      xla_output_shape.set_element_type(ctx->output_xla_type(0));
      ToXlaShape(&xla_output_shape, output_shapes.shape(0));
    } else {
      // Generating empty shape when ctx->num_outputs() == 0, which is expected
      for (int64_t idx = 0; idx < ctx->num_outputs(); ++idx) {
        xla::Shape* xla_shape = xla_output_shape.add_tuple_shapes();
        xla_shape->set_element_type(ctx->output_xla_type(idx));
        ToXlaShape(xla_shape, output_shapes.shape(idx));
      }
    }
    VLOG(1) << "xla_output_shape " << xla_output_shape.DebugString();
    const std::string& call_target = def().attr().at(kCustomCallTarget).s();
    const std::string& backend_config = def().attr().at(kBackendConfig).s();
    xla::XlaOp output = xla::CustomCall(ctx->builder(), call_target, operands,
                                        xla_output_shape, backend_config);
    ctx->SetOutput(0, output);
  }
};

REGISTER_XLA_OP(Name(tAwsNeuronCustomOp), AwsNeuronCustomXlaOp);
REGISTER_OP(tAwsNeuronCustomOp)
    .Attr("custom_call_target: string")
    .Attr("backend_config: string = \"\"")
    .Attr("input_dtypes: list(type) >= 0")
    .Attr("output_dtypes: list(type) >= 0")
    .Attr("output_shapes: list(shape)")
    .Input("input_tensors: input_dtypes")
    .Output("output_tensors: output_dtypes");

}  // namespace

}  // namespace convert
}  // namespace neuron
}  // namespace tensorflow
