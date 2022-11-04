/* Copyright Amazon Web Services and its Affiliates. Reserved.

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
#include "graph_constructor_wrapper.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

Status BackEdgeHelper::Remove(Graph* graph) {
  if (graph_ != nullptr) {
    return errors::Internal("BackEdgeHelper duplicate call to Remove.");
  }
  graph_ = graph;
  for (Node* n : graph_->nodes()) {
    if (n->IsMerge()) {
      for (const Edge* e : n->in_edges()) {
        if (e->src()->IsNextIteration()) {
          back_edges_.push_back(
              BackEdge{e, e->src(), e->src_output(), e->dst(), e->dst_input()});
        }
      }
    }
  }
  for (const BackEdge& be : back_edges_) {
    graph_->RemoveEdge(be.edge);
  }
  return Status::OK();
}

const std::vector<BackEdgeHelper::BackEdge>& BackEdgeHelper::RemovedEdges()
    const {
  return back_edges_;
}

Status BackEdgeHelper::Replace() {
  if (graph_ == nullptr) {
    return errors::Internal("BackEdgeHelper Replace called before Remove.");
  }
  if (replaced_) {
    return errors::Internal("BackEdgeHelper Replace called more than once.");
  }
  replaced_ = true;
  for (const BackEdge& be : back_edges_) {
    graph_->AddEdge(be.src, be.src_output, be.dst, be.dst_input);
  }
  return Status::OK();
}

namespace {

const char kNeuronInferredShapes[] = "_aws_neuron_inferred_shapes";

// Converts a shape inference handle to a PartialTensorShape.
Status ShapeHandleToTensorShape(shape_inference::InferenceContext* context,
                                const shape_inference::ShapeHandle& handle,
                                PartialTensorShape* shape) {
  // The default is already unknown
  if (!context->RankKnown(handle)) return Status::OK();

  std::vector<int64> dims(context->Rank(handle));
  for (int32 i = 0; i < dims.size(); ++i) {
    dims[i] = context->Value(context->Dim(handle, i));
  }
  return PartialTensorShape::MakePartialShape(dims.data(), dims.size(), shape);
}

Status PropagateShapes(Graph* graph,
                       const std::map<int, InferredShape>& arg_shapes,
                       const std::vector<BackEdgeHelper::BackEdge>& back_edges,
                       ShapeRefiner* shape_refiner) {
  std::map<const Node*, const Node*> merge_to_next_iteration;
  for (const auto& e : back_edges) {
    if (e.src->IsNextIteration() && e.dst->IsMerge()) {
      merge_to_next_iteration[e.dst] = e.src;
    }
  }

  // Visits the nodes in topological order (reverse post-order), inferring
  // shapes.
  // TODO(phawkins): handle cyclic graphs.
  std::vector<Node*> order;
  GetReversePostOrder(*graph, &order);

  std::unordered_map<std::string, int64> resolved_ints;
  std::unordered_map<std::string, TensorShape> resolved_shapes;
  typedef gtl::InlinedVector<int64, 4> GtlInt64Vector;
  std::unordered_map<std::string, GtlInt64Vector> resolved_vectors;
  std::unordered_map<std::string, int64> resolved_tensor_array_sizes;
  std::unordered_map<std::string, int64> resolved_range_sizes;
  for (Node* n : order) {
    // Ignore the status returned by the shape_refiner. We want the best effort
    // shapes, even if no shape function is registered for a node.
    Status status = shape_refiner->AddNode(n);
    if (!status.ok()) {
      VLOG(1) << "Shape inference failed for node " << n->name() << ": "
              << status;
    } else {
      shape_inference::InferenceContext* context = shape_refiner->GetContext(n);
      for (int i = 0; i < n->num_outputs(); i++) {
        shape_inference::ShapeHandle handle = context->output(i);
        VLOG(4) << "Output " << i << " for node " << n->name() << ": "
                << context->DebugString(handle);
        auto& attr = n->def().attr();
        if (attr.count(kNeuronInferredShapes)) {
          auto& shape_list = attr.at(kNeuronInferredShapes).list().shape();
          if (i < shape_list.size()) {
            PartialTensorShape shape(shape_list[i]);
            TF_RETURN_IF_ERROR(
                context->MakeShapeFromPartialTensorShape(shape, &handle));
            context->set_output(i, handle);
            if (shape.IsFullyDefined()) {
              std::string tensor_name = n->name() + ":" + std::to_string(i);
              std::string gd_tensor_name = i == 0 ? n->name() : tensor_name;
              resolved_shapes[gd_tensor_name] = shape_list[i];
              VLOG(1) << "Set fully defined shape for node " << n->name()
                      << " at output port " << i;
            }
          }
        }
      }
      if (n->type_string() == "TensorArrayGatherV3") {
        std::string input1_name = n->def().input(1);
        if (resolved_range_sizes.count(input1_name)) {
          PartialTensorShape shape(n->def().attr().at("element_shape").shape());
          if (shape.IsFullyDefined()) {
            shape.InsertDim(0, resolved_range_sizes[input1_name]);
            shape_inference::ShapeHandle handle = context->output(0);
            TF_RETURN_IF_ERROR(
                context->MakeShapeFromPartialTensorShape(shape, &handle));
            context->set_output(0, handle);
            VLOG(1) << "Inferred fully defined shape of " << n->name()
                    << " using TensorArray operator shape inference mechanism";
          }
        }
      }
    }

    if (n->type_string() == "Const") {
      const auto& tensor = n->def().attr().at("value").tensor();
      if (tensor.dtype() == DT_INT32 && tensor.int_val_size() == 1) {
        int64 int_val = tensor.int_val(0);
        resolved_ints[n->name()] = int_val;
        VLOG(2) << "filled resolved_ints[" << n->name() << "] with " << int_val;
      }
    }

    if (n->type_string() == "Shape") {
      std::string input0_name = n->def().input(0);
      if (resolved_shapes.count(input0_name)) {
        auto& shape = resolved_shapes[input0_name];
        resolved_vectors[n->name()] = shape.dim_sizes();
        VLOG(2) << "filled resolved_vectors[" << n->name() << "] with vector "
                << shape;
      }
    }

    if (n->type_string() == "StridedSlice") {
      const auto& n_def = n->def();
      const auto& attr = n_def.attr();
      if (attr.at("Index").type() == DT_INT32 &&
          attr.at("T").type() == DT_INT32 && attr.at("begin_mask").i() == 0 &&
          attr.at("ellipsis_mask").i() == 0 && attr.at("end_mask").i() == 0 &&
          attr.at("new_axis_mask").i() == 0 &&
          attr.at("shrink_axis_mask").i() == 1 &&
          resolved_vectors.count(n_def.input(0)) &&
          resolved_ints.count(n_def.input(1)) &&
          resolved_ints.count(n_def.input(2)) &&
          resolved_ints.count(n_def.input(3))) {
        int64 start = resolved_ints[n_def.input(1)];
        int64 end = resolved_ints[n_def.input(2)];
        int64 step = resolved_ints[n_def.input(3)];
        auto& vector = resolved_vectors[n_def.input(0)];
        if (end - start == 1 && step == 1 && start < (int64)vector.size()) {
          int64 int_val = vector[start];
          resolved_ints[n->name()] = int_val;
          VLOG(2) << "filled resolved_ints[" << n->name() << "] with "
                  << int_val;
        }
      }
    }

    if (n->type_string() == "TensorArrayV3") {
      const auto& n_def = n->def();
      const auto& attr = n_def.attr();
      if (!attr.at("dynamic_size").b() && resolved_ints.count(n_def.input(0))) {
        int64 int_val = resolved_ints[n_def.input(0)];
        resolved_tensor_array_sizes[n->name()] = int_val;
        VLOG(2) << "filled resolved_tensor_array_sizes[" << n->name()
                << "] with " << int_val;
      }
    }

    if (n->type_string() == "TensorArraySizeV3") {
      const auto& n_def = n->def();
      if (resolved_tensor_array_sizes.count(n_def.input(0))) {
        int64 int_val = resolved_tensor_array_sizes[n_def.input(0)];
        resolved_tensor_array_sizes[n->name()] = int_val;
        VLOG(2) << "filled resolved_tensor_array_sizes[" << n->name()
                << "] with " << int_val;
      }
    }

    if (n->type_string() == "Range") {
      const auto& n_def = n->def();
      const auto& attr = n_def.attr();
      if (attr.at("Tidx").type() == DT_INT32 &&
          resolved_ints.count(n_def.input(0)) &&
          resolved_tensor_array_sizes.count(n_def.input(1)) &&
          resolved_ints.count(n_def.input(2))) {
        int64 start = resolved_ints[n_def.input(0)];
        int64 end = resolved_tensor_array_sizes[n_def.input(1)];
        int64 step = resolved_ints[n_def.input(2)];
        int64 diff = end - start;
        int64 num_elements = diff / step + diff % step;
        resolved_range_sizes[n->name()] = num_elements;
        VLOG(2) << "filled resolved_range_sizes[" << n->name() << "] with "
                << num_elements;
      }
    }

    if (n->type_string() == "_Arg") {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      auto it = arg_shapes.find(index);
      if (it != arg_shapes.end()) {
        const InferredShape& arg_shape = it->second;
        shape_inference::InferenceContext* context =
            shape_refiner->GetContext(n);

        if (arg_shape.handle_type != DT_INVALID) {
          shape_inference::ShapeHandle handle;
          TF_RETURN_IF_ERROR(context->MakeShapeFromPartialTensorShape(
              arg_shape.handle_shape, &handle));

          // Sets the shape and type of the variable's value.
          context->set_output_handle_shapes_and_types(
              0, std::vector<shape_inference::ShapeAndType>{
                     {handle, arg_shape.handle_type}});
        }

        shape_inference::ShapeHandle handle;
        TF_RETURN_IF_ERROR(
            context->MakeShapeFromPartialTensorShape(arg_shape.shape, &handle));
        TF_RETURN_IF_ERROR(shape_refiner->SetShape(n, 0, handle));
      }
    }

    // Sometimes we have VariableShape nodes in while loop (after Enter nodes).
    // They won't be constant-folded because TensorFlow constant folding does
    // not handle Enter nodes (and thus does not handle any nodes after Enter
    // nodes). We try to replace such VariableShape nodes with Const nodes here.
    if (n->type_string() == "VariableShape") {
      shape_inference::InferenceContext* context = shape_refiner->GetContext(n);
      auto handle_shapes_and_types = context->input_handle_shapes_and_types(0);
      if (handle_shapes_and_types && !handle_shapes_and_types->empty()) {
        shape_inference::ShapeHandle handle =
            handle_shapes_and_types->at(0).shape;
        TensorShapeProto shape_proto;
        context->ShapeHandleToProto(handle, &shape_proto);
        if (!shape_proto.unknown_rank()) {
          NodeDef const_def;
          const_def.set_op("Const");
          Node* var_node;
          TF_RETURN_IF_ERROR(n->input_node(0, &var_node));
          const_def.set_name(
              graph->NewName(absl::StrCat("var_shape_", var_node->name())));
          DataType dtype = n->output_type(0);
          AddNodeAttr("dtype", dtype, &const_def);
          TensorProto value;
          value.set_dtype(dtype);
          value.mutable_tensor_shape()->add_dim()->set_size(
              shape_proto.dim_size());
          for (const auto& dim : shape_proto.dim()) {
            if (dtype == DT_INT32) {
              value.add_int_val(dim.size());
            } else {
              value.add_int64_val(dim.size());
            }
          }
          AddNodeAttr("value", value, &const_def);
          for (auto const& attr : n->attrs()) {
            if (*attr.first.begin() == '_') {
              AddNodeAttr(attr.first, attr.second, &const_def);
            }
          }

          Status s;
          Node* const_node = graph->AddNode(const_def, &s);
          TF_RETURN_IF_ERROR(s);

          graph->AddControlEdge(var_node, const_node);
          std::vector<const Edge*> out_edges(n->out_edges().begin(),
                                             n->out_edges().end());
          for (const Edge* e : out_edges) {
            if (e->IsControlEdge()) {
              graph->AddControlEdge(const_node, e->dst());
              graph->RemoveEdge(e);
            } else {
              Node* dst = e->dst();
              int dst_input = e->dst_input();
              graph->RemoveEdge(e);
              graph->AddEdge(const_node, 0, dst, dst_input);
            }
          }
        }
      }
    }

    // Merge node causes a loop so we remove NextIteration->Merge edge before
    // performing shape inference. But removing those edges also prevents us
    // from inferring output shape for Merge node (we need shapes for all its
    // inputs).
    // For loop invariant resource input's Merge node, we set output resource
    // shape as Enter node's resource shape.
    // TODO(b/129367850): clean this up.
    if (n->IsMerge() && n->output_type(0) == DT_RESOURCE) {
      // Check if this is a loop invariant input's Merge node. We do it by
      // checking if corresponding NextIteration node comes from Switch node
      // directly.
      auto iter = merge_to_next_iteration.find(n);
      if (iter != merge_to_next_iteration.end()) {
        const Node *next_iter = iter->second, *node = next_iter;
        do {
          TF_RETURN_IF_ERROR(node->input_node(0, &node));
        } while (node->IsIdentity());
        const Node* switch_input;
        bool is_loop_invariant = node->IsSwitch() &&
                                 node->input_node(0, &switch_input).ok() &&
                                 switch_input == n;
        if (is_loop_invariant) {
          shape_inference::InferenceContext* context =
              shape_refiner->GetContext(n);
          for (int i = 0; i < n->num_inputs(); i++) {
            const Node* input_node;
            if (n->input_node(i, &input_node).ok()) {
              auto shapes_and_types = context->input_handle_shapes_and_types(i);
              if (shapes_and_types) {
                context->set_output_handle_shapes_and_types(0,
                                                            *shapes_and_types);
              }
              break;
            }
          }
        }
      }
    }
  }
  return Status::OK();
}

// Store the shapes of the output tensors in a map
Status StoreOutputShapes(const Graph& graph, const ShapeRefiner& shape_refiner,
                         GraphShapeInfo* shape_info) {
  for (const Node* node : graph.nodes()) {
    shape_inference::InferenceContext* context = shape_refiner.GetContext(node);
    if (!context) continue;

    auto& outputs = (*shape_info)[node->name()];
    outputs.resize(context->num_outputs());
    for (int i = 0; i < context->num_outputs(); ++i) {
      auto& output = outputs[i];
      TF_RETURN_IF_ERROR(
          ShapeHandleToTensorShape(context, context->output(i), &output.shape));

      const auto* handle_shapes_and_types =
          context->output_handle_shapes_and_types(i);
      if (handle_shapes_and_types != nullptr) {
        if (handle_shapes_and_types->size() == 1) {
          TF_RETURN_IF_ERROR(ShapeHandleToTensorShape(
              context, (*handle_shapes_and_types)[0].shape,
              &output.handle_shape));
          output.handle_type = (*handle_shapes_and_types)[0].dtype;
        } else {
          // otherwise, it may be resource like a Queue, which can have
          // multiple shapes and types represented by a single handle.
        }
      }
      VLOG(4) << node->name() << " output " << i << " shape"
              << output.shape.DebugString() << " handle_type "
              << DataTypeString(output.handle_type) << " handle_shape "
              << output.handle_shape.DebugString();
    }
  }
  return Status::OK();
}

}  // namespace

Status InferShapes(Graph* graph, const std::map<int, InferredShape>& arg_shapes,
                   const tensorflow::FunctionLibraryDefinition* fnlib_def,
                   GraphShapeInfo* shape_info) {
  ShapeRefiner shape_refiner(graph->versions(), graph->op_registry());
  shape_refiner.set_require_shape_inference_fns(false);
  // TODO(dlibenzi): Verify if it is worth trying to infer shaped within
  // functions. Some functions can be called at multiple locations with
  // difference shapes, which will trigger a shape inference based on the
  // arguments passed at the first call.
  // shape_refiner.set_function_library_for_shape_inference(fnlib_def);

  // ShapeRefiner requires that all inputs of a node are present when
  // ShapeRefiner::AddNode is called. To get at least some shape information in
  // loops, we temporarily remove loop backedges and add them back again after
  // the shape inference is complete.
  BackEdgeHelper back_edge;
  TF_RETURN_IF_ERROR(back_edge.Remove(graph));
  TF_RETURN_IF_ERROR(PropagateShapes(graph, arg_shapes,
                                     back_edge.RemovedEdges(), &shape_refiner));
  TF_RETURN_IF_ERROR(back_edge.Replace());

  // Currently information does not flow "backward" from consumers to producers
  // in the shape inference, but we consume the shapes in a second pass in case
  // backward information flow is added in the future.
  return StoreOutputShapes(*graph, shape_refiner, shape_info);
}

xla::StatusOr<InferredShape> MergeInferredShapes(const InferredShape& a,
                                                 const InferredShape& b) {
  InferredShape result;
  TF_RETURN_IF_ERROR(a.shape.MergeWith(b.shape, &result.shape));

  if (a.handle_type == DT_INVALID) {
    result.handle_type = b.handle_type;
  } else if (b.handle_type == DT_INVALID) {
    result.handle_type = a.handle_type;
  } else if (a.handle_type == b.handle_type) {
    result.handle_type = a.handle_type;
  } else {
    return errors::InvalidArgument(
        "Mismatched resource types: ", DataTypeString(a.handle_type), " vs. ",
        DataTypeString(b.handle_type));
  }
  TF_RETURN_IF_ERROR(
      a.handle_shape.MergeWith(b.handle_shape, &result.handle_shape));
  return result;
}

namespace neuron {
namespace convert {

namespace {

Status PerformStaticShapeInferenceBeforeEncapsulation(Graph* graph) {
  // Perform shape inference.
  std::map<int, InferredShape> arg_shapes;
  GraphShapeInfo shape_info;
  TF_RETURN_IF_ERROR(
      InferShapes(graph, arg_shapes, /*fnlib_def=*/nullptr, &shape_info));

  // Add attribute for output shapes.
  auto node_name_index = graph->BuildNodeNameIndex();
  for (auto iter : shape_info) {
    std::vector<PartialTensorShape> output_shapes;
    std::transform(iter.second.begin(), iter.second.end(),
                   std::back_inserter(output_shapes),
                   [](const InferredShape& inferred_shape) {
                     return inferred_shape.shape;
                   });
    Node* node = node_name_index[iter.first];
    node->ClearAttr(kNeuronInferredShapes);
    node->AddAttr(kNeuronInferredShapes, output_shapes);
  }
  return Status::OK();
}

}  // end namespace

Status ShapeInference(GraphDef* new_graph_def, const GraphDef& graph_def) {
  std::unique_ptr<Graph> graph =
      std::unique_ptr<Graph>(new Graph(OpRegistry::Global()));
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, graph_def, graph.get()));
  TF_RETURN_IF_ERROR(
      PerformStaticShapeInferenceBeforeEncapsulation(graph.get()));
  graph->ToGraphDef(new_graph_def);
  return Status::OK();
}

}  // end namespace convert
}  // end namespace neuron
}  // end namespace tensorflow
