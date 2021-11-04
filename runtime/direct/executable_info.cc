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

#include "executable_info.h"
#include <google/protobuf/map.h>
#include <cstdint>
#include <string>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/default/logging.h"

namespace tensorflow {
namespace neuron {

// Required keys
constexpr char kExecutable[] = "executable";
constexpr char kGraphDef[] = "graph_def";
constexpr char kModelConfig[] = "model_config";
constexpr char kInputNames[] = "input_names";
constexpr char kInputDtypes[] = "input_dtypes";
constexpr char kInputShapes[] = "input_shapes";
constexpr char kInputBatchAxis[] = "input_batch_axis";
constexpr char kOutputNames[] = "output_names";
constexpr char kOutputDtypes[] = "output_dtypes";
constexpr char kOutputShapes[] = "output_shapes";
constexpr char kOutputBatchAxis[] = "output_batch_axis";

// Optional keys
constexpr char kInputShuffles[] = "_input_shuffles";

Status NeuronExecutableInfo::ParseFromNodeDef(const NodeDef& node_def) {
  name = node_def.name();
  const google::protobuf::Map<std::string, AttrValue>& attr = node_def.attr();
#define NODE_DEF_CHECK_KEY(key)                                 \
  if (TF_PREDICT_FALSE(!attr.count(key))) {                     \
    return errors::InvalidArgument("Key \"", key,               \
                                   "\" not found in attributes" \
                                   " of NodeDef \"",            \
                                   node_def.name(), "\".");     \
  }
  NODE_DEF_CHECK_KEY(kExecutable);
  NODE_DEF_CHECK_KEY(kGraphDef);
  NODE_DEF_CHECK_KEY(kModelConfig);
  NODE_DEF_CHECK_KEY(kInputNames);
  NODE_DEF_CHECK_KEY(kInputDtypes);
  NODE_DEF_CHECK_KEY(kInputShapes);
  NODE_DEF_CHECK_KEY(kInputBatchAxis);
  NODE_DEF_CHECK_KEY(kOutputNames);
  NODE_DEF_CHECK_KEY(kOutputDtypes);
  NODE_DEF_CHECK_KEY(kOutputShapes);
  NODE_DEF_CHECK_KEY(kOutputBatchAxis);
#undef NODE_DEF_CHECK_KEY
  executable = attr.at(kExecutable).s();
  serialized_graph_def = attr.at(kGraphDef).s();
  input_names = attr.at(kInputNames).list();
  input_dtypes = attr.at(kInputDtypes).list();
  input_shapes = attr.at(kInputShapes).list();
  input_batch_axis = attr.at(kInputBatchAxis).list();
  output_names = attr.at(kOutputNames).list();
  output_dtypes = attr.at(kOutputDtypes).list();
  output_shapes = attr.at(kOutputShapes).list();
  output_batch_axis = attr.at(kOutputBatchAxis).list();
#define SIZE_CHECK(cond, attr_name)                                       \
  if (TF_PREDICT_FALSE(!(cond))) {                                        \
    return errors::InvalidArgument("Invalid size found in attributes \"", \
                                   (attr_name), " of NodeDef \"",         \
                                   node_def.name(), "\".");               \
  }
  int num_inputs = input_dtypes.type_size();
  SIZE_CHECK(num_inputs == input_names.s_size(), kInputNames);
  SIZE_CHECK(num_inputs == input_shapes.shape_size(), kInputShapes);
  SIZE_CHECK(num_inputs == input_batch_axis.i_size(), kInputBatchAxis);
  int num_outputs = output_dtypes.type_size();
  SIZE_CHECK(num_outputs == output_names.s_size(), kOutputNames);
  SIZE_CHECK(num_outputs == output_shapes.shape_size(), kOutputShapes);
  SIZE_CHECK(num_outputs == output_batch_axis.i_size(), kOutputBatchAxis);
  if (attr.count(kInputShuffles)) {
    input_shuffles = &attr.at(kInputShuffles).list();
    SIZE_CHECK(num_inputs == input_shuffles->tensor_size(), kInputShuffles);
  }
#undef SIZE_CHECK
  return ParseModelConfig(node_def);
}

enum ModelConfigKey {
  kGlobalOptNumCores = 0,  // deprecated
  kOptNumCores = 1,
  kMaxNumDuplicates = 2,
  kModelConfigKeyBound,
};

Status NeuronExecutableInfo::ParseModelConfig(const NodeDef& node_def) {
  const google::protobuf::Map<std::string, AttrValue>& attr = node_def.attr();
  const AttrValue_ListValue& model_config_list = attr.at(kModelConfig).list();
#define MODEL_CONFIG_CHECK(cond)                                               \
  if (TF_PREDICT_FALSE(!(cond))) {                                             \
    return errors::InvalidArgument("Invalid model_config found on NodeDef \"", \
                                   node_def.name(), "\": `", (#cond),          \
                                   "` is not true");                           \
  }
  MODEL_CONFIG_CHECK(model_config_list.i_size() >= kModelConfigKeyBound);
  constexpr int32_t MAX_NUM_CORES = 1024;
  optimal_num_cores = (int32_t)model_config_list.i(kOptNumCores);
  VLOG(1) << "optimal_num_cores=" << optimal_num_cores;
  MODEL_CONFIG_CHECK(0 <= optimal_num_cores);
  MODEL_CONFIG_CHECK(optimal_num_cores < MAX_NUM_CORES);
  max_num_duplicates = (int32_t)model_config_list.i(kMaxNumDuplicates);
  VLOG(1) << "max_num_duplicates=" << max_num_duplicates;
  MODEL_CONFIG_CHECK(0 < max_num_duplicates);
  MODEL_CONFIG_CHECK(max_num_duplicates < MAX_NUM_CORES);
#undef MODEL_CONFIG_CHECK
  return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow
