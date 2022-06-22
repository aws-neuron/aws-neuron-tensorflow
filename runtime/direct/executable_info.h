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

#ifndef TENSORFLOW_NEURON_RUNTIME_DIRECT_EXECUTABLE_INFO_H_
#define TENSORFLOW_NEURON_RUNTIME_DIRECT_EXECUTABLE_INFO_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace neuron {

class NeuronExecutableInfo {
 public:
  Status ParseFromNodeDef(const NodeDef& node_def);
  StringPiece name;
  StringPiece executable;
  StringPiece serialized_graph_def;
  int32_t optimal_num_cores;
  int32_t max_num_duplicates;
  AttrValue_ListValue input_names;
  AttrValue_ListValue input_dtypes;
  AttrValue_ListValue input_shapes;
  AttrValue_ListValue input_batch_axis;
  AttrValue_ListValue output_names;
  AttrValue_ListValue output_dtypes;
  AttrValue_ListValue output_shapes;
  AttrValue_ListValue output_batch_axis;

  // Optional values
  const AttrValue_ListValue* input_shuffles = nullptr;
  bool auto_multicore_enabled = false;
  int32_t requested_num_cores = -1;
  const AttrValue_ListValue* real_input_names = nullptr;
  const AttrValue_ListValue* real_input_locations = nullptr;

 private:
  Status ParseModelConfig(const NodeDef& node_def);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DIRECT_EXECUTABLE_INFO_H_
