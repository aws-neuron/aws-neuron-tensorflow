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

#ifndef TENSORFLOW_NEURON_RUNTIME_DIRECT_NODE_DEF_KEYS_H_
#define TENSORFLOW_NEURON_RUNTIME_DIRECT_NODE_DEF_KEYS_H_

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
constexpr char kAutoMulticore[] = "_automatic_multicore";
constexpr char kInputShuffles[] = "_input_shuffles";
constexpr char kRealInputNames[] = "_real_input_names";
constexpr char kRealInputLocations[] = "_real_input_locations";
constexpr char kInstanceType[] = "_instance_type";

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DIRECT_NODE_DEF_KEYS_H_
