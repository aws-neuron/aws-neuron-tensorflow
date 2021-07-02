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

#include <cstddef>
#include <cstdint>

extern "C" {

// These definitions are from various places includine c/c_api.h
// and c/experimental/grappler/grappler.h. They are here because
// bazel target "//tensorflow/c/experimental/grappler:grappler_hdr"
// cannot be used as-is, as it pulls in almost a full copy of
// the tensorflow core runtime library.
typedef struct TF_Status TF_Status;

typedef enum TF_TriState {
  TF_TriState_Default = 0,
  TF_TriState_Off,
  TF_TriState_On,
} TF_TriState;

typedef struct TP_OptimizerConfigs {
  size_t struct_size;
  void* ext;  // reserved for future use
  TF_TriState disable_model_pruning;
  TF_TriState implementation_selector;
  TF_TriState function_optimization;
  TF_TriState common_subgraph_elimination;
  TF_TriState arithmetic_optimization;
  TF_TriState debug_stripper;
  TF_TriState constant_folding;
  TF_TriState shape_optimization;
  TF_TriState auto_mixed_precision;
  TF_TriState auto_mixed_precision_mkl;
  TF_TriState pin_to_host_optimization;
  TF_TriState layout_optimizer;
  TF_TriState remapping;
  TF_TriState loop_optimization;
  TF_TriState dependency_optimization;
  TF_TriState auto_parallel;
  TF_TriState memory_optimization;
  TF_TriState scoped_allocator_optimization;
} TP_OptimizerConfigs;

typedef struct TF_Buffer {
  const void* data;
  size_t length;
  void (*data_deallocator)(void* data, size_t length);
} TF_Buffer;

void AwsNeuronDoNothing(void* optimizer, TF_Buffer* input_graph,
                        TF_Buffer* output_graph, TF_Status* status) {
}

typedef struct TP_Optimizer {
  size_t struct_size;
  void* ext;  // reserved for future use

  // [Optional]
  // Create function for optimizer.
  void* (*create_func)();

  // Optimizer function for optimizer. The first param is an optimizer created
  // by create_func. The second param is input graph. The third param is output
  // graph.
  void (*optimize_func)(void*, TF_Buffer*, TF_Buffer*, TF_Status*);

  // [Optional]
  // Destroy function for optimizer. If Create function is provided, destroy
  // function is must.
  void (*destroy_func)(void*);
} TP_Optimizer;

typedef struct TP_OptimizerRegistrationParams {
  size_t struct_size;
  void* ext;  // reserved for future use

  // Graph C API version.
  int32_t major_version;
  int32_t minor_version;
  int32_t patch_version;

  // Backend device type supported by the optimizer.
  const char* device_type;
  TP_OptimizerConfigs* optimizer_configs;  // output, set by plugin
  TP_Optimizer* optimizer;                 // output, set by plugin
} TP_OptimizerRegistrationParams;

#define TF_OFFSET_OF_END(TYPE, MEMBER) \
  (offsetof(TYPE, MEMBER) + sizeof(((TYPE *)0)->MEMBER))
#define TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(TP_OptimizerRegistrationParams, optimizer)
#define TP_OPTIMIZER_CONFIGS_STRUCT_SIZE \
  TF_OFFSET_OF_END(TP_OptimizerConfigs, scoped_allocator_optimization)
#define TP_OPTIMIZER_STRUCT_SIZE TF_OFFSET_OF_END(TP_Optimizer, destroy_func)

void TF_InitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status) {
  // TODO: fix once we figure out why TF_InitGraph is always called twice
  static bool is_registered = false;
  if (is_registered) {
    params->device_type = "DEVICE_AWS_NEURON_NULL";
  } else {
    params->device_type = "DEVICE_AWS_NEURON_DUMMY";
  }
  params->struct_size = TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->optimizer_configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
  params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;
  params->optimizer->create_func = nullptr;
  params->optimizer->optimize_func = AwsNeuronDoNothing;
  params->optimizer->destroy_func = nullptr;
  is_registered = true;
}

void TF_InitKernel() {}

}
