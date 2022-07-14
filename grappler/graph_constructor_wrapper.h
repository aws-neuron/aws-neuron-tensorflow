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

#ifndef TENSORFLOW_NEURON_COMPILER_GRAPH_CONSTRUCTOR_H_
#define TENSORFLOW_NEURON_COMPILER_GRAPH_CONSTRUCTOR_H_

#include "tensorflow/neuron/runtime/direct/macros.h"

#if TF_VERSION_LESS_THAN(2, 3)
#include "tensorflow/core/graph/graph_constructor.h"
#else
#include "tensorflow/core/common_runtime/graph_constructor.h"
#endif

#endif  // TENSORFLOW_NEURON_COMPILER_GRAPH_CONSTRUCTOR_H_
