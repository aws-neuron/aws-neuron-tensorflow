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

#ifndef TENSORFLOW_NEURON_RUNTIME_ENV_H_
#define TENSORFLOW_NEURON_RUNTIME_ENV_H_

#include <string>
#include <vector>
#include <utility>
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace neuron {

std::string env_get(const char* env_var, const char* default_env_var = "");
int stoi_no_throw(const std::string& str);
std::vector<std::pair<int, int>> parse_engine_specs();
std::string mangle_op_name(StringPiece op_name);

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_ENV_H_
