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

#include "env.h"
#include <stdexcept>
#include <sstream>
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace neuron {

#define STOI_INVALID_RESULT -65536

std::string env_get(const char* env_var, const char* default_env_var) {
  char* str = std::getenv(env_var);
  return str ? str : default_env_var;
}

int stoi_no_throw(const std::string& str) {
  try {
    return std::stoi(str);
  } catch (std::invalid_argument&) {
    return STOI_INVALID_RESULT;
  } catch (std::out_of_range&) {
    return STOI_INVALID_RESULT;
  }
}

static std::string remove_pattern(std::string data, const std::string& pat) {
  size_t string_length = data.size();
  size_t pos = 0;
  for (size_t idx = 0; idx < string_length; ++idx) {
    pos = data.find(pat, pos);
    if (std::string::npos == pos) {
      break;
    }
    data.replace(pos, pat.size(), "");
  }
  return data;
}

std::vector<std::pair<int, int>> parse_engine_specs() {
  std::string engine_sizes_raw = env_get("NEURONCORE_GROUP_SIZES", "");
  // remove [ and ]
  engine_sizes_raw = remove_pattern(engine_sizes_raw, "[");
  engine_sizes_raw = remove_pattern(engine_sizes_raw, "]");

  // device spec in <num_cores, num_duplications> format
  std::vector<std::pair<int, int>> engine_specs;
  std::stringstream engine_sizes_stream(engine_sizes_raw);
  for (size_t idx = 0; idx < engine_sizes_raw.size(); ++idx) {
    if (!engine_sizes_stream.good()) {
      break;
    }
    std::string spec_raw;
    std::getline(engine_sizes_stream, spec_raw, ',');
    if (spec_raw.empty()) {
      continue;
    }
    int num_dup = 1;
    if (spec_raw.find("x") != std::string::npos) {
      size_t delim_pos = spec_raw.find("x");
      num_dup = stoi_no_throw(spec_raw.substr(0, delim_pos));
      spec_raw = spec_raw.substr(delim_pos + 1, std::string::npos);
    }
    int num_cores = stoi_no_throw(spec_raw);
    engine_specs.push_back(std::make_pair(num_cores, num_dup));
  }
  return engine_specs;
}

std::string mangle_op_name(StringPiece op_name) {
  std::string new_op_name(op_name);
  for (size_t idx = 0; idx < new_op_name.length(); ++idx) {
    if ('/' == new_op_name[idx]) {
      new_op_name[idx] = '+';
    }
  }
  return new_op_name;
}

}  // namespace neuron
}  // namespace tensorflow
