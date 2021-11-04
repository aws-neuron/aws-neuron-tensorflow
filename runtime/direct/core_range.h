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

#ifndef TENSORFLOW_NEURON_RUNTIME_DIRECT_CORE_RANGE_H_
#define TENSORFLOW_NEURON_RUNTIME_DIRECT_CORE_RANGE_H_

#include <cstdint>

namespace tensorflow {
namespace neuron {

struct NeuronCoreRange {
  NeuronCoreRange(int32_t start_nc, int32_t nc_count)
      : start_nc_(start_nc), nc_count_(nc_count) {}
  const int32_t start_nc_;
  const int32_t nc_count_;
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DIRECT_CORE_RANGE_H_
