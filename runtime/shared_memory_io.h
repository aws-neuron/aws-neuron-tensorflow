/* Copyright 2020 AWS Neuron. All Rights Reserved.

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

#ifndef TENSORFLOW_NEURON_RUNTIME_SHARED_MEMORY_IO_H_
#define TENSORFLOW_NEURON_RUNTIME_SHARED_MEMORY_IO_H_

namespace tensorflow {
namespace neuron {

typedef struct SharedMemory {
    std::vector<std::string*> input_paths_;
    std::vector<char*> input_ptrs_;
    std::vector<std::string*> output_paths_;
    std::vector<char*> output_ptrs_;
} SharedMemory;

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_SHARED_MEMORY_IO_H_
