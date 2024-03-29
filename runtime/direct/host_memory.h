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

#ifndef TENSORFLOW_NEURON_RUNTIME_DIRECT_HOST_MEMORY_H_
#define TENSORFLOW_NEURON_RUNTIME_DIRECT_HOST_MEMORY_H_

#include <cstddef>
#include <memory>
#include <string>

#include "adaptor.h"
#include "executable_info.h"
#include "macros.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace neuron {

class NeuronHostBuffer {
 public:
  NeuronHostBuffer(size_t size);
  NeuronHostBuffer(void* cpu_buffer, size_t size);
  ~NeuronHostBuffer();
  Status GetStatus();
  bool Owned() { return payload_ != 0; }
  size_t GetSize() { return size_; }
  Status CopyCpuToBuffer(const void* cpu_buffer, size_t size,
                         size_t offset = 0);
  Status CopyBufferToCpu(void* cpu_buffer, size_t size, size_t offset = 0);

 private:
  friend class NeuronHostBufferMap;
  NrtBuffer rt_buffer_;
  size_t size_ = 0;
  size_t payload_ = 0;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronHostBuffer);
};

class NeuronHostBufferMap {
 public:
  NeuronHostBufferMap();
  ~NeuronHostBufferMap();
  Status GetStatus() { return status_; }
  Status AddBuffer(const std::string& name, const NeuronHostBuffer& buffer);

 private:
  friend class NeuronExecutable;
  friend class NeuronExecutableProfiler;
  NrtBufferMap rt_buffer_map_;
  Status status_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronHostBufferMap);
};

class NeuronHostMemory {
 public:
  NeuronHostMemory() {}
  Status SetupBuffers(const NeuronExecutableInfo& info,
                      std::vector<Tensor>* input_tensors,
                      std::vector<Tensor>* output_tensors);
  Status SetupBuffersCacheWeights(const NeuronExecutableInfo& info,
                                  std::vector<Tensor>* input_tensors,
                                  std::vector<Tensor>* output_tensors);
  Status CopyCPUToInputBuffers(const std::vector<Tensor>& input_tensors);
  Status CopyOutputBuffersToCPU(const std::vector<Tensor>& output_tensors);

 private:
  friend class NeuronExecutable;
  friend class NeuronExecutableProfiler;
  std::vector<std::shared_ptr<NeuronHostBuffer>> input_buffers_;
  std::vector<std::shared_ptr<NeuronHostBuffer>> output_buffers_;
  NeuronHostBufferMap input_buffer_map_;
  NeuronHostBufferMap output_buffer_map_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronHostMemory);
};

class NeuronDeviceBuffer {
 public:
  NeuronDeviceBuffer(size_t size, int32_t memory_id);
  ~NeuronDeviceBuffer();
  Status GetStatus();
  Status CopyCpuToBuffer(const void* cpu_buffer, size_t size,
                         size_t offset = 0);
  Status CopyBufferToCpu(void* cpu_buffer, size_t size, size_t offset = 0);
  size_t GetSize() { return size_; }
  bool Owned() { return payload_ != 0; }

 private:
  friend class NeuronDeviceBufferMap;
  NrtBuffer rt_buffer_;
  size_t size_ = 0;
  size_t payload_ = 0;
  int32_t memory_id_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronDeviceBuffer);
};

class NeuronDeviceBufferMap {
 public:
  NeuronDeviceBufferMap();
  ~NeuronDeviceBufferMap();
  Status GetStatus() { return status_; }
  Status AddBuffer(const std::string& name, const NeuronDeviceBuffer& buffer);

 private:
  friend class NeuronExecutable;
  NrtBufferMap rt_buffer_map_;
  Status status_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronDeviceBufferMap);
};

class NeuronDeviceMemory {
 public:
  NeuronDeviceMemory(int32_t memory_id) { memory_id_ = memory_id; }
  Status SetupBuffers(const NeuronExecutableInfo& info,
                      std::vector<Tensor>* all_tensors,
                      std::vector<Tensor>* output_tensors,
                      std::vector<std::shared_ptr<NeuronDeviceBuffer>>& cache,
                      std::vector<int> ril);
  Status CopyOutputBuffersToCPU(const std::vector<Tensor>& output_tensors);

 private:
  friend class NeuronExecutable;
  std::vector<std::shared_ptr<NeuronDeviceBuffer>> input_buffers_;
  std::vector<std::shared_ptr<NeuronDeviceBuffer>> output_buffers_;
  NeuronDeviceBufferMap input_buffer_map_;
  NeuronDeviceBufferMap output_buffer_map_;
  int32_t memory_id_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronDeviceMemory);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DIRECT_HOST_MEMORY_H_
