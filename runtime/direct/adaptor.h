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

#ifndef TENSORFLOW_NEURON_RUNTIME_DIRECT_ADAPTOR_H_
#define TENSORFLOW_NEURON_RUNTIME_DIRECT_ADAPTOR_H_

#include <cstddef>
#include <cstdint>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace neuron {

class NrtModel {
 private:
  void* raw_ = nullptr;
  friend class Nrt;
};

class NrtBuffer {
 private:
  void* raw_ = nullptr;
  friend class Nrt;
};

class NrtBufferMap {
 private:
  void* raw_ = nullptr;
  friend class Nrt;
};

class Nrt {
 public:
  static Status Init();
  static Status Close();
  static Status AllocHostBuffer(NrtBuffer* buffer, size_t size);
  static Status FreeBuffer(NrtBuffer* buffer);
  static Status CopyCpuToBuffer(NrtBuffer* buffer, size_t offset,
                                const void* cpu_buffer, size_t size);
  static Status CopyBufferToCpu(void* cpu_buffer, size_t size,
                                const NrtBuffer& buffer, size_t offset);
  static Status Load(NrtModel* model, StringPiece executable, int32_t start_nc,
                     int32_t nc_count);
  static Status Unload(const NrtModel& model);
  static Status AllocBufferMap(NrtBufferMap* map);
  static Status FreeBufferMap(const NrtBufferMap& map);
  static Status BufferMapAdd(NrtBufferMap* map, const std::string& name,
                             const NrtBuffer& buffer);
  static Status BufferMapGet(NrtBuffer* buffer, const NrtBufferMap& map,
                             const std::string& name);
  static Status Execute(const NrtModel& model, const NrtBufferMap& input_map,
                        NrtBufferMap* output_map);
  // profiler functions
  static Status ProfileStart(const NrtModel& model, const char* filename);
  static Status ProfileStop(const char* filename);

 private:
  Nrt();
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DIRECT_ADAPTOR_H_
