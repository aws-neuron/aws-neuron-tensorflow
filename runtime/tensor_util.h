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

#ifndef TENSORFLOW_NEURON_RUNTIME_TENSOR_UTIL_H_
#define TENSORFLOW_NEURON_RUNTIME_TENSOR_UTIL_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace neuron {

using namespace tensorflow::thread;

void fast_memcpy(void* dst, const void* src, int64 count, ThreadPool* pool);
Status tensor_memcpy(Tensor* dst, const StringPiece& src, ThreadPool* pool);
Status tensor_memset(Tensor* dst, int ch);
Status tensor_copy(Tensor* dst, const Tensor& src, ThreadPool* pool = nullptr);
Status tensor_shuffle(Tensor* dst, const Tensor& src, const TensorProto& shf);

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_TENSOR_UTIL_H_
