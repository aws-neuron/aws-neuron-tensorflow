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

#include "tensor_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {
namespace neuron {

#define IS_4BYTE_ALIGNED(ptr) (((uintptr_t)(const void*)(ptr)) % 4u == 0)
#define IS_8BYTE_ALIGNED(ptr) (((uintptr_t)(const void*)(ptr)) % 8u == 0)
#define RETURN_ERROR_IF_CANNOT_MEMCPY(dtype, name)    \
  if (TF_PREDICT_FALSE(!DataTypeCanUseMemcpy(dtype))) \
    return errors::Unimplemented(name, " on data type ", dtype);

static void* memcpy_uint64(void* dst, const void* src, size_t size) {
  uint64_t* ss = (uint64_t*)src;
  uint64_t* dd = (uint64_t*)dst;
  size = size * sizeof(uint8_t) / sizeof(uint64_t);
  while (size--) *dd++ = *ss++;
  return dst;
}

static void* memcpy_uint32(void* dst, const void* src, size_t size) {
  uint32_t* ss = (uint32_t*)src;
  uint32_t* dd = (uint32_t*)dst;
  size = size * sizeof(uint8_t) / sizeof(uint32_t);
  while (size--) *dd++ = *ss++;
  return dst;
}

typedef std::function<void*(void*, const void*, size_t)> MemcpyFunc;

void fast_memcpy(void* dst, const void* src, int64 count, ThreadPool* pool) {
  char* char_dst = static_cast<char*>(dst);
  const char* char_src = static_cast<const char*>(src);
  int64 total_size = count;
  MemcpyFunc memcpy_func = std::memcpy;
  if (total_size < 1024) {
    std::copy_n(char_src, total_size, char_dst);
  } else if (total_size <= 1024 * 1024 * 4 || nullptr == pool) {
    int64 copy_size = total_size;
    if (IS_8BYTE_ALIGNED(char_src) && IS_8BYTE_ALIGNED(char_dst)) {
      copy_size = total_size / 8 * 8;
      memcpy_func = memcpy_uint64;
    } else if (IS_4BYTE_ALIGNED(char_src) && IS_4BYTE_ALIGNED(char_dst)) {
      copy_size = total_size / 4 * 4;
      memcpy_func = memcpy_uint32;
    }
    memcpy_func(char_dst, char_src, copy_size);
    if (copy_size != total_size) {
      std::memcpy(char_dst + copy_size, char_src + copy_size,
                  total_size - copy_size);
    }
  } else {
    int64 alignment = 1;
    if (IS_8BYTE_ALIGNED(char_src) && IS_8BYTE_ALIGNED(char_dst)) {
      alignment = 8;
      memcpy_func = memcpy_uint64;
    } else if (IS_4BYTE_ALIGNED(char_src) && IS_4BYTE_ALIGNED(char_dst)) {
      alignment = 4;
      memcpy_func = memcpy_uint32;
    }
    int64 num_parallel = 8;
    int64 slice_size = total_size / num_parallel;
    slice_size -= slice_size % alignment;
    int64 last_slice_size = total_size - slice_size * (num_parallel - 1);

    std::vector<int64> vec_slice_size(num_parallel, slice_size);
    vec_slice_size[num_parallel - 1] = last_slice_size;
    std::vector<MemcpyFunc> vec_memcpy_func(num_parallel, memcpy_func);
    if (last_slice_size % alignment) {
      vec_memcpy_func[num_parallel - 1] = std::memcpy;
    }

    auto memcpy_shard = [&char_dst, &char_src, &slice_size, &vec_memcpy_func,
                         &vec_slice_size](int64 begin, int64 end) {
      for (int64 idx = begin; idx < end; ++idx) {
        int64 offset = idx * slice_size;
        vec_memcpy_func[idx](char_dst + offset, char_src + offset,
                             vec_slice_size[idx]);
      }
    };
    pool->ParallelFor(num_parallel, slice_size, std::move(memcpy_shard));
  }
}

Status tensor_copy(Tensor* dst, const Tensor& src, ThreadPool* pool) {
  RETURN_ERROR_IF_CANNOT_MEMCPY(src.dtype(), "tensor_copy");
  return tensor_memcpy(dst, src.tensor_data(), pool);
}

Status tensor_memcpy(Tensor* dst, const StringPiece& src, ThreadPool* pool) {
  RETURN_ERROR_IF_CANNOT_MEMCPY(dst->dtype(), "tensor_memcpy");
  int64 src_size = src.size();
  int64 dst_size = dst->tensor_data().size();
  int64 memcpy_size = src_size < dst_size ? src_size : dst_size;
  const char* char_src = src.data();
  char* char_dst = const_cast<char*>(dst->tensor_data().data());
  fast_memcpy(char_dst, char_src, memcpy_size, pool);
  return Status::OK();
}

Status tensor_memset(Tensor* tensor, int ch) {
  std::fill_n(const_cast<char*>(tensor->tensor_data().data()),
              tensor->tensor_data().size(), ch);
  return Status::OK();
}

template <typename T>
static Status tensor_shuffle_impl(Tensor* dst, const Tensor& src,
                                  const TensorProto& shf) {
  const T* src_ptr = reinterpret_cast<const T*>(src.tensor_data().data());
  T* dst_ptr =
      reinterpret_cast<T*>(const_cast<char*>((dst->tensor_data().data())));
  for (auto idx = 0; idx < src.NumElements(); ++idx) {
    dst_ptr[idx] = src_ptr[shf.int64_val(idx)];
  }
  return Status::OK();
}

Status tensor_shuffle(Tensor* dst, const Tensor& src, const TensorProto& shf) {
  if (TF_PREDICT_FALSE(dst->tensor_data().size() < src.tensor_data().size())) {
    return errors::InvalidArgument("tensor_shuffle dst size ",
                                   dst->tensor_data().size(),
                                   ", src size ", src.tensor_data().size());
  }
  if (TF_PREDICT_FALSE(shf.int64_val_size() < src.NumElements())) {
    return errors::InvalidArgument("tensor_shuffle shuffle int64_val_size ",
                                   shf.int64_val_size(),
                                   ", src NumElements ", src.NumElements());
  }
  switch (src.dtype()) {
#define CASE(type)                                   \
  case DataTypeToEnum<type>::value: {                \
    return tensor_shuffle_impl<type>(dst, src, shf); \
    break;                                           \
  }
    TF_CALL_REAL_NUMBER_TYPES(CASE);
    TF_CALL_bool(CASE);
#undef CASE
    default:
      return errors::InvalidArgument("invalid data type ", src.dtype());
  }
  return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow
