/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#include "tensorflow/neuron/runtime/tensor_util.h"

namespace tensorflow {
namespace neuron {

#define IS_4BYTE_ALIGNED(ptr) (((uintptr_t)(const void *)(ptr)) % 4u == 0)
#define IS_8BYTE_ALIGNED(ptr) (((uintptr_t)(const void *)(ptr)) % 8u == 0)

static void *memcpy_uint64(void *dst, const void *src, size_t size) {
    uint64_t *ss = (uint64_t*)src;
    uint64_t *dd = (uint64_t*)dst;
    size = size * sizeof(uint8_t) / sizeof(uint64_t);
    while (size--)
        *dd++ = *ss++;
    return dst;
}

static void *memcpy_uint32(void *dst, const void *src, size_t size) {
    uint32_t *ss = (uint32_t*)src;
    uint32_t *dd = (uint32_t*)dst;
    size = size * sizeof(uint8_t) / sizeof(uint32_t);
    while (size--)
        *dd++ = *ss++;
    return dst;
}

typedef std::function<void*(void*, const void*, size_t)> MemcpyFunc;

void fast_memcpy(thread::ThreadPool *thread_pool, char *char_dst, const char *char_src, int64 total_size) {
    MemcpyFunc memcpy_func = std::memcpy;
    if (total_size < 1024) {
        std::copy_n(char_src, total_size, char_dst);
    } else if (total_size <= 1024 * 1024 * 4 || nullptr == thread_pool) {
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
            std::memcpy(char_dst + copy_size, char_src + copy_size, total_size - copy_size);
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
                vec_memcpy_func[idx](char_dst + offset, char_src + offset, vec_slice_size[idx]);
            }
        };
        thread_pool->ParallelFor(num_parallel, slice_size, std::move(memcpy_shard));
    }
}

Status tensor_memcpy(thread::ThreadPool *thread_pool, Tensor *tensor, StringPiece &source, int64 memcpy_size) {
    if (!DataTypeCanUseMemcpy(tensor->dtype())) {
        return errors::Unimplemented("tensor_memcpy on data type ", tensor->dtype(), " is not allowed");
    }
    int64 dst_size = tensor->tensor_data().size();
    if (memcpy_size < 0) {
        memcpy_size = dst_size;
    }
    if (memcpy_size > (int64)source.size() || memcpy_size > dst_size) {
        return errors::OutOfRange(
            "unexpected tensor size in tensor_memcpy, source size: ",
            source.size(), ", target size: ", tensor->tensor_data().size());
    }
    const char *char_src = source.data();
    char *char_dst = const_cast<char*>(tensor->tensor_data().data());
    fast_memcpy(thread_pool, char_dst, char_src, memcpy_size);
    return Status::OK();
}

Status tensor_memset(Tensor *tensor, int ch) {
    std::fill_n(const_cast<char*>(tensor->tensor_data().data()),
                tensor->tensor_data().size(), ch);
    return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow
