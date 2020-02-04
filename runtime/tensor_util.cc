/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#include "tensor_util.h"

namespace tensorflow {
namespace neuron {


Status tensor_memcpy(Tensor *tensor, StringPiece &source, int64 memcpy_size) {
    int64 dst_size = tensor->tensor_data().size();
    if (memcpy_size < 0) {
        memcpy_size = dst_size;
    }
    if (memcpy_size > (int64)source.size() || memcpy_size > dst_size) {
        return errors::OutOfRange(
            "unexpected tensor size in tensor_memcpy, source size: ",
            source.size(), ", target size: ", tensor->tensor_data().size());
    }
    std::copy_n(source.data(), memcpy_size,
                const_cast<char*>(tensor->tensor_data().data()));
    return Status::OK();
}

Status tensor_memset(Tensor *tensor, int ch) {
    std::fill_n(const_cast<char*>(tensor->tensor_data().data()),
                tensor->tensor_data().size(), ch);
    return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow
