/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_NEURON_CLIB_TENSOR_UTIL_H_
#define TENSORFLOW_NEURON_NEURON_CLIB_TENSOR_UTIL_H_

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace neuron {

typedef const AttrValue_ListValue AttrList;

Status tensor_memcpy(Tensor *tensor, StringPiece &source, int64 memcpy_size=-1);
Status tensor_memset(Tensor *tensor, int ch);

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_NEURON_CLIB_TENSOR_UTIL_H_
