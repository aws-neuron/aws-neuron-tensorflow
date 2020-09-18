/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_NEURON_CLIB_SHARED_MEMORY_H_
#define TENSORFLOW_NEURON_NEURON_CLIB_SHARED_MEMORY_H_

#include "tensorflow/core/platform/mutex.h"

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

#endif  // TENSORFLOW_NEURON_NEURON_CLIB_SHARED_MEMORY_H_
