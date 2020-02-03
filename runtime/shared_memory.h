/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_NEURON_CLIB_SHARED_MEMORY_H_
#define TENSORFLOW_NEURON_NEURON_CLIB_SHARED_MEMORY_H_

namespace tensorflow {
namespace neuron {

typedef struct SharedMemory {
    std::vector<std::string> input_paths_;
    std::vector<void*> input_ptrs_;
    std::vector<size_t> input_sizes_;
    std::vector<std::string> output_paths_;
    std::vector<void*> output_ptrs_;
    std::vector<size_t> output_sizes_;
    std::vector<std::string> nrt_input_paths_;
    std::vector<std::string> nrt_output_paths_;
} SharedMemory;

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_NEURON_CLIB_SHARED_MEMORY_H_
