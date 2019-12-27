/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_NEURON_CLIB_TIMESTAMPS_H_
#define TENSORFLOW_NEURON_NEURON_CLIB_TIMESTAMPS_H_

#include "tensorflow/core/platform/env.h"


namespace tensorflow {
namespace neuron {


class Timestamps {
public:
    void mark_enter() { enter_ = now(); }
    void mark_above_nrtd_infer() { above_nrtd_infer_ = now(); }
    void mark_below_nrtd_infer() { below_nrtd_infer_ = now(); }
    void mark_exit() { exit_ = now(); }
    std::string timing_string() {
        std::string result("NeuronOp enter timestamp: ");
        result += uint64_to_string(enter_);
        result += time_unit_;
        result += ", preprocessing time ";
        result += uint64_to_string(above_nrtd_infer_ - enter_);
        result += time_unit_;
        result += ", neuron-rtd infer time ";
        result += uint64_to_string(below_nrtd_infer_ - above_nrtd_infer_);
        result += time_unit_;
        result += ", postprocessing time ";
        result += uint64_to_string(exit_ - below_nrtd_infer_);
        result += time_unit_;
        return result;
    }
private:
    uint64 enter_ = 0;
    uint64 above_nrtd_infer_ = 0;
    uint64 below_nrtd_infer_ = 0;
    uint64 exit_ = 0;

    std::string time_unit_ = " us";
    uint64 now() { return Env::Default()->NowMicros(); }
    std::string uint64_to_string(uint64 number) {
        std::ostringstream oss;
        oss << number;
        return oss.str();
    }
};


}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_NEURON_CLIB_TIMESTAMPS_H_
