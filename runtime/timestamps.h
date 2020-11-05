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

#ifndef TENSORFLOW_NEURON_RUNTIME_TIMESTAMPS_H_
#define TENSORFLOW_NEURON_RUNTIME_TIMESTAMPS_H_

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

#endif  // TENSORFLOW_NEURON_RUNTIME_TIMESTAMPS_H_
