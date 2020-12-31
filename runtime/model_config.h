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

#ifndef TENSORFLOW_NEURON_RUNTIME_KERNELS_MODEL_CONFIG_H_
#define TENSORFLOW_NEURON_RUNTIME_KERNELS_MODEL_CONFIG_H_

#include "../macros.h"
#include "../env.h"

namespace tensorflow {
namespace neuron {

class NeuronModelConfig {
public:
    NeuronModelConfig() {}
    void parse_opt_device_size(AttrList &model_config) {
        if (model_config_valid(model_config)) {
            opt_device_size_ = model_config_global_opt_num_cores(model_config);
            max_num_duplicates_ = model_config_max_num_duplicates(model_config);
        }
    }

    void parse_timeout(AttrList &model_config) {
        if (model_config_valid(model_config)) {
            int64 int64_timeout = model_config_timeout(model_config);
            if (int64_timeout > 0) {
                timeout_ = (uint32_t)int64_timeout;
                return;
            }
        }
        timeout_ = 10;
        std::string infer_timeout_str = env_get("NEURON_FRAMEWORK_INFER_TIMEOUT_SEC", "10");
        int int_timeout = stoi_no_throw(infer_timeout_str);
        if (int_timeout <= 0) {
            LOG(WARNING) << "NEURON_FRAMEWORK_INFER_TIMEOUT_SEC=" << infer_timeout_str
                         << " is invalid; using default value " << timeout_ << " seconds.";
        } else {
            timeout_ = (uint32_t)int_timeout;
        }
    }

    void parse_ninfer(AttrList &model_config, const uint32_t num_cores,
                      const uint32_t min_num_cores, const uint32_t max_num_cores) {
        int64 max_num_threads = DEFAULT_MAX_NUM_INFER;
        if (model_config_valid(model_config)) {
            int64 opt_num_infer = model_config_this_opt_num_cores(model_config);
            if (opt_num_infer > 0 && opt_num_infer <= HARD_MAX_NUM_THREADS) {
                // add some extras for CPU nodes
                max_num_threads = opt_num_infer + NRTD_NUM_CPU_THREADS;
            } else {
                LOG(WARNING) << "model_config with opt_num_infer=" << opt_num_infer
                             << " is invalid; using default value "
                             << max_num_threads  << " instead.";
            }
        }
        std::string ninfer_str = env_get("NEURON_MAX_NUM_INFERS", "");
        bool num_infer_is_negative = false;
        if (!ninfer_str.empty()) {
            int64 env_ninfer = (int64)stoi_no_throw(ninfer_str);
            if (env_ninfer < -HARD_MAX_NUM_THREADS || env_ninfer > HARD_MAX_NUM_THREADS) {
                LOG(WARNING) << "NEURON_MAX_NUM_INFERS=" << ninfer_str
                             << " is invalid; using default value "
                             << max_num_threads  << " instead.";
            } else if (env_ninfer < 0) {
                num_infer_is_negative = true;
                max_num_threads = -env_ninfer;
            } else if (0 == env_ninfer) {
                LOG(WARNING) << "NEURON_MAX_NUM_INFERS=0 is invalid; using 1 instead.";
                max_num_threads = 1;
            } else {
                max_num_threads = env_ninfer;
            }
        }
        if (model_config_valid(model_config)) {
            // enforce max_num_threads = 1 if ncg size is insufficient
            int64 int64_opt_num_cores = model_config_this_opt_num_cores(model_config);
            if (int64_opt_num_cores < min_num_cores || int64_opt_num_cores > max_num_cores) {
                max_num_threads = NRTD_INSUFFICIENT_NUM_INFER;
            } else {
                uint32_t opt_num_cores = (uint32_t)int64_opt_num_cores;
                if (num_cores < opt_num_cores) {
                    max_num_threads = NRTD_INSUFFICIENT_NUM_INFER;
                }
            }
        }
        max_num_threads = max_num_threads > 1 ? max_num_threads : 1;
        max_num_threads = std::min(max_num_threads, HARD_MAX_NUM_THREADS);
        max_num_infers_ = (uint32_t)max_num_threads;
        ninfer_ = num_infer_is_negative ? max_num_infers_ : max_num_infers_ + 1;
    }

    void parse_device_index(AttrList &model_config) {
        device_index_ = model_config_device_index(model_config);
    }

    int64_t opt_device_size_ = -1;
    int64_t max_num_duplicates_ = 1;
    uint32_t max_num_infers_ = 4;
    uint32_t timeout_ = 2;
    uint32_t ninfer_ = 5;
    int64_t device_index_ = -1;
private:
    bool model_config_valid(AttrList &model_config) {
        return model_config.i_size() >= 4;
    }
    int64 model_config_global_opt_num_cores(AttrList &model_config) {
        return model_config.i(0);
    }
    int64 model_config_this_opt_num_cores(AttrList &model_config) {
        return model_config.i(1);
    }
    int64 model_config_max_num_duplicates(AttrList &model_config) {
        return model_config.i(2);
    }
    int64 model_config_timeout(AttrList &model_config) {
        return model_config.i(3);
    }
    int64 model_config_device_index(AttrList &model_config) {
        if (model_config.i_size() >= 5) {
            return model_config.i(4);
        } else {
            return -1;
        }
    }

    static const int64 DEFAULT_MAX_NUM_INFER = 4;
    static const int64 NRTD_INSUFFICIENT_NUM_INFER = 1;
    static const int64 NRTD_NUM_CPU_THREADS = 3;
    static const int64 HARD_MAX_NUM_THREADS = 1024;

    TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronModelConfig);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_KERNELS_MODEL_CONFIG_H_
