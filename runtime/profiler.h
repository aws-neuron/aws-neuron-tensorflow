/* Copyright 2020 AWS Neuron. All Rights Reserved.

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

#ifndef TENSORFLOW_NEURON_RUNTIME_PROFILER_H_
#define TENSORFLOW_NEURON_RUNTIME_PROFILER_H_

namespace tensorflow {
namespace neuron {

class ProfilerInterface {
public:
    void initialize(const std::string &profile_dir, const std::string &op_name);
    void dump_info(const std::string &graph_def, const std::string &executable);
    void start_session(const std::string &nrtd_address, const uint32_t nn_id);
    void stop_session();
    bool enabled_ = false;
private:
    int session_id_ = 0;
    std::string mangled_op_name_ = "";
    std::string profile_dir_ = "";
    std::string session_filename_ = "";
};
\
}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_PROFILER_H_
