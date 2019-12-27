/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_NEURON_CLIB_PROFILER_H_
#define TENSORFLOW_NEURON_NEURON_CLIB_PROFILER_H_


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


}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_NEURON_CLIB_PROFILER_H_
