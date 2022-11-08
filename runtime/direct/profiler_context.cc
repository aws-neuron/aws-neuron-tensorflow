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

#include "profiler_context.h"

#include <fstream>

#include "adaptor.h"
#include "env.h"
#include "executable_info.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace neuron {

ProfilerContext::ProfilerContext(const NrtModel& model,
                                 const std::string& profile_dir,
                                 const NeuronExecutableInfo& info) {
  model_ = model;

  status_ = Env::Default()->RecursivelyCreateDir(profile_dir);
  if (!status_.ok()) {
    LOG(ERROR) << "Cannot create directory for neuron-profile; turning off "
                  "profiler ...";
    return;
  }

  std::string mangled_op_name = mangle_op_name(info.name);
  std::string filename_prefix = profile_dir + "/" + mangled_op_name;

  std::string filename_neff = filename_prefix + ".neff";
  status_ = WriteStringToFile(Env::Default(), filename_neff, info.executable);
  if (!status_.ok()) {
    LOG(ERROR) << "Failed create neff file..., turning off profiler";
    return;
  }

  std::string filename_graph_def = filename_prefix + ".graph_def.pb";
  status_ = WriteStringToFile(Env::Default(), filename_graph_def,
                              info.serialized_graph_def);
  if (!status_.ok()) {
    LOG(ERROR) << "Failed create graph_def file..., turning off profiler";
    return;
  }

  path_to_profile_file_ = filename_prefix + ".ntff";
  status_ = Nrt::ProfileStart(model, get_path_to_profile_file());
  if (!status_.ok()) {
    LOG(ERROR) << "Failed to start profiling at " << get_path_to_profile_file();
    return;
  }
}
ProfilerContext::ProfilerContext() {}

ProfilerContext::~ProfilerContext() {
  status_ = Nrt::ProfileStop(get_path_to_profile_file());
  if (!status_.ok()) {
    LOG(ERROR) << "Failed to stop profiling at " << get_path_to_profile_file();
    return;
  }
}

const char* ProfilerContext::get_path_to_profile_file() {
  return path_to_profile_file_.c_str();
}

}  // namespace neuron
}  // namespace tensorflow
