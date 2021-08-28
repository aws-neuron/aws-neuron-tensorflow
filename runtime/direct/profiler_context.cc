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

#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace neuron {
ProfilerContext::ProfilerContext(const NrtModel& model, std::string profile_dir,
                                 const StringPiece& executable) {
  model_ = model;

  status_ = Env::Default()->RecursivelyCreateDir(profile_dir);
  if (!status_.ok()) {
    LOG(ERROR) << "Cannot create directory for neuron-profile; turning off "
                  "profiler ...";
    return;
  }

  std::string filename_neff = profile_dir + "/someneffname.neff";

  std::unique_ptr<WritableFile> file;
  status_ = Env::Default()->NewWritableFile(filename_neff, &file);
  if (!status_.ok()) {
    LOG(ERROR) << "Failed create neff file..., turning off profiler";
    return;
  }
  status_ = file->Append(executable);
  if (!status_.ok()) {
    LOG(ERROR) << "Failed to write to neff file...";
  }
  file->Close();

  path_to_profile_file_ = profile_dir + "/someopname.ntff";
  Nrt::ProfileStart(model, get_path_to_profile_file());
}
ProfilerContext::ProfilerContext() {}

ProfilerContext::~ProfilerContext() {}

const char* ProfilerContext::get_path_to_profile_file() {
  return path_to_profile_file_.c_str();
}

}  // namespace neuron
}  // namespace tensorflow
