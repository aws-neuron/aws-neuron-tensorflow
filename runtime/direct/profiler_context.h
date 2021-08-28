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

#ifndef PROFILER_CONTEXT_H
#define PROFILER_CONTEXT_H

#include "adaptor.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace neuron {
class ProfilerContext {
 private:
  NrtModel model_;
  std::string path_to_profile_file_;
  Status status_;

 public:
  ProfilerContext(const NrtModel& model, std::string profile_dir,
                  const StringPiece& executable);
  ProfilerContext();
  ~ProfilerContext();
  const char* get_path_to_profile_file();
};
}  // namespace neuron
}  // namespace tensorflow

#endif