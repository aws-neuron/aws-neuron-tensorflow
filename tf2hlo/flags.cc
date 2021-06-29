/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/neuron/tf2hlo/flags.h"

namespace tensorflow {
namespace tfcompile {

void AppendMainFlags(std::vector<Flag>* flag_list, MainFlags* flags) {
  const std::vector<Flag> tmp = {
      {"graph", &flags->graph,
       "Input GraphDef file.  If the file ends in '.pbtxt' it is expected to "
       "be in the human-readable proto text format, otherwise it is expected "
       "to be in the proto binary format."},
      {"config", &flags->config,
       "Input file containing Config proto.  If the file ends in '.pbtxt' it "
       "is expected to be in the human-readable proto text format, otherwise "
       "it is expected to be in the proto binary format."},
      {"out_session_module", &flags->out_session_module,
       "Output session module proto."},
      {"in_session_module", &flags->in_session_module,
       "Input session module proto, for verification purpose."},
  };
  flag_list->insert(flag_list->end(), tmp.begin(), tmp.end());
}

}  // namespace tfcompile
}  // namespace tensorflow
