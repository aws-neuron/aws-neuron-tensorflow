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

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/neuron/tf2hlo/compile.h"
#include "tensorflow/neuron/tf2hlo/flags.h"

namespace tensorflow {
namespace neuron {

const char kUsageHeader[] =
    "aws_neuron_tf2hlo performs ahead-of-time compilation of a TensorFlow "
    "graph,\n"
    "resulting in a serialized HloSnapshot protobuf file.\n"
    "A typical invocation looks like this:\n"
    "\n"
    "   $ aws_neuron_tf2hlo --graph=mygraph.pb --config=myfile.pb "
    "--out_session_module=hlo_snapshot.pb\n"
    "\n";

}  // end namespace neuron
}  // end namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::tfcompile::MainFlags flags;

  std::vector<tensorflow::Flag> flag_list;
  AppendMainFlags(&flag_list, &flags);
  xla::AppendDebugOptionsFlags(&flag_list);

  tensorflow::string usage = tensorflow::neuron::kUsageHeader;
  usage += tensorflow::Flags::Usage(argv[0], flag_list);
  if (argc > 1 && absl::string_view(argv[1]) == "--help") {
    std::cerr << usage << "\n";
    return 0;
  }
  bool parsed_flags_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  QCHECK(parsed_flags_ok) << "\n" << usage;

  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(argc == 1) << "\nERROR: This command does not take any arguments "
                       "other than flags. See --help.\n\n";
  tensorflow::Status status = tensorflow::neuron::Main(flags);
  if (status.code() == tensorflow::error::INVALID_ARGUMENT) {
    std::cerr << "INVALID ARGUMENTS: " << status.error_message() << "\n\n";
    return 1;
  } else {
    TF_QCHECK_OK(status);
  }
  return 0;
}
