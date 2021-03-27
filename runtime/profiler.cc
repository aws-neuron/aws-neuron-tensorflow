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

#include "profiler.h"
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fstream>
#include "macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace neuron {

template <typename... Args>
Status subprocess_run(Args... args) {
  pid_t fork_pid;
  SYS_FAIL_RETURN((fork_pid = fork()) < 0, "fork");
  if (0 == fork_pid) {
    execlp(args..., (char*)NULL);
    LOG(ERROR) << "execlp failed";
    _exit(1);
  } else {
    int status;
    SYS_FAIL_RETURN(waitpid(fork_pid, &status, 0) < 0, "waitpid");
    if (!(WIFEXITED(status) && 0 == WEXITSTATUS(status))) {
      return errors::Internal("child process did not exit gracefully");
    }
  }
  return Status::OK();
}

static std::string mangle_op_name(const std::string& op_name) {
  std::string new_op_name(op_name);
  for (size_t idx = 0; idx < new_op_name.length(); ++idx) {
    if ('/' == new_op_name[idx]) {
      new_op_name[idx] = '+';
    }
  }
  return new_op_name;
}

void ProfilerInterface::initialize(const std::string& profile_dir,
                                   const std::string& op_name) {
  profile_dir_ = profile_dir;
  mangled_op_name_ = mangle_op_name(op_name);
  enabled_ = !profile_dir_.empty();
}

void ProfilerInterface::dump_info(const std::string& graph_def,
                                  const StringPiece& executable) {
  Status status = Env::Default()->RecursivelyCreateDir(profile_dir_);
  if (!status.ok()) {
    LOG(ERROR) << "Cannot create directory for neuron-profile; turning off "
                  "profiler ...";
    enabled_ = false;
    return;
  }
  std::string filename_base = profile_dir_ + "/" + mangled_op_name_;
  std::string filename_pb = filename_base + ".pb";
  std::string filename_neff = filename_base + ".neff";
  std::ofstream(filename_pb, std::ios::binary) << graph_def;
  std::ofstream(filename_neff, std::ios::binary) << executable;
}

void ProfilerInterface::start_session(const std::string& nrtd_address,
                                      const uint32_t nn_id) {
  if (!enabled_) {
    VLOG(1) << "Skipping start_session as profiler is not enabled";
    return;
  }
  std::ostringstream filename_stream;
  filename_stream << profile_dir_ << "/" << mangled_op_name_ << "-" << nn_id
                  << "-" << session_id_ << ".ntff";
  session_filename_ = filename_stream.str();
  std::ostringstream cmd_stream;
  cmd_stream << "neuron-profile start-session -s " << session_filename_
             << " -a " << nrtd_address << " " << nn_id;
  VLOG(1) << "Starting profiling session by " << cmd_stream.str();
  std::ostringstream nn_id_stream;
  nn_id_stream << nn_id;
  Status status =
      subprocess_run("neuron-profile", "neuron-profile", "start-session", "-s",
                     session_filename_.c_str(), "-a", nrtd_address.c_str(),
                     nn_id_stream.str().c_str());
  if (!status.ok()) {
    session_filename_ = "";
    LOG(WARNING) << "neuron-profile start-session failed. "
                 << "Did you install aws-neuron-tools?";
    return;
  }
  session_id_++;
}

void ProfilerInterface::stop_session() {
  if (!enabled_) {
    VLOG(1) << "Skipping stop_session as profiler is not enabled";
    return;
  }
  if (!session_filename_.empty()) {
    std::ostringstream cmd_stream;
    cmd_stream << "neuron-profile stop-session -s " << session_filename_;
    VLOG(1) << "Stopping profiling session by " << cmd_stream.str();
    Status status =
        subprocess_run("neuron-profile", "neuron-profile", "stop-session", "-s",
                       session_filename_.c_str());
    if (!status.ok()) {
      LOG(ERROR) << "neuron-profile stop-session failed";
    }
    status = subprocess_run("neuron-profile", "neuron-profile", "show-session",
                            "-s", session_filename_.c_str());
    if (!status.ok()) {
      LOG(ERROR) << "neuron-profile show-session failed";
    }
    session_filename_ = "";
  }
}

}  // namespace neuron
}  // namespace tensorflow
