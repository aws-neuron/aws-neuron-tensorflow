#include "profiler_context.h"

#include <fstream>

#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace neuron {
ProfilerContext::ProfilerContext(const NrtModel& model, std::string profile_dir,
                                 const StringPiece& executable) {
  model_ = model;
  profile_dir_ = profile_dir;

  Status status = Env::Default()->RecursivelyCreateDir(profile_dir_);
  if (!status.ok()) {
    LOG(ERROR) << "Cannot create directory for neuron-profile; turning off "
                  "profiler ...";
    return;
  }

  std::string filename_neff = profile_dir + "/someneffname.neff";
  // std::ofstream(filename_neff, std::ios::binary) << executable; //better
  // function for this

  std::unique_ptr<WritableFile> file;
  Env::Default()->NewWritableFile(filename_neff, &file);
  status = file->Append(executable);
  file->Close();

  const char* filename_ntff = (profile_dir + "/someopname.ntff").c_str();
  Nrt::ProfileStart(model, filename_ntff);
}
ProfilerContext::ProfilerContext() {}

ProfilerContext::~ProfilerContext() {
  Nrt::ProfileStop(this->profile_dir_.c_str());
}

}  // namespace neuron
}  // namespace tensorflow
