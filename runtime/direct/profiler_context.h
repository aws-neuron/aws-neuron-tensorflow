#ifndef PROFILER_CONTEXT_H
#define PROFILER_CONTEXT_H

#include "adaptor.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace neuron {
class ProfilerContext {
 private:
  NrtModel model_;
  std::string profile_dir_ = "";

 public:
  ProfilerContext(const NrtModel& model, std::string profile_dir,
                  const StringPiece& executable);
  ProfilerContext();
  ~ProfilerContext();
};
}  // namespace neuron
}  // namespace tensorflow

#endif