/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_KAENA_UTIL_LOGGING
#define TENSORFLOW_CONTRIB_KAENA_UTIL_LOGGING

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace kaena {
namespace logging {

int LogLevelStrToInt(const char* tf_env_var_val);

#define EI_LOG_IS_ON(lvl) ((lvl) <= tensorflow::kaena::logging::LogLevelStrToInt(   \
  std::getenv("ENABLE_EI_DBG_LOGGING")))

#define EILOG(lvl)                         \
  if (TF_PREDICT_FALSE(EI_LOG_IS_ON(lvl))) \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, tensorflow::INFO)


#define INFERENTIA_OP_ERROR(CTX, ...) {                                     \
  ::tensorflow::Status _s = tensorflow::errors::Unknown(__VA_ARGS__);       \
  LOG(ERROR) << "InferentiaOp kernel Error at "                             \
             << __FILE__ << ":" << __LINE__ << " : " << _s;   \
  CTX->SetStatus(_s);                                                       \
  return;                                                                   \
}

#define KAENA_LOG_IS_ON(lvl)                                                \
    ((lvl) <= tensorflow::kaena::logging::LogLevelStrToInt(                 \
        std::getenv("ENABLE_KAENA_DBG_LOGGING")))

#define KAENALOG(lvl)                                                       \
    if (TF_PREDICT_FALSE(KAENA_LOG_IS_ON(lvl)))                             \
        ::tensorflow::internal::LogMessage(__FILE__, __LINE__, tensorflow::INFO)

#define KAENALOG_ERROR()                                                    \
    ::tensorflow::internal::LogMessage("KAENA-ERROR: ", 0, tensorflow::INFO)


}  // namespace logging
}  // namespace kaena
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_KAENA_UTIL_LOGGING
