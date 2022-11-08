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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace neuron {

typedef const AttrValue_ListValue AttrList;

#define NRT_INVALID_NN_ID 0
#define NRT_INVALID_EG_ID 0

#define SYS_FAIL_RETURN(failure_expr, fn_name)                          \
  {                                                                     \
    if (TF_PREDICT_FALSE(failure_expr)) {                               \
      return errors::Internal((fn_name), " failed with errno ", errno); \
    }                                                                   \
  }

#define SYS_FAIL_LOG(failure_expr, fn_name)                      \
  {                                                              \
    if (TF_PREDICT_FALSE(failure_expr)) {                        \
      LOG(ERROR) << (fn_name) << " failed with errno " << errno; \
    }                                                            \
  }

#define SYS_FAIL_LOG_RETURN(failure_expr, fn_name)               \
  {                                                              \
    if (TF_PREDICT_FALSE(failure_expr)) {                        \
      LOG(ERROR) << (fn_name) << " failed with errno " << errno; \
      return;                                                    \
    }                                                            \
  }

#define TF_LOG_RETURN_IF_ERROR(...)                                       \
  {                                                                       \
    Status _status = (__VA_ARGS__);                                       \
    if (TF_PREDICT_FALSE(!_status.ok())) {                                \
      LOG(ERROR) << "error code " << _status.code() << ", error message " \
                 << _status.error_message();                              \
      return;                                                             \
    }                                                                     \
  }

#define TF_LOG_IF_ERROR(status)             \
  {                                         \
    if (TF_PREDICT_FALSE(!(status).ok())) { \
      LOG(ERROR) << (status);               \
    }                                       \
  }

#define TFN_CHECK_ARG(cond, ...)                                       \
  {                                                                    \
    if (TF_PREDICT_FALSE(cond)) {                                      \
      return errors::InvalidArgument(__FILE__, __LINE__, __VA_ARGS__); \
    }                                                                  \
  }

#define TFN_DISALLOW_COPY_MOVE_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;           \
  void operator=(const TypeName&) = delete;     \
  TypeName(TypeName&&);                         \
  void operator=(TypeName&&) = delete;

#define TF_VERSION_LESS_THAN(MAJOR, MINOR) \
  (TF_MAJOR_VERSION < (MAJOR) ||           \
   (TF_MAJOR_VERSION == (MAJOR) && TF_MINOR_VERSION < (MINOR)))

#define VLOG_TIME_BASE(start, lvl, msg) \
  VLOG(lvl) << msg << " " << Env::Default()->NowMicros() - start;

#define CHECK_VALID_PTR(ptr)            \
  if (TF_PREDICT_FALSE(ptr == nullptr)) \
    return errors::InvalidArgument("null pointer ", (#ptr), "(", (ptr), ")");

#define CHECK_SIZES_MATCH(lhs_size, rhs_size)                             \
  if (TF_PREDICT_FALSE((int64)(lhs_size) != (int64)(rhs_size)))           \
    return errors::InvalidArgument("size mismatch: ", (#lhs_size),        \
                                   " == ", (lhs_size), ", ", (#rhs_size), \
                                   " == ", (rhs_size));

#define TFN_RETURN_IF_NULLPTR(ptr)                                  \
  if (TF_PREDICT_FALSE(nullptr == (ptr))) {                         \
    return errors::InvalidArgument(__func__, " called on nullptr"); \
  }
#define TFN_RETURN_IF_ZERO_SIZE(size)                                   \
  if (TF_PREDICT_FALSE(0 == (size))) {                                  \
    return errors::InvalidArgument(__func__, " called with size == 0"); \
  }
#define TFN_RETURN_FAILED_PRECONDITION_IF_ERROR(status)           \
  if (TF_PREDICT_FALSE(!(status).ok())) {                         \
    return errors::FailedPrecondition(                            \
        (__func__), " called without successful initialization"); \
  }

}  // namespace neuron
}  // namespace tensorflow
