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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace neuron {

#define NRT_INVALID_NN_ID 0
#define NRT_INVALID_EG_ID 0

#define SYS_FAIL_RETURN(failure_expr, fn_name) {                            \
    if (failure_expr) {                                                     \
        return errors::Internal((fn_name), " failed with errno ", errno);   \
    }                                                                       \
}

#define SYS_FAIL_LOG(failure_expr, fn_name) {                       \
    if (failure_expr) {                                             \
        LOG(ERROR) << (fn_name) << " failed with errno " << errno;  \
    }                                                               \
}

#define SYS_FAIL_LOG_RETURN(failure_expr, fn_name) {                \
    if (failure_expr) {                                             \
        LOG(ERROR) << (fn_name) << " failed with errno " << errno;  \
        return;                                                     \
    }                                                               \
}

#define TF_LOG_RETURN_IF_ERROR(...) {                                   \
    Status _status = (__VA_ARGS__);                                     \
    if (TF_PREDICT_FALSE(!_status.ok())) {                              \
        LOG(ERROR) << "error code " << _status.code()                   \
                   << ", error message " << _status.error_message();    \
        return;                                                         \
    }                                                                   \
}

#define TF_LOG_IF_ERROR(status) {   \
    if (!(status).ok()) {           \
        LOG(ERROR) << (status);     \
    }                               \
}

// Note: this macro must be used after ctx->allocate_output
#define OK_IGNORE_ABORTED(CTX, ...) {                               \
    Status status(__VA_ARGS__);                                     \
    if (status.code() == tensorflow::error::Code::ABORTED) {        \
        VLOG(1) << "ignored error " << status.error_message();      \
        return;                                                     \
    }                                                               \
    OP_REQUIRES_OK(CTX, status);                                    \
}

#define TF_VERSION_LESS_THAN(MAJOR, MINOR) \
    (TF_MAJOR_VERSION < (MAJOR) || (TF_MAJOR_VERSION == (MAJOR) && TF_MINOR_VERSION < (MINOR)))

}  // neuron
}  // tensorflow
