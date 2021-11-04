/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "../device.h"
#include "registration.h"

#include "pooling_utils.h"


//#include "tensorflow/core/kernels/maxpooling_op.h"

#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace neuron {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MaxPoolingNoMaskOp : public OpKernel {
 public:
  explicit MaxPoolingNoMaskOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    /*OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument(
            "Default MaxPoolingNoMaskOp only supports NHWC on device type ",
            DeviceTypeString(context->device_type())));*/

    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(
        context, padding_ != EXPLICIT,
        errors::Unimplemented(
            "Explicit padding is not supported for MaxPoolingNoMaskOp."));
  }
  void Compute(OpKernelContext* context) override {
    VLOG(1) << "Starting AvgPool compute.";
    const Tensor& tensor_in = context->input(0);
    if (!context->status().ok()) {
      return;
    }

    // For avgpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));

    Tensor* output = nullptr;

    int batch_size = tensor_in.shape().dim_size(0);
    int rows;
    int cols;
    int channels;
    int newNumRows;
    int newNumCols;
    int rowKernelSize;
    int colKernelSize;
    int rowStrideLen;
    int colStrideLen;

    TensorShape outputshape;
    Tensor paddedTensor;

    init_basic_info(tensor_in, ksize_, stride_, data_format_ == FORMAT_NHWC,
                    rows, cols, channels, rowKernelSize, colKernelSize,
                    rowStrideLen, colStrideLen);

    if (padding_ == 1) {
      VLOG(1) << "Valid Padding";
      valid_padding_new_num_rows_and_cols(rows, cols, rowKernelSize,
                                          colKernelSize, rowStrideLen,
                                          colStrideLen, newNumRows, newNumCols);

    } else if (padding_ == 2) {
      VLOG(1) << "Same Padding";

      same_padding_new_num_rows_and_cols(
          tensor_in, paddedTensor, context, rows, cols, batch_size, channels,
          rowKernelSize, colKernelSize, rowStrideLen, colStrideLen, newNumRows,
          newNumCols, -std::numeric_limits<float>::infinity(), data_format_ == FORMAT_NHWC);

    } else {
      VLOG(1)
          << "unrecognized padding type but not caught durring error checking";
    }

    VLOG(1) << "paddedTensor shape: " << paddedTensor.DebugString();

    if (data_format_ == FORMAT_NHWC) {
      outputshape = TensorShape{batch_size, newNumRows, newNumCols, channels};
    } else {
      outputshape = TensorShape{batch_size, channels, newNumRows, newNumCols};
    }

    OP_REQUIRES_OK(context, context->allocate_output(0, outputshape, &output));

    VLOG(1) << "Output Shape: " << output->DebugString();
    auto outmatrix = output->tensor<float, 4>();

    // this is a special case where we just average the whole input tensor
    if (newNumRows == 1 && newNumCols == 1) {
      VLOG(1) << "Using the special case!";
      std::function<float(Tensor, int, int, int, int, bool)> special_case_func =
          special_case_function;

      special_case(tensor_in, output, rows, cols, batch_size, special_case_func, channels,
                   data_format_ == FORMAT_NHWC);

      // don't need to do any more calculations
      return;
    }

    std::function<float(Tensor, int, int, int, int, int, int, bool)>
        pooling_func = calculate_max;
    // valid padding
    // need to distinguish based on whether or not to use paddedTensor or
    // tensor_in
    if (padding_ == 1) {
      do_pooling(tensor_in, output, batch_size, newNumRows, rowStrideLen,
                 rowKernelSize, newNumCols, colStrideLen, colKernelSize,
                 channels, pooling_func, data_format_ == FORMAT_NHWC);
    } else if (padding_ == 2) {
      do_pooling(paddedTensor, output, batch_size, newNumRows, rowStrideLen,
                 rowKernelSize, newNumCols, colStrideLen, colKernelSize,
                 channels, pooling_func, data_format_ == FORMAT_NHWC);
    }
  }

  /*  if (data_format_ == FORMAT_NHWC) {
      rows = tensor_in.shape().dim_size(1);
      cols = tensor_in.shape().dim_size(2);
      channels = tensor_in.shape().dim_size(3);
      rowKernelSize = ksize_[1];
      colKernelSize = ksize_[2];
      rowStrideLen = stride_[1];
      colStrideLen = stride_[2];
    } else if (data_format_ == FORMAT_NCHW) {
      rows = tensor_in.shape().dim_size(2);
      cols = tensor_in.shape().dim_size(3);
      channels = tensor_in.shape().dim_size(1);
      rowKernelSize = ksize_[2];
      colKernelSize = ksize_[3];
      rowStrideLen = stride_[2];
      colStrideLen = stride_[3];
    } else
      VLOG(1) << "Unrecognized format, but didn't get caught by error checking";
    if (padding_ == 1) {
      VLOG(1) << "Valid Padding";

      // now we must calculate the actual newNumRows/Cols if striding is in play
      newNumRows = rows - rowKernelSize + 1;
      newNumCols = cols - colKernelSize + 1;
      // ceiling(newNumRows/rowStrideLen)
      newNumRows = (newNumRows + rowStrideLen - 1) / rowStrideLen;
      // ceiling(newNumRows/rowStrideLen)
      newNumCols = (newNumCols + colStrideLen - 1) / colStrideLen;

    } else if (padding_ == 2) {
      VLOG(1) << "Same Padding";
      // int row_offset = rowKernelSize / 2;
      // int col_offset = colKernelSize / 2;
      int totalRowPadding = rowKernelSize - 1;
      int totalColPadding = colKernelSize - 1;
      int paddedRowLen = rows + totalRowPadding;
      int paddedColLen = cols + totalColPadding;
      // ceiling(paddedRowLen/rowStrideLen)
      newNumRows = ((paddedRowLen - rowKernelSize) / rowStrideLen) + 1;
      newNumCols = ((paddedColLen - colKernelSize) / colStrideLen) + 1;
      // newNumRows and newNumCols to be used later when generating output shape

      // floor(totalRowPadding / 2)
      int row_offset_up = (totalRowPadding) / 2;
      // ceil(totalRowPadding / 2)
      int row_offset_down = (totalRowPadding + 1) / 2;
      // floor(totalColPadding / 2)
      int col_offset_left = (totalColPadding) / 2;
      // ceil(totalColPadding / 2)
      int col_offset_right = (totalColPadding + 1) / 2;

      if (data_format_ == FORMAT_NHWC) {
        newinputshape =
            TensorShape{batch_size, paddedRowLen, paddedColLen, channels};
      } else {
        newinputshape =
            TensorShape{batch_size, channels, paddedRowLen, paddedColLen};
      }

      OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, newinputshape,
                                                     &paddedTensor));
      VLOG(1) << "Successfully allocated the paddedTensor for same paddding.";

      auto newmatrix = paddedTensor.tensor<float, 4>();
      auto t = tensor_in.tensor<float, 4>();
      float neg_inf = -std::numeric_limits<float>::infinity();

      if (data_format_ == FORMAT_NHWC) {
        VLOG(1) << "Channels last padded tensor being created";
        // create the padded tensor surrounded with proper padding
        for (int b = 0; b < batch_size; b++) {
          for (int r = 0; r < paddedRowLen; r++) {
            for (int c = 0; c < paddedColLen; c++) {
              for (int ch = 0; ch < channels; ch++) {
                if ((r < row_offset_up || c < col_offset_left) ||
                    (r >= paddedRowLen - row_offset_down ||
                     c >= paddedColLen - col_offset_right)) {
                  newmatrix(b, r, c, ch) = neg_inf;
                } else {
                  newmatrix(b, r, c, ch) =
                      t(b, r - row_offset_up, c - col_offset_left, ch);
                }
              }
            }
          }
        }
      } else {
        VLOG(1) << "Channels First padded tensor being created";
        for (int b = 0; b < batch_size; b++) {
          for (int ch = 0; ch < channels; ch++) {
            for (int r = 0; r < paddedRowLen; r++) {
              for (int c = 0; c < paddedColLen; c++) {
                if ((r < row_offset_up || c < col_offset_left) ||
                    (r >= paddedRowLen - row_offset_down ||
                     c >= paddedColLen - col_offset_right)) {
                  newmatrix(b, ch, r, c) = neg_inf;
                } else {
                  newmatrix(b, ch, r, c) =
                      t(b, ch, r - row_offset_up, c - col_offset_left);
                }
              }
            }
          }
        }
      }
    }

    VLOG(1) << "paddedTensor shape: " << paddedTensor.DebugString();

    if (data_format_ == FORMAT_NHWC) {
      outputshape = TensorShape{batch_size, newNumRows, newNumCols, channels};
    } else {
      outputshape = TensorShape{batch_size, channels, newNumRows, newNumCols};
    }

    OP_REQUIRES_OK(context, context->allocate_output(0, outputshape, &output));

    VLOG(1) << "Output Shape: " << output->DebugString();
    auto outmatrix = output->tensor<float, 4>();

    // this is a special case where we just average the whole input tensor
    if (newNumRows == 1 && newNumCols == 1) {
      VLOG(1) << "Using the special case!";
      for (int b = 0; b < batch_size; b++) {
        for (int ch = 0; ch < channels; ch++) {
          // less efficient here because I think we can get away with it as it
          // is only 1x1 image
          if (data_format_ == FORMAT_NHWC) {
            outmatrix(b, 0, 0, ch) =
                special_case(tensor_in, b, ch, rows, cols, true);
          } else {
            outmatrix(b, ch, 0, 0) =
                special_case(tensor_in, b, ch, rows, cols, false);
          }
        }
      }
      // don't need to do any more calculations
      return;
    }

    // VALID PADDING
    if (padding_ == 1) {
      VLOG(1) << "Valid padding AVGPool step";
      if (data_format_ == FORMAT_NHWC) {
        VLOG(1) << "Using channels last format";
        for (int b = 0; b < batch_size; b++) {
          for (int r = 0, r1 = 0; r1 < newNumRows; r += rowStrideLen, r1++) {
            for (int c = 0, c1 = 0; c1 < newNumCols; c += colStrideLen, c1++) {
              for (int ch = 0; ch < channels; ch++) {
                outmatrix(b, r1, c1, ch) = calculate_max(
                    tensor_in, b, r, c, ch, rowKernelSize, colKernelSize, true);
              }
            }
          }
        }
      } else {
        VLOG(1) << "Using channels first format";
        for (int b = 0; b < batch_size; b++) {
          for (int ch = 0; ch < channels; ch++) {
            for (int r = 0, r1 = 0; r1 < newNumRows; r += rowStrideLen, r1++) {
              for (int c = 0, c1 = 0; c1 < newNumCols;
                   c += colStrideLen, c1++) {
                outmatrix(b, ch, r1, c1) =
                    calculate_max(tensor_in, b, r, c, ch, rowKernelSize,
                                  colKernelSize, false);
              }
            }
          }
        }
      }
    }
    // SAME PADDING
    else if (padding_ == 2) {
      VLOG(1) << "Same padding AVG pool step.";
      if (data_format_ == FORMAT_NHWC) {
        VLOG(1) << "Using channels last format";
        for (int b = 0; b < batch_size; b++) {
          for (int r = 0, r1 = 0; r1 < newNumRows; r += rowStrideLen, r1++) {
            for (int c = 0, c1 = 0; c1 < newNumCols; c += colStrideLen, c1++) {
              for (int ch = 0; ch < channels; ch++) {
                outmatrix(b, r1, c1, ch) =
                    calculate_max(paddedTensor, b, r, c, ch, rowKernelSize,
                                  colKernelSize, true);
              }
            }
          }
        }
      } else {
        VLOG(1) << "Using channels first format";
        for (int b = 0; b < batch_size; b++) {
          for (int ch = 0; ch < channels; ch++) {
            for (int r = 0, r1 = 0; r1 < newNumRows; r += rowStrideLen, r1++) {
              for (int c = 0, c1 = 0; c1 < newNumCols;
                   c += colStrideLen, c1++) {
                outmatrix(b, ch, r1, c1) =
                    calculate_max(paddedTensor, b, r, c, ch, rowKernelSize,
                                  colKernelSize, false);
              }
            }
          }
        }
      }
    }

*/
  

  static float calculate_max(const Tensor& tensor_in, int b, int r, int c,
                             int ch, int rowKernelSize, int colKernelSize,
                             bool channelsLast) {
    auto t = tensor_in.tensor<float, 4>();
    float neg_inf = -std::numeric_limits<float>::infinity();
    float max = neg_inf;
    float value;
    if (channelsLast) {
      for (int r_offset = 0; r_offset < rowKernelSize; r_offset++) {
        for (int c_offset = 0; c_offset < colKernelSize; c_offset++) {
          value = t(b, r + r_offset, c + c_offset, ch);
          if (value > max) {
            max = value;
          }
        }
      }
    }

    else {
      for (int r_offset = 0; r_offset < rowKernelSize; r_offset++) {
        for (int c_offset = 0; c_offset < colKernelSize; c_offset++) {
          value = t(b, ch, r + r_offset, c + c_offset);
          if (value > max) {
            max = value;
          }
        }
      }
    }
    return max;
  }

  static float special_case_function(const Tensor& tensor_in, int b, int ch, int r,
                            int c, bool channelsLast) {
    float neg_inf = -std::numeric_limits<float>::infinity();
    float max = neg_inf;
    float value;
    auto t = tensor_in.tensor<float, 4>();
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        // called very little so I think we can get away with if inside the for
        // loops here
        if (channelsLast) {
          value = t(b, i, j, ch);
          if (value > max) {
            max = value;
          }
        } else {
          value = t(b, ch, i ,j);
          if (value > max) {
            max = value;
          }
        }
      }
    }
    return max;
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

typedef MaxPoolingNoMaskOp<CPUDevice, float> MaxPoolCPUFloat;

NEURON_REGISTER_KERNEL_BUILDER("MaxPool", DEVICE_NEURON, MaxPoolCPUFloat);

}  // namespace neuron
}  // namespace tensorflow
