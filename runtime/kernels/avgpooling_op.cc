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
#include <iostream>

#include "../device.h"
#include "registration.h"
// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

//#include "tensorflow/core/kernels/avgpooling_op.h"

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
//#include "tensorflow/core/kernels/eigen_pooling.h"
//#include "tensorflow/core/kernels/ops_util.h"
//#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace neuron {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class AvgPoolingOp : public UnaryOp<T> {
 public:
  explicit AvgPoolingOp(OpKernelConstruction* context) : UnaryOp<T>(context) {
    VLOG(1) << "using neuron implementation";
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    /*OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default AvgPoolingOp only supports NHWC ",
                                "on device type ",
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
  }

  void Compute(OpKernelContext* context) override {
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
    TensorShape newinputshape;
    Tensor paddedTensor;

    if (data_format_ == FORMAT_NHWC) {
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
      VLOG(1) << "unrecognized format, but didn't get caught by error checking";
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
      VLOG(1) << "something is wrong";
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

      VLOG(1) << "must be related to allocation";
      OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, newinputshape,
                                                     &paddedTensor));
      VLOG(1) << "must be related to allocation";

      auto newmatrix = paddedTensor.tensor<float, 4>();
      auto t = tensor_in.tensor<float, 4>();
      float inf = std::numeric_limits<float>::infinity();

      if (data_format_ == FORMAT_NHWC) {
        // create the padded tensor surrounded with proper padding
        for (int b = 0; b < batch_size; b++) {
          for (int r = 0; r < paddedRowLen; r++) {
            for (int c = 0; c < paddedColLen; c++) {
              for (int ch = 0; ch < channels; ch++) {
                if ((r < row_offset_up || c < col_offset_left) ||
                    (r >= paddedRowLen - row_offset_down ||
                     c >= paddedColLen - col_offset_right)) {
                  newmatrix(b, r, c, ch) = inf;
                } else {
                  newmatrix(b, r, c, ch) =
                      t(b, r - row_offset_up, c - col_offset_left, ch);
                }
              }
            }
          }
        }
      } else {
        for (int b = 0; b < batch_size; b++) {
          for (int ch = 0; ch < channels; ch++) {
            for (int r = 0; r < paddedRowLen; r++) {
              for (int c = 0; c < paddedColLen; c++) {
                if ((r < row_offset_up || c < col_offset_left) ||
                    (r >= paddedRowLen - row_offset_down ||
                     c >= paddedColLen - col_offset_right)) {
                  newmatrix(b, ch, r, c) = inf;
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

    if (data_format_ == FORMAT_NHWC) {
      outputshape = TensorShape{batch_size, newNumRows, newNumCols, channels};
    } else {
      outputshape = TensorShape{batch_size, channels, newNumRows, newNumCols};
    }
    OP_REQUIRES_OK(context, context->allocate_output(0, outputshape, &output));
    auto outmatrix = output->tensor<float, 4>();

    // VALID PADDING
    if (padding_ == 1) {
      if (data_format_ == FORMAT_NHWC) {
        for (int b = 0; b < batch_size; b++) {
          for (int r = 0, r1 = 0; r1 < newNumRows; r += rowStrideLen, r1++) {
            for (int c = 0, c1 = 0; c1 < newNumCols; c += colStrideLen, c1++) {
              for (int ch = 0; ch < channels; ch++) {
                outmatrix(b, r1, c1, ch) = calculate_sum_and_average_valid(
                    tensor_in, b, r, c, ch, rowKernelSize, colKernelSize, true);
              }
            }
          }
        }
      } else {
        for (int b = 0; b < batch_size; b++) {
          for (int ch = 0; ch < channels; ch++) {
            for (int r = 0, r1 = 0; r1 < newNumRows; r += rowStrideLen, r1++) {
              for (int c = 0, c1 = 0; c1 < newNumCols;
                   c += colStrideLen, c1++) {
                outmatrix(b, ch, r1, c1) = calculate_sum_and_average_valid(
                    tensor_in, b, r, c, ch, rowKernelSize, colKernelSize,
                    false);
              }
            }
          }
        }
      }
    }
    // SAME PADDING
    else if (padding_ == 2) {
      VLOG(1) << "we at least get to here";
      if (data_format_ == FORMAT_NHWC) {
        for (int b = 0; b < batch_size; b++) {
          for (int r = 0, r1 = 0; r1 < newNumRows; r += rowStrideLen, r1++) {
            for (int c = 0, c1 = 0; c1 < newNumCols; c += colStrideLen, c1++) {
              for (int ch = 0; ch < channels; ch++) {
                outmatrix(b, r1, c1, ch) = calculate_sum_and_average_same(
                    paddedTensor, b, r, c, ch, rowKernelSize, colKernelSize,
                    true);
              }
            }
          }
        }
      } else {
        for (int b = 0; b < batch_size; b++) {
          for (int ch = 0; ch < channels; ch++) {
            for (int r = 0, r1 = 0; r1 < newNumRows; r += rowStrideLen, r1++) {
              for (int c = 0, c1 = 0; c1 < newNumCols;
                   c += colStrideLen, c1++) {
                outmatrix(b, ch, r1, c1) = calculate_sum_and_average_same(
                    paddedTensor, b, r, c, ch, rowKernelSize, colKernelSize,
                    false);
              }
            }
          }
        }
      }
    }

    // context->set_output(0, context->input(0));
    // SpatialAvgPool<Device, T>(context, output, tensor_in, params, padding_);
  }
  static float calculate_sum_and_average_valid(const Tensor& tensor_in, int b,
                                               int r, int c, int ch,
                                               int rowKernelSize,
                                               int colKernelSize,
                                               bool channelsLast) {
    auto t = tensor_in.tensor<float, 4>();
    float sum = 0;
    float total = rowKernelSize * colKernelSize;
    if (channelsLast) {
      for (int r_offset = 0; r_offset < rowKernelSize; r_offset++) {
        for (int c_offset = 0; c_offset < colKernelSize; c_offset++) {
          sum += t(b, r + r_offset, c + c_offset, ch);
        }
      }
    }

    else {
      for (int r_offset = 0; r_offset < rowKernelSize; r_offset++) {
        for (int c_offset = 0; c_offset < colKernelSize; c_offset++) {
          sum += t(b, ch, r + r_offset, c + c_offset);
        }
      }
    }
    return sum / total;
  }
  static float calculate_sum_and_average_same(const Tensor& tensor_in, int b,
                                              int r, int c, int ch,
                                              int rowKernelSize,
                                              int colKernelSize,
                                              bool channelsLast) {
    auto t = tensor_in.tensor<float, 4>();
    float sum = 0;
    float total = rowKernelSize * colKernelSize;
    float value;
    float inf = std::numeric_limits<float>::infinity();
    if (channelsLast) {
      for (int r_offset = 0; r_offset < rowKernelSize; r_offset++) {
        for (int c_offset = 0; c_offset < colKernelSize; c_offset++) {
          value = t(b, r + r_offset, c + c_offset, ch);
          if (value == inf) {
            total--;
          } else {
            sum += value;
          }
        }
      }
    }

    else {
      for (int r_offset = 0; r_offset < rowKernelSize; r_offset++) {
        for (int c_offset = 0; c_offset < colKernelSize; c_offset++) {
          value = t(b, ch, r + r_offset, c + c_offset);
          if (value == inf) {
            total--;
          } else {
            sum += value;
          }
        }
      }
    }
    return sum / total;
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

typedef AvgPoolingOp<CPUDevice, float> AvgPoolCPUFloat;

NEURON_REGISTER_KERNEL_BUILDER("AvgPool", DEVICE_NEURON, AvgPoolCPUFloat);

}  // namespace neuron
}  // namespace tensorflow
