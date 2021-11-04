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


#include "pooling_utils.h"

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
          newNumCols, std::numeric_limits<float>::infinity(), data_format_ == FORMAT_NHWC);

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
        pooling_func = calculate_sum_and_average;
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

  static float calculate_sum_and_average(const Tensor& tensor_in, int b, int r,
                                         int c, int ch, int rowKernelSize,
                                         int colKernelSize, bool channelsLast) {
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

  static float special_case_function(const Tensor& tensor_in, int b, int ch, int r,
                            int c, bool channelsLast) {
    auto t = tensor_in.tensor<float, 4>();
    float sum = 0;
    float total = r * c;
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        // called very little so I think we can get away with if inside the
        // for loops here
        if (channelsLast) {
          sum += t(b, i, j, ch);
        } else {
          sum += t(b, ch, i, j);
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
