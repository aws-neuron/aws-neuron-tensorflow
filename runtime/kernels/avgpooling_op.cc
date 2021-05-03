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
#include "registration.h"
#include "../device.h"
#include <iostream>
// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

//#include "tensorflow/core/kernels/avgpooling_op.h"

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_shape_util.h"
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
    
    if (data_format_ == FORMAT_NHWC){
      VLOG(1) << "im in this if statement";
      rows = tensor_in.shape().dim_size(1);
      cols = tensor_in.shape().dim_size(2);
      channels = tensor_in.shape().dim_size(3); 
      rowKernelSize = ksize_[1];
      colKernelSize = ksize_[2];
      newNumRows = rows - rowKernelSize + 1;
      newNumCols = cols - colKernelSize + 1;
    }
    else if (data_format_ == FORMAT_NCHW){
      rows = tensor_in.shape().dim_size(2);
      cols = tensor_in.shape().dim_size(3);
      channels = tensor_in.shape().dim_size(1); 
      rowKernelSize = ksize_[2];
      colKernelSize = ksize_[3];
      newNumRows = rows - rowKernelSize + 1;
      newNumCols = cols - colKernelSize + 1;
    }
    else
      VLOG(1) << "unrecognized format, but didn't get caught by error checking";
    

    VLOG(1) << ksize_[0] << " " << ksize_[1]<< " " << ksize_[2] << " " << ksize_[3] << " " << channels;
    TensorShape outputshape;
    if (data_format_ == FORMAT_NHWC){
      outputshape = TensorShape{batch_size, newNumRows, newNumCols, channels};
    }
    else{
     outputshape = TensorShape{batch_size, channels, newNumRows, newNumCols};
    }


    OP_REQUIRES_OK(context, context->allocate_output(
                                0, outputshape, &output));

    VLOG(1) << "before or after?";
    auto outmatrix= output->tensor<float, 4>();
     
     for (int b = 0; b < batch_size; b++){
       for(int r = 0; r < newNumRows; r+=stride_[1]){
         for(int c = 0; c < newNumCols; c+=stride_[2]){
           for(int ch = 0; ch < channels; ch++){
             if (data_format_ == FORMAT_NHWC){
              outmatrix(b, r, c, ch) = calculate_sum_and_average(tensor_in, b, r, c, ch, rowKernelSize, colKernelSize, true);
             }
             else
              outmatrix(b, ch, r, c) = calculate_sum_and_average(tensor_in, b, r, c, ch, rowKernelSize, colKernelSize, false);
           }
         }
       }
     }

     //context->set_output(0, context->input(0));
    //SpatialAvgPool<Device, T>(context, output, tensor_in, params, padding_);
  }
  static float calculate_sum_and_average(const Tensor& tensor_in,
                                int b, int r, int c, int ch, int rowKernelSize, int colKernelSize, bool channelsLast){
      auto t = tensor_in.tensor<float, 4>();
      float sum = 0;
      float total = rowKernelSize * colKernelSize;

      for(int r_offset = 0; r_offset < rowKernelSize; r_offset++){
        for (int c_offset = 0; c_offset < colKernelSize; c_offset++){
          if (channelsLast)
           sum += t(b, r + r_offset, c + c_offset, ch);
          else{
           sum += t(b, ch, r + r_offset, c + c_offset);
          }
        }
      }
      return sum/total;
    }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

typedef AvgPoolingOp<CPUDevice, float> AvgPoolCPUFloat;

NEURON_REGISTER_KERNEL_BUILDER("AvgPool", DEVICE_NEURON, AvgPoolCPUFloat);


} //namespace neuron
}  // namespace tensorflow
