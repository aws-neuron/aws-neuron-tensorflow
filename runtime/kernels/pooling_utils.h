#ifndef POOLING_UTILS_H
#define POOLING_UTILS_H

namespace tensorflow{

static void init_basic_info(Tensor tensor_in, std::vector<int32> ksize_,
                     std::vector<int32> stride_, bool isNHWC, int& rows,
                     int& cols, int& channels, int& rowKernelSize,
                     int& colKernelSize, int& rowStrideLen, int& colStrideLen) {
  if (isNHWC) {
    rows = tensor_in.shape().dim_size(1);
    cols = tensor_in.shape().dim_size(2);
    channels = tensor_in.shape().dim_size(3);
    rowKernelSize = ksize_[1];
    colKernelSize = ksize_[2];
    rowStrideLen = stride_[1];
    colStrideLen = stride_[2];
  } else {
    rows = tensor_in.shape().dim_size(2);
    cols = tensor_in.shape().dim_size(3);
    channels = tensor_in.shape().dim_size(1);
    rowKernelSize = ksize_[2];
    colKernelSize = ksize_[3];
    rowStrideLen = stride_[2];
    colStrideLen = stride_[3];
  }
}

static void valid_padding_new_num_rows_and_cols(int rows, int cols, int rowKernelSize,
                                         int colKernelSize, int rowStrideLen,
                                         int colStrideLen, int& newNumRows,
                                         int& newNumCols) {
  // now we must calculate the actual newNumRows/Cols if striding is in play
  newNumRows = rows - rowKernelSize + 1;
  newNumCols = cols - colKernelSize + 1;
  // ceiling(newNumRows/rowStrideLen)
  newNumRows = (newNumRows + rowStrideLen - 1) / rowStrideLen;
  // ceiling(newNumRows/rowStrideLen)
  newNumCols = (newNumCols + colStrideLen - 1) / colStrideLen;
}

static void same_padding_new_num_rows_and_cols(Tensor tensor_in, Tensor& paddedTensor,
                                        OpKernelContext* context, int rows,
                                        int cols, int batch_size, int channels,
                                        int rowKernelSize, int colKernelSize,
                                        int rowStrideLen, int colStrideLen,
                                        int& newNumRows, int& newNumCols, float paddingValue,
                                        bool isNHWC) {
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

  TensorShape newinputshape;
  if (isNHWC) {
    newinputshape =
        TensorShape{batch_size, paddedRowLen, paddedColLen, channels};
  } else {
    newinputshape =
        TensorShape{batch_size, channels, paddedRowLen, paddedColLen};
  }

  OP_REQUIRES_OK(
      context, context->allocate_temp(DT_FLOAT, newinputshape, &paddedTensor));
  VLOG(1) << "Successfully allocated the paddedTensor for same paddding.";

  auto newmatrix = paddedTensor.tensor<float, 4>();
  auto t = tensor_in.tensor<float, 4>();

  if (isNHWC) {
    VLOG(1) << "Channels last padded tensor being created";
    // create the padded tensor surrounded with proper padding
    for (int b = 0; b < batch_size; b++) {
      for (int r = 0; r < paddedRowLen; r++) {
        for (int c = 0; c < paddedColLen; c++) {
          for (int ch = 0; ch < channels; ch++) {
            if ((r < row_offset_up || c < col_offset_left) ||
                (r >= paddedRowLen - row_offset_down ||
                 c >= paddedColLen - col_offset_right)) {
              newmatrix(b, r, c, ch) = paddingValue;
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
              newmatrix(b, ch, r, c) = paddingValue;
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

static void do_pooling(Tensor tensor_in, Tensor* output, int batch_size,
                int newNumRows, int rowStrideLen, int rowKernelSize,
                int newNumCols, int colStrideLen, int colKernelSize,
                int channels,
                std::function<float(Tensor, int, int, int, int, int, int, bool)>
                    pooling_func,
                bool isNHWC) {
  auto outmatrix = output->tensor<float, 4>();
  VLOG(1) << "Pooling Step";
  if (isNHWC) {
    VLOG(1) << "Using channels last format";
    for (int b = 0; b < batch_size; b++) {
      for (int r = 0, r1 = 0; r1 < newNumRows; r += rowStrideLen, r1++) {
        for (int c = 0, c1 = 0; c1 < newNumCols; c += colStrideLen, c1++) {
          for (int ch = 0; ch < channels; ch++) {
            outmatrix(b, r1, c1, ch) = pooling_func(
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
          for (int c = 0, c1 = 0; c1 < newNumCols; c += colStrideLen, c1++) {
            outmatrix(b, ch, r1, c1) = pooling_func(
                tensor_in, b, r, c, ch, rowKernelSize, colKernelSize, false);
          }
        }
      }
    }
  }
}

static void special_case(
    Tensor tensor_in, Tensor* output, int rows, int cols, int batch_size,
    std::function<float(Tensor, int, int, int, int, bool)> special_case_func,
    int channels, bool isNHWC) {
  auto outmatrix = output->tensor<float, 4>();
  for (int b = 0; b < batch_size; b++) {
    for (int ch = 0; ch < channels; ch++) {
      // less efficient here because I think we can get away with it as it
      // is only 1x1 image
      if (isNHWC) {
        outmatrix(b, 0, 0, ch) =
            special_case_func(tensor_in, b, ch, rows, cols, true);
      } else {
        outmatrix(b, ch, 0, 0) =
            special_case_func(tensor_in, b, ch, rows, cols, false);
      }
    }
  }
}
}  // namespace tensorflow

#endif
