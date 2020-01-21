/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#include "tensor_util.h"

namespace tensorflow {
namespace neuron {

#define SWITCH_CASE_DTYPE(MACRO, dtype) {   \
    switch (dtype) {                        \
        MACRO(float, DT_FLOAT);             \
        MACRO(double, DT_DOUBLE);           \
        MACRO(int32, DT_INT32);             \
        MACRO(uint32, DT_UINT32);           \
        MACRO(uint16, DT_UINT16);           \
        MACRO(uint8, DT_UINT8);             \
        MACRO(int16, DT_INT16);             \
        MACRO(int8, DT_INT8);               \
        MACRO(complex64, DT_COMPLEX64);     \
        MACRO(complex128, DT_COMPLEX128);   \
        MACRO(int64, DT_INT64);             \
        MACRO(uint64, DT_UINT64);           \
        MACRO(bool, DT_BOOL);               \
        MACRO(qint8, DT_QINT8);             \
        MACRO(quint8, DT_QUINT8);           \
        MACRO(qint16, DT_QINT16);           \
        MACRO(quint16, DT_QUINT16);         \
        MACRO(qint32, DT_QINT32);           \
        MACRO(bfloat16, DT_BFLOAT16);       \
        MACRO(Eigen::half, DT_HALF);        \
    case (DT_STRING):   /* fall through */  \
    case (DT_RESOURCE): /* fall through */  \
    case (DT_VARIANT):  /* fall through */  \
    default:                                \
        return errors::InvalidArgument(     \
            "data type", (dtype),           \
            " is unsupported");             \
    }                                       \
}

Status tensor_memcpy(Tensor *tensor, StringPiece &source, int64 memcpy_size) {
    int64 dst_size = tensor->tensor_data().size();
    if (memcpy_size < 0) {
        memcpy_size = dst_size;
    }
    if (memcpy_size > (int64)source.size() || memcpy_size > dst_size) {
        return errors::OutOfRange(
            "unexpected tensor size in tensor_memcpy, source size: ",
            source.size(), ", target size: ", tensor->tensor_data().size());
    }
    #define CASE_MEMCPY_TENSOR(TTYPE, TF_DataType) {            \
        case (TF_DataType):                                     \
            std::memcpy(tensor->unaligned_flat<TTYPE>().data(), \
                        source.data(), memcpy_size);            \
        break;                                                  \
    }
    SWITCH_CASE_DTYPE(CASE_MEMCPY_TENSOR, tensor->dtype());
    return Status::OK();
}

Status tensor_memset(Tensor *tensor, int ch) {
    #define CASE_MEMSET_TENSOR(TTYPE, TF_DataType) {            \
        case (TF_DataType):                                     \
            std::memset(tensor->unaligned_flat<TTYPE>().data(), \
                        ch, tensor->tensor_data().size());      \
        break;                                                  \
    }
    SWITCH_CASE_DTYPE(CASE_MEMSET_TENSOR, tensor->dtype());
    return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow
