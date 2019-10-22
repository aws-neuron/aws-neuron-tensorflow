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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

REGISTER_OP("NeuronOp")
    .SetIsStateful()
    .Attr("graph_def: string")
    .Attr("input_names: list(string)")
    .Attr("input_dtypes: list(type) >= 0")
    .Attr("input_shapes: list(shape)")
    .Attr("output_names: list(string)")
    .Attr("output_dtypes: list(type) >= 0")
    .Attr("output_shapes: list(shape)")
    .Attr("executable: string = ''")
    .Attr("input_batch_axis: list(int) = []")
    .Attr("output_batch_axis: list(int) = []")
    .Input("input_tensors: input_dtypes")
    .Output("output_tensors: output_dtypes");
} // namespace tensorflow
