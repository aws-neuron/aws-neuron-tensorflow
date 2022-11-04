# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from tensorflow_neuron.python import custom_call as cstmcl
from tensorflow_neuron.python.utils import decorate_methods_with


@decorate_methods_with(staticmethod)
class BatchHloInstructionPool:
    """
    Return value format:
        1. None: this op does not allow batch propagation nor define a batch dimension.
        2. tuple (list[int], int): (batch_propagable_ids, seed_batch_axis)
    Requirements:
        1. All tensors in batch_propagable_ids should have the same shape[0].
        2. seed_batch_axis should have value if the instruction has a well-defined batch dimension
           (only a small number of instruction types do), and be None otherwise.
    """

    def abs(op):
        return [op.id, *op.operand_ids], None

    def add(op):
        return [op.id, *op.operand_ids], None

    def _and(op):
        return [op.id, *op.operand_ids], None

    def batch_norm_inference(op):
        if op.inst.feature_index != 0:
            return [op.id, op.operand_ids[0]], 0
        else:
            return None

    def batch_norm_training(op):
        if op.inst.feature_index != 0:
            return [op.id, op.operand_ids[0]], 0
        else:
            return None

    def broadcast(op):
        if 0 in op.inst.dimensions:
            return [op.id, *op.operand_ids], None
        else:
            return None

    def clamp(op):
        return [op.id, *op.operand_ids], None

    def compare(op):
        return [op.id, *op.operand_ids], None

    def concatenate(op):
        if op.inst.dimensions[0] != 0:
            return [op.id, *op.operand_ids], None
        else:
            return None

    def constant(op):
        return None

    def convert(op):
        return [op.id, *op.operand_ids], None

    def convolution(op):
        conv_dim_nums = op.inst.convolution_dimension_numbers
        if conv_dim_nums.input_batch_dimension == conv_dim_nums.output_batch_dimension == 0:
            return [op.id, op.operand_ids[0]], 0
        else:
            return None

    def custom_call(op):
        target = op.inst.custom_call_target
        if target in {cstmcl.targetAwsNeuronErf, cstmcl.targetAwsNeuronSoftplus}:
            return [op.id, *op.operand_ids], None
        else:
            return None

    def divide(op):
        return [op.id, *op.operand_ids], None

    def dot(op):
        dot_dim_nums = op.inst.dot_dimension_numbers
        lhs_has_batch = 0 in dot_dim_nums.lhs_batch_dimensions
        rhs_has_batch = 0 in dot_dim_nums.rhs_batch_dimensions
        batch_axis = 0 if lhs_has_batch and rhs_has_batch else None
        if batch_axis is None and len(op.input_shapes[0]) == 2 and len(op.input_shapes[1]) == 2:
            # regular MatMul
            batch_axis = 0
        batch_propagable_ids = [op.id]
        if 0 not in dot_dim_nums.lhs_contracting_dimensions:
            batch_propagable_ids.append(op.operand_ids[0])
        if rhs_has_batch:
            batch_propagable_ids.append(op.operand_ids[1])
        if len(batch_propagable_ids) == 1:
            batch_propagable_ids = None
            batch_axis = None
        return batch_propagable_ids, batch_axis

    def exponential(op):
        return [op.id, *op.operand_ids], None

    def exponential_minus_one(op):
        return [op.id, *op.operand_ids], None

    def get_dimension_size(op):
        return None

    def get_tuple_element(op):
        return None

    def log(op):
        return [op.id, *op.operand_ids], None

    def log_plus_one(op):
        return [op.id, *op.operand_ids], None

    def logistic(op):
        return [op.id, *op.operand_ids], None

    def maximum(op):
        return [op.id, *op.operand_ids], None

    def minimum(op):
        return [op.id, *op.operand_ids], None

    def multiply(op):
        return [op.id, *op.operand_ids], None

    def negate(op):
        return [op.id, *op.operand_ids], None

    def _not(op):
        return [op.id, *op.operand_ids], None

    def pad(op):
        return [op.id, op.operand_ids[0]], None

    def parameter(op):
        return None

    def power(op):
        return [op.id, op.operand_ids[0]], None

    def reduce(op):
        if 0 not in op.inst.dimensions:
            return [op.id, op.operand_ids[0]], None
        else:
            return None

    def reduce_window(op):
        batch_window_dim = op.inst.window.dimensions[0]
        property_names = ['size', 'stride', 'window_dilation', 'base_dilation']
        if all(getattr(batch_window_dim, name) == 1 for name in property_names):
            return [op.id, op.operand_ids[0]], 0
        else:
            return None

    def reshape(op):
        input0_shape = op.input_shapes[0]
        if input0_shape and op.shape and input0_shape[0] == op.shape[0]:
            return [op.id, *op.operand_ids], None
        else:
            return None

    def reverse(op):
        if 0 not in op.inst.dimensions:
            return [op.id, *op.operand_ids], None
        else:
            return None

    def rsqrt(op):
        return [op.id, *op.operand_ids], None

    def select(op):
        return [op.id, *op.operand_ids], None

    def slice(op):
        batch_dim = op.inst.slice_dimensions[0]
        if batch_dim.start == 0 and batch_dim.limit == op.shape[0] and batch_dim.stride == 1:
            return [op.id, *op.operand_ids], None
        else:
            return None

    def sqrt(op):
        return [op.id, *op.operand_ids], None

    def subtract(op):
        return [op.id, *op.operand_ids], None

    def tanh(op):
        return [op.id, *op.operand_ids], None

    def transpose(op):
        if op.inst.dimensions[0] == 0:
            return [op.id, *op.operand_ids], None
        else:
            return None

    def tuple(op):
        return None
