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
import logging
import numpy as np
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.service import hlo_pb2
from tensorflow.python.framework import dtypes
from tensorflow.neuron.python.utils import decorate_methods_with
from tensorflow.neuron.python.hlo.instruction import BatchHloInstructionPool


logger = logging.getLogger('aws_neuron_hlo_optimizer')


# Constants for buffer/cache utilization analysis
_DEFAULT_IO_QUEUE_DEPTH = 2
_DEFAULT_IO_BUFFER_NUM_BYTES = 128 * 1024 * 1024
_DEFAULT_CACHE_CAPACITY = 12 * 1024 * 1024


class HloOptimizer:

    tuple_output_opcodes = {'batch-norm-training'}

    def __init__(self, hlo_module):
        self.hlo_module = hlo_module
        self.broadcasted_from = {}
        self.builder = None
        self.id_to_computation = {cpt.id: cpt for cpt in hlo_module.computations}
        output_tuple = [inst for inst in self.entry_instructions if inst.opcode == 'tuple'][-1]
        self.output_tuple_op = HloOp(output_tuple)

        # get rid of some no-op instructions
        self.output_ids = set()
        self.fold_no_op_instructions()

        # determine inputs/outputs
        self.output_ids = set(self.output_tuple_op.operand_ids)
        self.parameter_name_to_id = {}
        for inst in self.entry_instructions:
            if inst.opcode == 'parameter':
                param_name, _ = inst.name.split('.')
                self.parameter_name_to_id[param_name] = inst.id
        self.assign_io_names()
        self.inputs, self.outputs = self.get_io_tensors()
        self.input_shuffles = None
        self.hlo_ops_with_batch_axis = None
        self.original_batch_size = None
        self.skip_shm_ids = set()

    def get_snapshot(self):
        hlo_snapshot = hlo_pb2.HloSnapshot()
        hlo_snapshot.hlo.hlo_module.CopyFrom(self.hlo_module)
        return hlo_snapshot

    def get_io_ops(self):
        entry_ops = [HloOp(inst) for inst in self.entry_instructions]
        id_to_op = {op.id: op for op in entry_ops}
        parameter_names = self.hlo_module.host_program_shape.parameter_names
        input_ids = [self.parameter_name_to_id[name] for name in parameter_names]
        output_ids = self.output_tuple_op.operand_ids
        input_ops = [id_to_op[iid] for iid in input_ids]
        output_ops = [id_to_op[oid] for oid in output_ids]
        return input_ops, output_ops

    def assign_io_names(self):
        input_ops, output_ops = self.get_io_ops()
        for idx, op in enumerate(input_ops):
            op.neff_input_name = 'input{}'.format(idx)
        neff_output_names = ['output{}'.format(idx) for idx, _ in enumerate(output_ops)]
        self.output_tuple_op.neff_output_names = neff_output_names

    def get_io_tensors(self):
        input_ops, output_ops = self.get_io_ops()
        inputs = [HloTensor(op) for op in input_ops]
        outputs = [HloTensor(op) for op in output_ops]
        for ts, op in zip(inputs, input_ops):
            ts.name = op.neff_input_name
        for ts, name in zip(outputs, self.output_tuple_op.neff_output_names):
            ts.name = name
        return inputs, outputs

    @property
    def entry_instructions(self):
        entry_computation = self.id_to_computation[self.hlo_module.entry_computation_id]
        assert entry_computation.program_shape == self.hlo_module.host_program_shape, 'program_shape mismatch'
        return entry_computation.instructions

    def constant_folding(self):
        value_map = {}
        for inst in self.entry_instructions:
            if inst.opcode == 'constant':
                value_map[inst.id] = HloOp(inst).literal_value
            elif all(oid in value_map for oid in inst.operand_ids):
                op = HloOp(inst)
                func = getattr(ConstantFoldingInstructionPool, op.legal_opcode, None)
                if func is not None and inst.operand_ids:
                    input_values = [value_map[oid] for oid in inst.operand_ids]
                    output_value = func(op, *input_values)
                    inst.opcode = 'constant'
                    inst.operand_ids[:] = []
                    inst.literal.shape.CopyFrom(inst.shape)
                    attr_name = HloOp.xla_dtype_to_literal_attr_name[inst.shape.element_type]
                    literals = getattr(inst.literal, attr_name)
                    if isinstance(literals, bytes):
                        setattr(inst.literal, attr_name, output_value.tobytes())
                    else:
                        literals[:] = output_value.ravel()
                    value_map[inst.id] = output_value

    def dead_code_elimination(self):
        id_to_inst = {inst.id: inst for inst in self.entry_instructions}
        visited_ids = set()
        io_inst_types = {'parameter', 'tuple'}
        stack = [inst.id for inst in self.entry_instructions if inst.opcode in io_inst_types]
        while stack:
            iid = stack.pop()
            if iid in visited_ids:
                continue
            visited_ids.add(iid)
            inst = id_to_inst[iid]
            stack.extend(inst.operand_ids)
        visited_instructions = [inst for inst in self.entry_instructions if inst.id in visited_ids]
        while self.entry_instructions:
            self.entry_instructions.pop()
        self.entry_instructions.extend(visited_instructions)

    def flip_broadcast_gather(self):
        hlo_op_list = [HloOp(inst) for inst in self.entry_instructions]
        id_to_op = {op.id: op for op in hlo_op_list}
        moving_broadcast_ids = set()
        gather_to_broadcast = {}
        for op in hlo_op_list:
            if op.opcode != 'gather' or op.id in self.output_ids:
                continue
            operand_id, start_indices_id = op.operand_ids
            operand_op = id_to_op[operand_id]
            if operand_op.opcode != 'constant':
                continue
            broadcast_op = id_to_op[start_indices_id]
            if broadcast_op.opcode != 'broadcast' or broadcast_op.id in self.output_ids:
                continue
            start_indices_id, = broadcast_op.operand_ids
            start_indices_op = id_to_op[start_indices_id]
            gdns = op.inst.gather_dimension_numbers

            # only support scalar -> broadcast -> gather for now
            if not _is_simple_gather(op) or start_indices_op.shape != []:
                continue

            # generate new broadcast instruction
            broadcast_op.inst.operand_ids[0] = op.id
            broadcast_op.inst.shape.dimensions[:] = op.shape
            broadcast_op.inst.shape.element_type = op.inst.shape.element_type
            broadcast_op.inst.dimensions[:] = gdns.offset_dims
            minor_to_major = broadcast_op.inst.shape.layout.minor_to_major
            minor_to_major[:] = reversed([idx for idx, _ in enumerate(op.shape)])
            broadcast_op.inst.shape.is_dynamic_dimension[:] = [False for _ in op.shape]

            # generate new gather instruction
            op.inst.operand_ids[1] = start_indices_op.id
            op.inst.shape.dimensions[:] = [op.shape[dim] for dim in gdns.offset_dims]
            is_dynamic_dimension = op.inst.shape.is_dynamic_dimension
            is_dynamic_dimension[:] = [False for _ in gdns.offset_dims]
            minor_to_major = op.inst.shape.layout.minor_to_major
            minor_to_major[:] = reversed([idx for idx, _ in enumerate(gdns.offset_dims)])
            gdns.index_vector_dim = 0
            gdns.offset_dims[:] = [0]
            gdns.collapsed_slice_dims[:] = [0]
            gdns.start_index_map[:] = [0]

            # prepare for code motion
            moving_broadcast_ids.add(broadcast_op.id)
            gather_to_broadcast[op.id] = broadcast_op.id
        if gather_to_broadcast:
            reordered_instructions = []
            for inst in self.entry_instructions:
                if inst.id not in moving_broadcast_ids:
                    reordered_instructions.append(inst)
                    inst.operand_ids[:] = [gather_to_broadcast.get(i, i) for i in inst.operand_ids]
                if inst.id in gather_to_broadcast:
                    broadcast_id = gather_to_broadcast[inst.id]
                    reordered_instructions.append(id_to_op[broadcast_id].inst)
            while self.entry_instructions:
                self.entry_instructions.pop()
            self.entry_instructions.extend(reordered_instructions)

    def fold_no_op_instructions(self):
        # remove instructions that are equivalent to No-Op
        id_to_inst = {inst.id: inst for inst in self.entry_instructions}

        def reshape_into_original_shape(inst):
            input_inst = id_to_inst[inst.operand_ids[0]]
            return input_inst.shape.dimensions == inst.shape.dimensions

        no_op_checker_map = {
            'reshape': reshape_into_original_shape,
            'transpose': lambda inst: inst.dimensions == sorted(inst.dimensions),
        }
        id_map = {}
        for inst in self.entry_instructions:
            if inst.id not in self.output_ids and inst.operand_ids:
                if no_op_checker_map.get(inst.opcode, lambda inst: False)(inst):
                    input_inst = id_to_inst[inst.operand_ids[0]]
                    id_map[inst.id] = id_map.get(input_inst.id, input_inst.id)
        for inst in self.entry_instructions:
            inst.operand_ids[:] = [id_map.get(input_id, input_id) for input_id in inst.operand_ids]

    def batchify_reshape_dot_reshape(self):
        # rewrite (batch) -> reshape -> dot -> reshape -> (batch) with dot to enable batch analyzer
        hlo_op_list = [HloOp(inst) for inst in self.entry_instructions]
        id_to_op = {op.id: op for op in hlo_op_list}
        for op in hlo_op_list:
            op.consumer_ids = []
        for op in hlo_op_list:
            for oid in op.operand_ids:
                oop = id_to_op[oid]
                if op.id not in oop.consumer_ids:
                    oop.consumer_ids.append(op.id)
        reshape_id_to_dot_reshape_list = {}
        for op in hlo_op_list:
            if op.opcode == 'reshape' and op.id not in self.output_ids:
                dot_op = id_to_op[op.operand_ids[0]]
                if dot_op.opcode != 'dot' or dot_op.id in self.output_ids:
                    continue
                if len(dot_op.consumer_ids) != 1:
                    continue
                lhs_id, rhs_id = dot_op.operand_ids
                reshape_op = id_to_op[lhs_id]
                rhs_op = id_to_op[rhs_id]
                ddn = dot_op.inst.dot_dimension_numbers
                if not (len(reshape_op.shape) == 2 and ddn.lhs_contracting_dimensions == [1]):
                    continue
                if not (len(rhs_op.shape) == 2 and ddn.rhs_contracting_dimensions == [0]):
                    continue
                if reshape_op.opcode != 'reshape' or reshape_op.id in self.output_ids:
                    continue
                if np.prod(reshape_op.shape[:-1]) != np.prod(op.shape[:-1]):
                    continue
                input_op = id_to_op[reshape_op.operand_ids[0]]
                if input_op.shape[0] != op.shape[0]:  # need same input/output batch sizes
                    continue
                if reshape_op.id not in reshape_id_to_dot_reshape_list:
                    reshape_id_to_dot_reshape_list[reshape_op.id] = []
                reshape_id_to_dot_reshape_list[reshape_op.id].append([dot_op, op])
        for reshape_id, dot_reshape_list in reshape_id_to_dot_reshape_list.items():
            input_reshape_op = id_to_op[reshape_id]
            if len(input_reshape_op.consumer_ids) != len(dot_reshape_list) or not dot_reshape_list:
                continue
            _, output_reshape_op = dot_reshape_list[0]
            if any(op.shape[:-1] != output_reshape_op.shape[:-1] for _, op in dot_reshape_list):
                continue
            input_shape = input_reshape_op.inst.shape
            input_shape.dimensions[:-1] = output_reshape_op.shape[:-1]
            input_shape.is_dynamic_dimension[:] = [False for _ in input_reshape_op.shape]
            input_shape.layout.minor_to_major[:] = reversed(range(len(input_reshape_op.shape)))
            for dot_op, output_reshape_op in dot_reshape_list:
                output_nd = len(output_reshape_op.shape)
                dot_shape = dot_op.inst.shape
                dot_shape.dimensions[:] = output_reshape_op.shape
                dot_shape.is_dynamic_dimension[:] = [False for _ in output_reshape_op.shape]
                dot_shape.layout.minor_to_major[:] = reversed(range(output_nd))
                dot_op.inst.dot_dimension_numbers.lhs_contracting_dimensions[:] = [output_nd - 1]

    def maybe_enable_dynamic_batch_size(self):
        hlo_op_list = [HloOp(inst) for inst in self.entry_instructions]
        id_to_op = {op.id: op for op in hlo_op_list}
        for op in hlo_op_list:
            op.input_shapes = [id_to_op[oid].shape for oid in op.operand_ids]
            op.consumer_ids = []
            op.batch_propagable_ids = None
            op.batch_propagable_neighbor_ids = []
            op.batch_axis = None
            op.batch_size_multiplier = 1
            op.is_batch_axis_seed = False
            op.batch_axis_source_ids = []
        for op in hlo_op_list:
            for oid in op.operand_ids:
                oop = id_to_op[oid]
                if op.id not in oop.consumer_ids:
                    oop.consumer_ids.append(op.id)
        equivalent_source_ids = {}
        for op in hlo_op_list:
            func = getattr(BatchHloInstructionPool, op.legal_opcode, None)
            if func is not None:
                batch_definition = func(op)
                if batch_definition is not None:
                    op.batch_propagable_ids, op.batch_axis = batch_definition
                    if op.batch_axis is not None:
                        op.is_batch_axis_seed = True
                        op.batch_axis_source_ids.append(op.id)
                        equivalent_source_ids[op.id] = {op.id}
        for op in hlo_op_list:
            if op.opcode == 'get-tuple-element':
                if id_to_op[op.operand_ids[0]].opcode == 'batch-norm-training':
                    if op.inst.tuple_index == 0:
                        op.batch_propagable_ids = [op.id, *op.operand_ids]

        def match_rtrcrt(op):
            if op.opcode != 'transpose':
                return None
            out_r_op = id_to_op[op.operand_ids[0]]
            if out_r_op.opcode != 'reshape':
                return None
            convolution_op = id_to_op[out_r_op.operand_ids[0]]
            if convolution_op.opcode != 'convolution':
                return None
            conv_out_non_batch_shape = convolution_op.shape[1:]
            if out_r_op.shape[-len(conv_out_non_batch_shape):] != conv_out_non_batch_shape:
                return None
            in_r_op = id_to_op[convolution_op.operand_ids[0]]
            if in_r_op.opcode != 'reshape':
                return None
            in_t_op = id_to_op[in_r_op.operand_ids[0]]
            if in_t_op.opcode != 'transpose':
                return None
            conv_in_non_batch_shape = in_r_op.shape[1:]
            if in_t_op.shape[-len(conv_in_non_batch_shape):] != conv_in_non_batch_shape:
                return None
            if len(in_t_op.inst.dimensions) != len(op.inst.dimensions):
                return None
            batch_dim = 0
            transposed_batch_dim = list(in_t_op.inst.dimensions).index(batch_dim)
            if op.inst.dimensions[batch_dim] != transposed_batch_dim:
                return None
            r_op = id_to_op[in_t_op.operand_ids[0]]
            if r_op.opcode != 'reshape':
                return None
            if r_op.batch_propagable_ids is None:
                return None
            return r_op, in_t_op, in_r_op, convolution_op, out_r_op, op

        # reshape -> transpose -> reshape -> convolution -> reshape -> transpose pattern
        for op in hlo_op_list:
            rtrcrt = match_rtrcrt(op)
            if rtrcrt is not None and op.batch_propagable_ids is None:
                r_op, *_ = rtrcrt
                op.batch_propagable_ids = [op.id, r_op.id]
                r_op.batch_propagable_ids = [r_op.id, r_op.operand_ids[0], op.id]

        # setup neighboring nodes
        for op in hlo_op_list:
            if op.batch_propagable_ids is not None:
                op.batch_propagable_neighbor_ids.extend(op.batch_propagable_ids)
                propagable_ops = [id_to_op[bpid] for bpid in op.batch_propagable_ids]
                for pop in propagable_ops:
                    if op.id not in pop.batch_propagable_neighbor_ids:
                        pop.batch_propagable_neighbor_ids.append(op.id)

        # propagate batch dimension information by traversing the graph
        source_op_candidates = [op for op in hlo_op_list if op.is_batch_axis_seed]
        source_id_to_root_id = {}
        while source_op_candidates:
            source_op = source_op_candidates.pop()
            visited_ids = set()
            stack = [source_op]
            while stack:
                current_op = stack.pop()
                if current_op.id in visited_ids:
                    continue
                visited_ids.add(current_op.id)
                if current_op.batch_propagable_ids is not None:
                    propagable_ops = [id_to_op[bpid] for bpid in current_op.batch_propagable_ids]
                    for op in propagable_ops:
                        if op.is_batch_axis_seed:
                            equivalent_source_ids[source_op.id].update(op.batch_axis_source_ids)
                        else:
                            op.batch_axis = source_op.batch_axis
                        if source_op.id not in op.batch_axis_source_ids:
                            op.batch_axis_source_ids.append(source_op.id)
                if current_op.legal_opcode != 'tuple':
                    # tuple should be a sink node that doesn't link different outputs together
                    stack.extend(id_to_op[oid] for oid in current_op.batch_propagable_neighbor_ids)

            # remove transitively equivalent source candidates
            visited_ids = set()
            stack = list(equivalent_source_ids.keys())
            while stack:
                current_id = stack.pop()
                if current_id in visited_ids:
                    continue
                if current_id not in source_id_to_root_id:
                    source_id_to_root_id[current_id] = current_id
                root_id = source_id_to_root_id[current_id]
                for equivalent_id in equivalent_source_ids[current_id]:
                    source_id_to_root_id[equivalent_id] = root_id
            root_id = source_id_to_root_id[source_op.id]
            source_op_candidates = []
            for op in source_op_candidates:
                if source_id_to_root_id[op.batch_axis_source_id] != root_id:
                    source_op_candidates.append(op)

        # enable dynamic batch size only if there is a single root source of batch dimension
        parameter_ops = []
        for name in self.hlo_module.host_program_shape.parameter_names:
            parameter_id = self.parameter_name_to_id[name]
            parameter_ops.append(id_to_op[parameter_id])
        output_ops = [id_to_op[oid] for oid in self.output_tuple_op.operand_ids]
        io_ops = parameter_ops + output_ops
        io_ops_with_batch_axis = [op for op in io_ops if op.batch_axis is not None]
        if not io_ops_with_batch_axis:
            return
        io_ops_root_ids = []
        for op in io_ops_with_batch_axis:
            root_ids = {source_id_to_root_id.get(sid, sid) for sid in op.batch_axis_source_ids}
            io_ops_root_ids.append(root_ids)
        if all(len(root_ids) == 1 for root_ids in io_ops_root_ids):
            io_ops_source_id = io_ops_with_batch_axis[-1].batch_axis_source_ids[0]
            io_ops_root_id = source_id_to_root_id.get(io_ops_source_id, io_ops_source_id)
            reject = False
            for op in hlo_op_list:
                root_ids = {source_id_to_root_id.get(sid, sid) for sid in op.batch_axis_source_ids}
                if io_ops_root_id in root_ids and len(root_ids) != 1:
                    # source conflict on instructions affecting IO instructions; reject
                    op.batch_axis = None
                    reject = True
            if reject:
                return
        else:
            # source conflict on IO instructions directly; reject
            return

        # TODO: this will be unnecessary once BatchHloInstructionPool is fully populated
        if any(op.batch_axis is None for op in io_ops):
            return

        # reshape -> transpose -> reshape -> convolution -> reshape -> transpose pattern again
        for op in hlo_op_list:
            rtrcrt = match_rtrcrt(op)
            if rtrcrt is not None:
                if [o.batch_axis for o in rtrcrt] == [0, None, None, 0, None, 0]:
                    r_op, in_t_op, in_r_op, convolution_op, out_r_op, out_t_op = rtrcrt
                    in_t_op.batch_axis = list(in_t_op.inst.dimensions).index(0)
                    conv_in_non_batch_shape = in_r_op.shape[1:]
                    in_t_op_multiplier_shape = in_t_op.shape[:-len(conv_in_non_batch_shape)]
                    in_t_op_multiplier_shape.pop(in_t_op.batch_axis)
                    batch_size_multiplier = int(np.prod(in_t_op_multiplier_shape))
                    in_r_op.batch_axis = 0
                    in_r_op.batch_size_multiplier = batch_size_multiplier
                    convolution_op.batch_axis = 0
                    convolution_op.batch_size_multiplier = batch_size_multiplier
                    out_r_op.batch_axis = out_t_op.inst.dimensions[out_t_op.batch_axis]

        # write input_batch_axis and output_batch_axis in runtime format
        input_batch_axis = [op.batch_axis for op in parameter_ops]
        output_batch_axis = [op.batch_axis for op in output_ops]
        _assert_same_len(self.inputs, input_batch_axis, 'inputs', 'input_batch_axis')
        _assert_same_len(self.outputs, output_batch_axis, 'outputs', 'output_batch_axis')
        for args in [self.inputs, input_batch_axis], [self.outputs, output_batch_axis]:
            for ts, axis in zip(*args):
                ts.batch_axis = axis
        self.hlo_ops_with_batch_axis = []
        for op in hlo_op_list:
            if op.batch_axis is not None:
                if op.opcode not in HloOptimizer.tuple_output_opcodes:
                    self.hlo_ops_with_batch_axis.append(op)
        self.original_batch_size = self.get_batch_size()

    def get_batch_size(self):
        if self.hlo_ops_with_batch_axis:
            op = self.hlo_ops_with_batch_axis[0]
            return op.shape[op.batch_axis]
        else:
            return None

    def rewrite_batch_size(self, batch_size, final=False):
        if self.hlo_ops_with_batch_axis is not None:
            for op in self.hlo_ops_with_batch_axis:
                op.inst.shape.dimensions[op.batch_axis] = batch_size * op.batch_size_multiplier
                if final and op.opcode == 'constant':
                    attr_name = HloOp.xla_dtype_to_literal_attr_name[op.inst.shape.element_type]
                    literals = getattr(op.inst.literal, attr_name)
                    if len(literals) % self.original_batch_size == 0:
                        len_new_literals = len(literals) // self.original_batch_size * batch_size
                        new_literals = literals[:len_new_literals]
                        if isinstance(literals, bytes):
                            setattr(op.inst.literal, attr_name, new_literals)
                        else:
                            literals[:] = new_literals

    def maybe_enable_rtr_shuffle(self):
        id_to_inst = {inst.id: inst for inst in self.entry_instructions}
        parameter_insts = [inst for inst in self.entry_instructions if inst.opcode == 'parameter']
        parameter_id_to_consumer_ids = {inst.id: [] for inst in parameter_insts}
        for inst in self.entry_instructions:
            if inst.operand_ids:
                input_inst_id = inst.operand_ids[0]
                input_inst = id_to_inst[input_inst_id]
                if input_inst.opcode == 'parameter':
                    consumer_ids = parameter_id_to_consumer_ids[input_inst_id]
                    if inst.id not in consumer_ids:
                        consumer_ids.append(inst.id)
        parameter_id_to_shuffle = {}
        for inst in parameter_insts:
            consumers = [id_to_inst[cid] for cid in parameter_id_to_consumer_ids[inst.id]]
            rewriters = [get_rtr_rewriter(cinst, id_to_inst, self.output_ids) for cinst in consumers]
            if rewriters and all(rwt is not None for rwt in rewriters):
                if not _all_arrays_equal(rwt.shuffle_indices for rwt in rewriters):
                    continue
                kernel_id_to_kernel_rewriters = {cinst.operand_ids[1]: [] for cinst in consumers}
                for rwt in rewriters:
                    container = kernel_id_to_kernel_rewriters[rwt.kernel_inst.id]
                    if rwt not in container:
                        container.append(rwt)
                kernel_rewriters_inconsistent = False
                for kernel_rewriters in kernel_id_to_kernel_rewriters.values():
                    if not _all_arrays_equal(rwt.kernel_array_prtr for rwt in kernel_rewriters):
                        kernel_rewriters_inconsistent = True
                        break
                if kernel_rewriters_inconsistent:
                    continue
                shuffle_indices = rewriters[0].rewrite_input(inst)
                for kernel_id, kernel_rewriters in kernel_id_to_kernel_rewriters.items():
                    if kernel_rewriters:
                        kernel_inst = id_to_inst[kernel_id]
                        kernel_rewriters[0].rewrite_kernel(kernel_inst)
                parameter_id_to_shuffle[inst.id] = shuffle_indices
        input_shuffles = []
        for name in self.hlo_module.host_program_shape.parameter_names:
            parameter_id = self.parameter_name_to_id[name]
            shuffle = parameter_id_to_shuffle.get(parameter_id, None)
            input_shuffles.append(shuffle)
        if any(shuffle is not None for shuffle in input_shuffles):
            self.input_shuffles = input_shuffles

        # change host program shape and entry compuation program shape as well
        self._reestablish_program_shapes()

    def estimate_cache_demand(self):
        entry_ops = [HloOp(inst) for inst in self.entry_instructions]
        id_to_op = {op.id: op for op in entry_ops}
        all_cache_demands = []
        tensorized_opcodes = {'convolution', 'dot'}
        for op in entry_ops:
            if op.opcode in tensorized_opcodes:
                cache_demand = min(HloTensor(id_to_op[oid]).num_bytes for oid in op.operand_ids)
                cache_demand += HloTensor(op).num_bytes
                all_cache_demands.append(cache_demand)
        return max(all_cache_demands) if all_cache_demands else None

    def maybe_rewrite_batch_size(self):
        batch_size = self.get_batch_size()
        if batch_size is None:
            return

        # disallow rewriting if some ops have unknown batch semantics
        hlo_op_list = [HloOp(inst) for inst in self.entry_instructions]
        id_to_op = {op.id: op for op in hlo_op_list}
        non_batch_ids = set()
        for op in hlo_op_list:
            op.input_shapes = [id_to_op[oid].shape for oid in op.operand_ids]
            op.propagable_ids = []
            op.is_non_batch = False
        batched_ids = {op.id for op in self.hlo_ops_with_batch_axis}
        for op in hlo_op_list:
            func = getattr(BatchHloInstructionPool, op.legal_opcode, None)
            if func is not None:
                batch_definition = func(op)
                if batch_definition is None:
                    batch_definition = [op.id], None
                op.propagable_ids, _ = batch_definition
                id_oids = [op.id, *op.operand_ids]
                non_batch_ids.update(i for i in id_oids if i not in op.propagable_ids)
        non_batch_ids.difference_update(batched_ids)
        for op in hlo_op_list:
            for oid in op.operand_ids:
                oop = id_to_op[oid]
                if oop.id in op.propagable_ids and op.id not in oop.propagable_ids:
                    oop.propagable_ids.append(op.id)
        for op in hlo_op_list:
            if op.id in non_batch_ids:
                op.is_non_batch = True
        visited_ids = set()
        stack = [op for op in hlo_op_list if op.is_non_batch]
        while stack:
            non_batch_op = stack.pop()
            if non_batch_op.id in visited_ids:
                continue
            visited_ids.add(non_batch_op.id)
            non_batch_op.is_non_batch = True
            stack.extend(id_to_op[pid] for pid in non_batch_op.propagable_ids)
        non_batched_ids = {op.id for op in hlo_op_list if op.is_non_batch}
        all_analyzed_ids = batched_ids.union(non_batched_ids)
        all_analyzed_ids.add(self.output_tuple_op.id)
        if len(all_analyzed_ids) != len(self.entry_instructions):
            return
        if batched_ids.intersection(non_batched_ids):
            return

        # rewrite if IO buffer demand or cache demand is too high
        inputs, outputs = self.get_io_tensors()
        total_io_num_bytes = sum(ts.num_bytes for ts in inputs + outputs)
        io_queue_num_bytes = total_io_num_bytes * _DEFAULT_IO_QUEUE_DEPTH
        io_queue_too_large = io_queue_num_bytes > _DEFAULT_IO_BUFFER_NUM_BYTES
        cache_demand_num_bytes = self.estimate_cache_demand()
        if cache_demand_num_bytes is None:
            cache_demand_too_high = False
        else:
            cache_demand_too_high = cache_demand_num_bytes > 2 * _DEFAULT_CACHE_CAPACITY
        if not (io_queue_too_large or cache_demand_too_high):
            return
        bytes_to_mbytes = lambda num_bytes: int(num_bytes / 1024 / 1024)
        if batch_size == 1:
            return
        if io_queue_too_large:
            num_mb = bytes_to_mbytes(io_queue_num_bytes)
            reason = 'batch size {} would require {} MB IO buffer'.format(batch_size, num_mb)
        elif cache_demand_too_high:
            num_mb = bytes_to_mbytes(cache_demand_num_bytes)
            reason = 'batch size {} would create {} MB cache demand'.format(batch_size, num_mb)
        logger.warning('{}; rewriting batch size to mitigate'.format(reason))
        self.rewrite_batch_size(1)

        # estimate from IO queue size
        inputs, outputs = self.get_io_tensors()
        total_io_num_bytes = sum(ts.num_bytes for ts in inputs + outputs)
        io_queue_num_bytes = total_io_num_bytes * _DEFAULT_IO_QUEUE_DEPTH
        batch_size_from_io = round(_DEFAULT_IO_BUFFER_NUM_BYTES / io_queue_num_bytes)

        # estimate from cache demand
        cache_demand = self.estimate_cache_demand()
        batch_size_from_cache = round(_DEFAULT_CACHE_CAPACITY / cache_demand)

        # choose the smaller estimate
        batch_size = min(batch_size, batch_size_from_cache, batch_size_from_io)
        batch_size = max(batch_size, 1)
        reason = 'IO queue size' if batch_size_from_io < batch_size_from_cache else 'cache demand'
        if batch_size > 64:
            batch_size = batch_size // 64 * 64
        else:
            batch_size = 2 ** (batch_size.bit_length() - 1)
        logger.info('estimated optimal batch size {} from {}'.format(batch_size, reason))
        self.rewrite_batch_size(batch_size, final=True)

        # change input_shuffles to new batch size
        def change_batch_size(shuffle):
            if shuffle is None:
                return shuffle
            else:
                return shuffle.reshape([batch_size, -1])[:batch_size].ravel()

        if self.input_shuffles is not None:
            self.input_shuffles = [change_batch_size(shuffle) for shuffle in self.input_shuffles]

        # change host program shape and entry compuation program shape as well
        self._reestablish_program_shapes()
        self._legalize_instructions()

    def engrave_io_tensors(self):
        inputs = self.inputs
        outputs = self.outputs

        # assign batch_axis and shuffles
        if self.input_shuffles is None:
            self.input_shuffles = [None for _ in inputs]
        _assert_same_len(inputs, self.input_shuffles, 'inputs', 'input_shuffles')
        batch_size = self.get_batch_size()
        if batch_size is not None:
            for tensors in inputs, outputs:
                for ts in tensors:
                    if ts.batch_axis is not None:
                        ts.shape[ts.batch_axis] = batch_size
        input_can_use_shm = [ts.id not in self.skip_shm_ids for ts in inputs]
        output_can_use_shm = [ts.id not in self.skip_shm_ids for ts in outputs]
        inputs = [HloTensor(*args) for args in zip(inputs, self.input_shuffles, input_can_use_shm)]
        outputs = [HloTensor(ts, can_use_shm=shm) for ts, shm in zip(outputs, output_can_use_shm)]
        return inputs, outputs

    def _reestablish_program_shapes(self):
        id_to_inst = {inst.id: inst for inst in self.entry_instructions}
        parameter_name_to_shape = {name: id_to_inst[pid].shape for name, pid in self.parameter_name_to_id.items()}
        entry_computation = self.id_to_computation[self.hlo_module.entry_computation_id]
        output_shapes = [id_to_inst[oid].shape for oid in self.output_tuple_op.operand_ids]
        for program_shape in self.hlo_module.host_program_shape, entry_computation.program_shape:
            for name, shape in zip(program_shape.parameter_names, program_shape.parameters):
                shape.CopyFrom(parameter_name_to_shape[name])
            for out_shape, shape in zip(program_shape.result.tuple_shapes, output_shapes):
                out_shape.CopyFrom(shape)
        # for some reason self.output_tuple_op.inst is not the original tuple instruction in HloModule
        # TODO: debug this
        output_tuple_inst = id_to_inst[self.output_tuple_op.id]
        for out_shape, shape in zip(output_tuple_inst.shape.tuple_shapes, output_shapes):
            out_shape.CopyFrom(shape)

    def _legalize_instructions(self):
        id_to_inst = {inst.id: inst for inst in self.entry_instructions}
        for inst in self.entry_instructions:
            if inst.opcode == 'slice':
                input_id, = inst.operand_ids
                input_shape = id_to_inst[input_id].shape
                for slice_dim, dim_size in zip(inst.slice_dimensions, inst.shape.dimensions):
                    slice_dim.limit = min(slice_dim.limit, dim_size)


def _assert_same_len(lhs, rhs, lhs_name, rhs_name):
    assert len(lhs) == len(rhs), '{} and {} have different length'.format(lhs_name, rhs_name)


def _is_simple_gather(op):
    gdns = op.inst.gather_dimension_numbers
    if not (len(gdns.offset_dims) == 1 and gdns.index_vector_dim == gdns.offset_dims[0]):
        return False
    # TODO: check collapsed_slice_dims
    if any(si != fsi for si, fsi in enumerate(gdns.start_index_map)):
        return False
    return True


class HloTensor:

    def __init__(self, op, shuffle=None, can_use_shm=True):
        self.name = op.name
        self.id = op.id
        self.dtype = op.dtype
        self.shape = list(op.shape)
        self.batch_axis = getattr(op, 'batch_axis', None)
        self.shuffle = shuffle
        self.can_use_shm = can_use_shm

    @property
    def num_bytes(self):
        itemsize = 2 if self.dtype == 'bfloat16' else np.dtype(self.dtype).itemsize
        return itemsize * int(np.prod(self.shape))


@decorate_methods_with(staticmethod)
class ConstantFoldingInstructionPool:

    def convert(op, value):
        return value.astype(op.dtype)

    def gather(op, value, start_indices):
        if not _is_simple_gather(op):
            raise NotImplementedError(op.inst)
        gdns = op.inst.gather_dimension_numbers
        axis = gdns.index_vector_dim - gdns.offset_dims[0]
        return value.take(start_indices, axis=axis)

    def reshape(op, value):
        return value.reshape(op.shape)

    def reverse(op, value):
        slices = [slice(None) for _ in value.shape]
        for dim in op.inst.dimensions:
            slices[dim] = slice(None, None, -1)
        return value[tuple(slices)]


class RtrRewriter:

    def __init__(self, inst, kernel_inst, input_r_shape, shuffle_indices, kernel_array_prtr):
        self.kernel_inst = kernel_inst
        self.input_r_shape = input_r_shape
        self.shuffle_indices = shuffle_indices
        self.kernel_array_prtr = kernel_array_prtr
        self.window_dims = inst.window.dimensions
        kernel_spatial_dimensions = inst.convolution_dimension_numbers.kernel_spatial_dimensions
        self.prtr_window_sizes = [kernel_array_prtr.shape[dim] for dim in kernel_spatial_dimensions]

    def rewrite_input(self, input_inst):
        input_inst.shape.dimensions[:] = self.input_r_shape
        return self.shuffle_indices

    def rewrite_kernel(self, kernel_inst):
        kernel_inst.shape.dimensions[:] = self.kernel_array_prtr.shape
        kernel_inst.literal.shape.dimensions[:] = self.kernel_array_prtr.shape
        literal_attr_name = HloOp.xla_dtype_to_literal_attr_name[kernel_inst.shape.element_type]
        literals = getattr(kernel_inst.literal, literal_attr_name)
        if isinstance(literals, bytes):
            setattr(kernel_inst.literal, literal_attr_name, self.kernel_array_prtr.tobytes())
        else:
            literals[:] = self.kernel_array_prtr.ravel()
        for dim, prtr_wsize in zip(self.window_dims, self.prtr_window_sizes):
            dim.size = prtr_wsize
            dim.stride = 1


def _all_arrays_equal(array_iter):
    array_list = list(iter(array_iter))
    for array0, array1 in zip(array_list[:-1], array_list[1:]):
        if array0.dtype != array1.dtype:
            return False
        if array0.shape != array1.shape:
            return False
        if (array0 != array1).any():
            return False
    return True


def get_rtr_rewriter(inst, id_to_inst, output_ids):
    if inst.opcode != 'convolution':
        return None
    input_id, kernel_id = inst.operand_ids
    if input_id in output_ids or kernel_id in output_ids:
        return None
    input_inst = id_to_inst[input_id]
    kernel_inst = id_to_inst[kernel_id]
    input_shape = list(input_inst.shape.dimensions)
    window_dims = inst.window.dimensions
    window_sizes = [dim.size for dim in window_dims]
    strides = [dim.stride for dim in window_dims]
    padding_lows = [dim.padding_low for dim in window_dims]
    padding_highs = [dim.padding_high for dim in window_dims]
    base_dilations = [dim.base_dilation for dim in window_dims]
    dim_nums = inst.convolution_dimension_numbers
    spatial_dimensions = dim_nums.input_spatial_dimensions
    feature_dimension = dim_nums.input_feature_dimension
    kernel_spatial_dimensions = dim_nums.kernel_spatial_dimensions
    kernel_input_feature_dimension = dim_nums.kernel_input_feature_dimension
    base_sizes = [input_shape[dim] for dim in spatial_dimensions]

    # necessary conditions for enabling rtr shuffle
    symm_strides = len(set(strides)) == 1
    all_strided = all(strd > 1 for strd in strides)
    spatial_divisible = all(bsz % strd == 0 for bsz, strd in zip(base_sizes, strides))
    stride_condition = symm_strides and all_strided and spatial_divisible
    input_is_param = input_inst.opcode == 'parameter'
    no_padding = all(pad == 0 for pad in padding_lows + padding_highs)
    few_channels = input_shape[feature_dimension] < 32
    input_condition = input_is_param and no_padding and few_channels
    window_condition = all(ws > 1 for ws in window_sizes) and kernel_inst.opcode == 'constant'
    group_condition = inst.feature_group_count == 1
    if not (stride_condition and input_condition and window_condition and group_condition):
        return None

    # pad and shuffle kernel
    kernel_array = HloOp(kernel_inst).literal_value
    kernel_spatial_paddings = [[0, 0] for _ in kernel_array.shape]
    for dim, stride in zip(kernel_spatial_dimensions, strides):
        kernel_spatial_paddings[dim][1] = kernel_array.shape[dim] % stride
    kernel_array_p = np.pad(kernel_array, kernel_spatial_paddings)
    kernel_array_prtr = _rtr_transform(kernel_array_p, kernel_spatial_dimensions,
                                       kernel_input_feature_dimension, strides)
    prtr_window_sizes = [kernel_array_prtr.shape[dim] for dim in kernel_spatial_dimensions]

    # don't enable rtr shuffle if explicit dilation is still require after padding kernel
    if not all(ws >= dil for ws, dil in zip(prtr_window_sizes, base_dilations)):
        return None

    # shuffle indices and new input shape
    num_elements = np.prod([int(size) for size in input_shape])
    indices = np.arange(num_elements).reshape(input_shape)
    indices_rtr = _rtr_transform(indices, spatial_dimensions, feature_dimension, strides)
    shuffle_indices = indices_rtr.reshape([-1])
    input_r_shape = [int(size) for size in input_shape]
    for dim, stride in zip(spatial_dimensions, strides):
        input_r_shape[dim] //= stride
        input_r_shape[feature_dimension] *= stride

    # return rewriter
    return RtrRewriter(inst, kernel_inst, input_r_shape, shuffle_indices, kernel_array_prtr)


def _rtr_transform(array, spatial_dimensions, feature_dimension, strides):
    # reshape to (..., C, ..., H//strideH, strideH, W//strideW, strideW, ...)
    r_shape = list(array.shape)
    ext_spatial_dimensions = []
    for idx, (dim, stride) in enumerate(zip(spatial_dimensions, strides)):
        pos = dim + idx
        ext_spatial_dimensions.append(pos)
        r_shape[pos] //= stride
        r_shape.insert(pos+1, stride)
    array_r = array.reshape(r_shape)

    # transpose to (..., strideH, strideW, C, ..., H//strideH, W//strideW, ...)
    ext_feature_dimension = feature_dimension
    for dim in spatial_dimensions:
        if dim < feature_dimension:
            ext_feature_dimension += 1
    perm = list(range(len(r_shape)))
    for dim in ext_spatial_dimensions:
        spatial_stride_dim = dim + 1
        perm.remove(spatial_stride_dim)
        perm.insert(perm.index(ext_feature_dimension), spatial_stride_dim)
    array_rt = array_r.transpose(perm)

    # reshape to (..., strideH*strideW*C, ..., H//strideH, W//strideW, ...)
    rtr_shape = list(array.shape)
    for dim, stride in zip(spatial_dimensions, strides):
        rtr_shape[dim] //= stride
        rtr_shape[feature_dimension] *= stride
    return array_rt.reshape(rtr_shape)


@decorate_methods_with(property)
class HloOp:

    xla_dtype_to_name = {
        xla_data_pb2.PRIMITIVE_TYPE_INVALID: 'PRIMITIVE_TYPE_INVALID',
        xla_data_pb2.PRED: 'uint8',
        xla_data_pb2.S8: 'int8',
        xla_data_pb2.S16: 'int16',
        xla_data_pb2.S32: 'int32',
        xla_data_pb2.S64: 'int64',
        xla_data_pb2.U8: 'uint8',
        xla_data_pb2.U16: 'uint16',
        xla_data_pb2.U32: 'uint32',
        xla_data_pb2.U64: 'uint64',
        xla_data_pb2.F16: 'float16',
        xla_data_pb2.F32: 'float32',
        xla_data_pb2.BF16: 'bfloat16',
        xla_data_pb2.F64: 'float64',
        xla_data_pb2.C64: 'complex64',
        xla_data_pb2.C128: 'complext128',
        xla_data_pb2.TUPLE: 'TUPLE',
        xla_data_pb2.OPAQUE_TYPE: 'OPAQUE_TYPE',
        xla_data_pb2.TOKEN: 'TOKEN',
    }

    xla_dtype_to_literal_attr_name = {
        xla_data_pb2.PRIMITIVE_TYPE_INVALID: 'PRIMITIVE_TYPE_INVALID',
        xla_data_pb2.PRED: 'preds',
        xla_data_pb2.S8: 's8s',
        xla_data_pb2.S16: 's16s',
        xla_data_pb2.S32: 's32s',
        xla_data_pb2.S64: 's64s',
        xla_data_pb2.U8: 'u8s',
        xla_data_pb2.U16: 'u16s',
        xla_data_pb2.U32: 'u32s',
        xla_data_pb2.U64: 'u64s',
        xla_data_pb2.F16: 'f16s',
        xla_data_pb2.F32: 'f32s',
        xla_data_pb2.BF16: 'bf16s',
        xla_data_pb2.F64: 'f64s',
        xla_data_pb2.C64: 'c64s',
        xla_data_pb2.C128: 'c128s',
        xla_data_pb2.TUPLE: 'tuple_literals',
        xla_data_pb2.OPAQUE_TYPE: 'OPAQUE_TYPE',
        xla_data_pb2.TOKEN: 'TOKEN',
    }

    opcode_map = {
        'and': '_and',
        'not': '_not',
    }

    def __init__(self, inst):
        self.inst = inst

    def opcode(self):
        return self.inst.opcode

    def legal_opcode(self):
        opc = self.inst.opcode
        return HloOp.opcode_map.get(opc, opc).replace('-', '_')

    def name(self):
        return self.inst.name

    def dtype(self):
        return self.xla_dtype_to_name[self.inst.shape.element_type]

    def shape(self):
        return list(self.inst.shape.dimensions)

    def operand_ids(self):
        return list(self.inst.operand_ids)

    def id(self):
        return self.inst.id

    def literal_value(self):
        literal_attr_name = self.xla_dtype_to_literal_attr_name[self.inst.shape.element_type]
        literals = getattr(self.inst.literal, literal_attr_name)
        dtype = dtypes.bfloat16.as_numpy_dtype if self.dtype == 'bfloat16' else self.dtype
        constructor = np.frombuffer if isinstance(literals, bytes) else np.asarray
        return constructor(literals, dtype=dtype).reshape(self.shape)

    @property
    def neff_input_name(self):
        return _get_frontend_attribute(self, 'neff_input_name')

    @neff_input_name.setter
    def neff_input_name(self, value):
        self.inst.frontend_attributes.map['neff_input_name'] = value

    @property
    def neff_output_names(self):
        return _get_frontend_attribute(self, 'neff_output_names').split(',')

    @neff_output_names.setter
    def neff_output_names(self, value):
        self.inst.frontend_attributes.map['neff_output_names'] = ','.join(value)


def _get_frontend_attribute(self, key):
    value = self.inst.frontend_attributes.map[key]
    if not value:
        raise ValueError('invalid {} found on instruction {}'.format(key, self.name))
    return value

