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
import struct


def get_model_config(executable):
    default_model_config = [-1, -1, -1, 10]
    if not executable:
        return default_model_config
    tuple_cores = get_cores_from_executable(executable)
    if tuple_cores is None:
        return default_model_config
    opt_num_cores, min_num_cores = tuple_cores
    est_infer_timeout = len(executable) / 1e8
    infer_timeout = max(est_infer_timeout, 10)
    model_config = [-1, opt_num_cores, opt_num_cores, infer_timeout]
    return model_config


_NC_HEADER_SIZE = 544
_MAX_NUM_CORES = 64


def get_cores_from_executable(executable):
    header = executable[:_NC_HEADER_SIZE]
    if len(header) != _NC_HEADER_SIZE:
        return None
    info = struct.unpack('168xI304xI64B', header)
    if len(info) != 1 + 1 + _MAX_NUM_CORES:
        return None
    opt_num_cores = info[0]
    if opt_num_cores <= 0 or opt_num_cores > _MAX_NUM_CORES:
        return None
    min_num_cores = max(info[2:])
    if min_num_cores <= 0 or min_num_cores > _MAX_NUM_CORES:
        return None
    return opt_num_cores, min_num_cores
