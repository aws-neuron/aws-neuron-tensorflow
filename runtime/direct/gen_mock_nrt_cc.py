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
import sys


def main():
    fake_body_map = {
        'NRT_STATUS ': ' { return NRT_FAILURE; }',
        'void ': ' {}',
        'size_t ': ' { return 0; }',
    }
    api_lines = []
    for path in sys.argv[1:]:
        header_name = path[path.find('nrt/nrt'):]
        print('#include "{}"'.format(header_name))
        with open(path) as f:
            context = None
            for line in f:
                line = line.strip()
                if context is None:
                    for ctx in fake_body_map:
                        if line.startswith(ctx) and '(' in line:
                            context = ctx
                            break
                if context is not None:
                    api_lines.append((line, context))
                    if line.endswith(');'):
                        context = None
    api_lines = [line.replace(';', fake_body_map[ctx]) for line, ctx in api_lines]
    print('extern "C" {')
    for line in api_lines:
        print(line)
    print('}  // extern "C"')


if __name__ == '__main__':
    main()
