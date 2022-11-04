/* Copyright Amazon Web Services and its Affiliates. All Rights Reserved.

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

#ifndef TENSORFLOW_NEURON_RUNTIME_VERSION_H_
#define TENSORFLOW_NEURON_RUNTIME_VERSION_H_

namespace tensorflow {
namespace neuron {

#define TFN_MAJOR_VERSION 99999
#define TFN_MINOR_VERSION 99999

#define TFN_STR_HELPER(x) #x
#define TFN_STR(x) TFN_STR_HELPER(x)

// e.g. "1.1" or "1.12".
#define TFN_VERSION_STRING \
  (TFN_STR(TFN_MAJOR_VERSION) "." TFN_STR(TFN_MINOR_VERSION))

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_VERSION_H_
