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

#ifndef TENSORFLOW_NEURON_GRAPPLER_GRAPH_OPTIMIZER_REGISTRY_H_
#define TENSORFLOW_NEURON_GRAPPLER_GRAPH_OPTIMIZER_REGISTRY_H_

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"

namespace tensorflow {
namespace grappler {
namespace neuron {

class NeuronGraphOptimizerRegistry : CustomGraphOptimizerRegistry {
 public:
  static void RegisterOptimizerOnce(
      const CustomGraphOptimizerRegistry::Creator& creator,
      const std::string& name) {
    auto opts = GetRegisteredOptimizers();
    if (std::find(opts.begin(), opts.end(), name) == opts.end()) {
      RegisterOptimizerOrDie(creator, name);
    }
  }
};

class NeuronGraphOptimizerRegistrar {
 public:
  explicit NeuronGraphOptimizerRegistrar(
      const CustomGraphOptimizerRegistry::Creator& creator,
      const std::string& name) {
    NeuronGraphOptimizerRegistry::RegisterOptimizerOnce(creator, name);
  }
};

#define REGISTER_NEURON_GRAPH_OPTIMIZER_AS(MyCustomGraphOptimizerClass, name) \
  namespace {                                                                 \
  static ::tensorflow::grappler::neuron::NeuronGraphOptimizerRegistrar        \
      MyCustomGraphOptimizerClass##_registrar(                                \
          []() { return new MyCustomGraphOptimizerClass; }, (name));          \
  }  // namespace

}  // end namespace neuron
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_NEURON_GRAPPLER_GRAPH_OPTIMIZER_REGISTRY_H_
