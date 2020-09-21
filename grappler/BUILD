package(default_visibility = ["//visibility:public"])


cc_library(
    name = "fuse_supported_operators",
    srcs = ["fuse_supported_operators.cc"],
    hdrs = ["fuse_supported_operators.h"],
    deps = [
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/neuron/convert:convert_graph",
        ":graph_optimizer_registry",
    ],
)

cc_library(
    name = "static_shape_inference",
    srcs = [
        "static_shape_inference.cc",
        "shape_inference.cc",
        "shape_inference_helpers.cc",
    ],
    hdrs = ["static_shape_inference.h"],
    deps = [
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        "//tensorflow/compiler/xla:statusor",
        ":graph_optimizer_registry",
    ],
)

cc_library(
    name = "graph_optimizer_registry",
    srcs = ["graph_optimizer_registry.h"],
    hdrs = ["graph_optimizer_registry.h"],
    deps = [
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ],
)
