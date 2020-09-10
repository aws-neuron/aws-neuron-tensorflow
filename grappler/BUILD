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
    name = "graph_optimizer_registry",
    srcs = ["graph_optimizer_registry.h"],
    hdrs = ["graph_optimizer_registry.h"],
    deps = [
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ],
)
