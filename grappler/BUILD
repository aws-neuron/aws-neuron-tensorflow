package(default_visibility = ["//visibility:public"])


cc_library(
    name = "fuse_supported_operators",
    srcs = ["fuse_supported_operators.cc"],
    hdrs = ["fuse_supported_operators.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/neuron/convert:convert_graph",
    ],
)
