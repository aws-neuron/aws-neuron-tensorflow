package(default_visibility = ["//visibility:public"])


cc_library(
    name = "passes",
    deps = [
        ":static_shape_inference",
        ":fuse_supported_operators",
        ":auto_mixed_precision",
        ":mark_ops_in_fixed_shape_context",
        ":split_conv2d_same_padding",
    ],
)

cc_library(
    name = "fuse_supported_operators",
    srcs = ["fuse_supported_operators.cc"],
    hdrs = ["fuse_supported_operators.h"],
    deps = [
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/neuron/grappler/convert:convert_graph",
        ":graph_optimizer_registry",
    ],
)

cc_library(
    name = "auto_mixed_precision",
    srcs = ["auto_mixed_precision.cc"],
    hdrs = ["auto_mixed_precision.h", "auto_mixed_precision_lists.h"],
    deps = [
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/neuron/grappler/convert:convert_graph",
        ":graph_optimizer_registry"
    ],
)

cc_library(
    name = "static_shape_inference",
    srcs = ["static_shape_inference.cc"],
    hdrs = ["static_shape_inference.h"],
    deps = [
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/neuron/grappler/convert:shape_inference",
        ":graph_optimizer_registry",
    ],
)

cc_library(
    name = "mark_ops_in_fixed_shape_context",
    srcs = ["mark_ops_in_fixed_shape_context.cc"],
    deps = [
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/neuron/grappler/convert:mark_shape_context",
        ":graph_optimizer_registry",
    ],
    alwayslink = 1,
)

cc_library(
    name = "split_conv2d_same_padding",
    srcs = ["split_conv2d_same_padding.cc"],
    deps = [
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/neuron/grappler/convert:graph_constructor_wrapper",
        "//tensorflow/lite/toco:toco_port",
        ":graph_optimizer_registry",
    ],
    alwayslink = 1,
)

cc_library(
    name = "graph_optimizer_registry",
    srcs = ["graph_optimizer_registry.h"],
    hdrs = ["graph_optimizer_registry.h"],
    deps = [
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ],
)
