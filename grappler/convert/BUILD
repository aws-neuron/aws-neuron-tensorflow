package(default_visibility = ["//visibility:public"])


cc_library(
    name = "convert_graph",
    srcs = ["convert_graph.cc"],
    hdrs = ["convert_graph.h"],
    deps = [
        ":segment",
        "//tensorflow/neuron/grappler:graph_constructor_wrapper",
        "//tensorflow/neuron/runtime:direct/macros",
    ],
)

cc_library(
    name = "segment",
    srcs = [
        "segment.cc",
        "union_find.h",
    ],
    hdrs = [
        "segment.h",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
    ],
)
