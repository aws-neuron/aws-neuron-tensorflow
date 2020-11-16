package(default_visibility = ["//visibility:public"])


cc_library(
    name = "convert_graph",
    srcs = ["convert_graph.cc"],
    hdrs = ["convert_graph.h"],
    copts = ["-DGOOGLE_CUDA=1", "-DGOOGLE_TENSORRT=1"],
    deps = [
        ":segment",
        "//tensorflow/neuron/runtime:macros",
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
    copts = ["-DGOOGLE_CUDA=1", "-DGOOGLE_TENSORRT=1"],
    deps = [
        "//tensorflow/core:core_cpu",
    ],
)
