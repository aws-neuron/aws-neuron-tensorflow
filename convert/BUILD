package(default_visibility = ["//visibility:public"])


cc_library(
    name = "convert_graph",
    srcs = ["convert_graph.cc"],
    hdrs = ["convert_graph.h"],
    copts = ["-DGOOGLE_CUDA=1", "-DGOOGLE_TENSORRT=1"],
    deps = [
        ":segment",
        "//tensorflow/neuron/runtime:macros",
        "//tensorflow/core:core_cpu_headers_lib",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
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
        "//tensorflow/core:core",
        "//tensorflow/core:protos_all_cc",
        "@com_google_protobuf//:protobuf_headers",
    ],
)
