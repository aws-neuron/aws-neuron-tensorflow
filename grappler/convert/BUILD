package(default_visibility = ["//visibility:public"])

cc_library(
    name = "convert_graph",
    srcs = ["convert_graph.cc"],
    hdrs = ["convert_graph.h"],
    deps = [
        ":segment",
        ":graph_constructor_wrapper",
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
        "@org_tensorflow//tensorflow/core:core_cpu",
    ],
)

cc_library(
    name = "graph_constructor_wrapper",
    hdrs = ["graph_constructor_wrapper.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:core_cpu",
    ],
)
