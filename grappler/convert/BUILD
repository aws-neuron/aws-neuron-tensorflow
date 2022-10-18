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
    name = "mark_shape_context",
    srcs = ["mark_shape_context.cc"],
    hdrs = ["mark_shape_context.h"],
    deps = [
        ":graph_constructor_wrapper",
    ],
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/jit:xla_jit_headers_lib",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        ":graph_constructor_wrapper",
    ],
)

cc_library(
    name = "split_conv2d",
    srcs = ["split_conv2d.cc"],
    hdrs = ["split_conv2d.h"],
    deps = [
        ":graph_constructor_wrapper",
    ],
)

cc_library(
    name = "graph_constructor_wrapper",
    hdrs = ["graph_constructor_wrapper.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:core_cpu",
    ],
)
