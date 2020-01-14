package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load("//tensorflow:tensorflow.bzl", "tf_copts")

tf_py_wrap_cc(
    name = "whitelist_partition_swig",
    srcs = ["whitelist_partition.i"],
    copts = tf_copts(),
    deps = [
        ":convert_graph",
        "//tensorflow/python/neuron:neuron_op_kernel",
        "//tensorflow/c:tf_status_helper",
        "//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "convert_graph",
    srcs = ["convert_graph.cc"],
    hdrs = ["convert_graph.h"],
    copts = ["-DGOOGLE_CUDA=1", "-DGOOGLE_TENSORRT=1"],
    deps = [
        ":segment",
        "//tensorflow/python/neuron:all_ops",
        "//tensorflow/core:core_cpu_headers_lib",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "segment",
    srcs = [
        "//tensorflow/compiler/tf2tensorrt:segment/segment.cc",
        "//tensorflow/compiler/tf2tensorrt:segment/union_find.h",
    ],
    hdrs = [
        "//tensorflow/compiler/tf2tensorrt:segment/segment.h",
    ],
    copts = ["-DGOOGLE_CUDA=1", "-DGOOGLE_TENSORRT=1"],
    deps = [
        "//tensorflow/core:core",
        "//tensorflow/core:protos_all_cc",
        "@com_google_protobuf//:protobuf_headers",
    ],
)
