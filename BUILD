package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library_additional_deps")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")


cc_library(
    name = "all_ops",
    deps = [":inferentia_op_op_lib"],
)

cc_library(
    name = "all_kernels",
    deps = [":inferentia_op_kernel"],
)

cc_library(
    name = "neuron_logging",
    srcs = ["util/logging.cc"],
    hdrs = ["util/logging.h"],
    deps = ["//tensorflow/core:logger"],
)

tf_py_wrap_cc(
    name = "whitelist_partition_swig",
    srcs = ["convert/whitelist_partition.i"],
    copts = tf_copts(),
    deps = [
        ":convert_graph",
        ":inferentia_op_kernel",
        "//tensorflow/c:tf_status_helper",
        "//third_party/python_runtime:headers",
    ],
)

py_library(
    name = "graph_util_py",
    srcs = [
        "python/graph_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":whitelist_partition_swig"],
)

py_library(
    name = "graph_util_test_py",
    srcs = [
        "python/graph_util_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":graph_util_py"],
)

py_library(
    name = "predictor_py",
    srcs = [
        "python/predictor/neuron_predictor.py",
        "python/saved_model_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":graph_util_py"],
)

py_library(
    name = "saved_model_py",
    srcs = [
        "python/saved_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":graph_util_py"],
)

py_library(
    name = "saved_model_test_py",
    srcs = [
        "python/saved_model_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":saved_model_py"],
)

py_library(
    name = "fuse_test_py",
    srcs = [
        "python/fuse_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":fuse_py"],
)

tf_custom_op_library(
    name = "python/ops/_inferentia_op.so",
    srcs = ["ops/inferentia_op.cc"],
    deps = [":inferentia_op_kernel"],
)

cc_library(
    name = "inferentia_op_kernel",
    srcs = ["kernels/inferentia_op.cc"],
    hdrs = [
        "kernels/inferentia_op.h",
        "util/logging.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        ":neuron_logging",
        ":neuron_clib",
    ] + tf_custom_op_library_additional_deps(),
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["inferentia_op"],
)

tf_gen_op_wrapper_py(
    name = "inferentia_op",
    out = "ops/gen_inferentia_op.py",
    deps = [":inferentia_op_op_lib"],
)

tf_custom_op_py_library(
    name = "inferentia_py",
    srcs = glob(["python/ops/*.py"]),
    dso = [":python/ops/_inferentia_op.so"],
    kernels = [":inferentia_op_kernel",":inferentia_op_op_lib"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python:framework_for_generated_wrappers"]
)

cc_library(
    name = "convert_graph",
    srcs = ["convert/convert_graph.cc"],
    hdrs = ["convert/convert_graph.h"],
    deps = [
        ":segment",
		"//tensorflow/python/neuron:all_ops",
        "//tensorflow/core:core_cpu_headers_lib",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        ":neuron_logging",
        ":neuron_clib",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "segment",
    srcs = ["segment/segment.cc"],
    hdrs = [
        "segment/segment.h",
        "segment/union_find.h",
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:core",
        "//tensorflow/core:protos_all_cc",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

cc_library(
    name = "neuron_clib",
    deps = ["//tensorflow/python/neuron/neuron_clib:neuron_clib"],
)

py_library(
    name = "inferentia_ops_py",
    srcs_version = "PY2AND3",
    deps = [":inferentia_op", ":inferentia_py"],
)

py_library(
    name = "fuse_py",
    srcs = [
        "python/fuse.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":inferentia_ops_py"],
)

py_library(
    name = "neuron_py",
    deps = [
        ":saved_model_py",
        ":graph_util_py",
        ":predictor_py",
        ":inferentia_ops_py",
        ":fuse_py",
        ":graph_util_test_py",
        ":saved_model_test_py",
        ":fuse_test_py",
    ],
)
