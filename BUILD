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
    deps = [":neuron_op_op_lib"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "all_kernels",
    deps = [":neuron_op_kernel"],
    visibility = ["//visibility:public"],
)

tf_py_wrap_cc(
    name = "whitelist_partition_swig",
    srcs = ["convert/whitelist_partition.i"],
    copts = tf_copts(),
    deps = [
        ":convert_graph",
        ":neuron_op_kernel",
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
    name = "python/ops/_neuron_op.so",
    srcs = ["ops/neuron_op.cc"],
    deps = [":neuron_op_kernel"],
)

cc_library(
    name = "neuron_op_kernel",
    srcs = ["kernels/neuron_op.cc"],
    hdrs = [
        "kernels/neuron_op.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        ":neuron_clib",
    ] + tf_custom_op_library_additional_deps(),
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["neuron_op"],
)

tf_gen_op_wrapper_py(
    name = "neuron_op",
    out = "ops/gen_neuron_op.py",
    deps = [":neuron_op_op_lib"],
)

tf_custom_op_py_library(
    name = "neuron_op_py",
    srcs = glob(["python/ops/*.py"]),
    dso = [":python/ops/_neuron_op.so"],
    kernels = [":neuron_op_kernel",":neuron_op_op_lib"],
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
    name = "neuron_ops_py",
    srcs_version = "PY2AND3",
    deps = [":neuron_op", ":neuron_op_py"],
)

py_library(
    name = "fuse_py",
    srcs = [
        "python/fuse.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":neuron_ops_py"],
)

py_library(
    name = "neuron_py",
    deps = [
        ":saved_model_py",
        ":graph_util_py",
        ":predictor_py",
        ":neuron_ops_py",
        ":fuse_py",
        ":graph_util_test_py",
        ":saved_model_test_py",
        ":fuse_test_py",
    ],
)
