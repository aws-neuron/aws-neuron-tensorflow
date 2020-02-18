package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")


cc_library(
    name = "all_ops",
    deps = ["//tensorflow/neuron/runtime:neuron_op_op_lib"],
)

cc_library(
    name = "all_kernels",
    deps = ["//tensorflow/neuron/runtime:neuron_op"],
)

py_library(
    name = "graph_util_py",
    srcs = [
        "python/graph_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/neuron/convert:whitelist_partition_swig"],
)

py_library(
    name = "unittest_py",
    srcs = [
        "python/graph_util_test.py",
        "python/saved_model_test.py",
        "python/keras_test.py",
        "python/fuse_test.py",
        "python/op_register_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_util_py",
        ":saved_model_py",
        ":fuse_py"
    ],
)

py_library(
    name = "predictor_py",
    srcs = [
        "python/predictor.py",
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

tf_custom_op_library(
    name = "python/ops/_neuron_op.so",
    srcs = ["//tensorflow/neuron/runtime:ops/neuron_op.cc"],
    deps = ["//tensorflow/neuron/runtime:neuron_op"],
)

tf_gen_op_wrapper_py(
    name = "neuron_op",
    out = "ops/gen_neuron_op.py",
    deps = ["//tensorflow/neuron/runtime:neuron_op_op_lib"],
)

tf_custom_op_py_library(
    name = "neuron_op_py",
    srcs = glob(["python/ops/*.py"]),
    dso = [":python/ops/_neuron_op.so"],
    kernels = [
        "//tensorflow/neuron/runtime:neuron_op",
        "//tensorflow/neuron/runtime:neuron_op_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python:framework_for_generated_wrappers"]
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
    srcs = ["__init__.py"],
    deps = [
        ":saved_model_py",
        ":graph_util_py",
        ":predictor_py",
        ":neuron_ops_py",
        ":fuse_py",
        ":unittest_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/profiler",
        "//tensorflow/python/saved_model",
    ],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        ":neuron_py",
    ],
)
