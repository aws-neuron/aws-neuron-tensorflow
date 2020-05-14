package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")


py_library(
    name = "graph_util_py",
    srcs = [
        "python/graph_util.py",
    ],
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
    deps = [":graph_util_py"],
)

py_library(
    name = "saved_model_py",
    srcs = [
        "python/saved_model.py",
    ],
    deps = [":graph_util_py"],
)

tf_custom_op_library(
    name = "python/ops/_neuron_op.so",
    deps = [
        "//tensorflow/neuron/runtime:neuron_op_op_lib",
        "//tensorflow/neuron/runtime:neuron_op_kernel",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_neuron_op_py",
    out = "ops/gen_neuron_op.py",
    deps = ["//tensorflow/neuron/runtime:neuron_op_op_lib"],
)

tf_custom_op_py_library(
    name = "neuron_op_py",
    dso = [":python/ops/_neuron_op.so"],
    deps = [":gen_neuron_op_py"]
)

py_library(
    name = "fuse_py",
    srcs = [
        "python/fuse.py",
    ],
    deps = [":neuron_op_py"],
)

py_library(
    name = "neuron_py",
    srcs = ["__init__.py"],
    deps = [
        ":saved_model_py",
        ":graph_util_py",
        ":predictor_py",
        ":neuron_op_py",
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
