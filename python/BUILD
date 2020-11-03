package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")


py_library(
    name = "graph_util_py",
    srcs = [
        "graph_util.py",
    ],
    deps = [
        ":graph_def_util_py",
        ":neuron_op_py",
    ],
)

py_library(
    name = "graph_def_util_py",
    srcs = [
        "graph_def_util.py",
    ],
)

py_library(
    name = "predictor_py",
    srcs = [
        "predictor.py",
        "saved_model_util.py",
    ],
    deps = [":graph_util_py"],
)

py_library(
    name = "saved_model_py",
    srcs = [
        "saved_model.py",
    ],
    deps = [":graph_util_py"],
)

tf_custom_op_library(
    name = "aws_neuron_plugin",
    per_os_targets = True,
    deps = [
        "//tensorflow/neuron/runtime:neuron_op_op_lib",
        "//tensorflow/neuron/runtime:neuron_op_kernel",
        "//tensorflow/neuron/grappler:fuse_supported_operators",
    ],
)

tf_custom_op_py_library(
    name = "neuron_op_py",
    dso = [":aws_neuron_plugin"],
    deps = [":gen_neuron_op_py"]
)

tf_gen_op_wrapper_py(
    name = "gen_neuron_op_py",
    out = "ops/gen_neuron_op.py",
    deps = ["//tensorflow/neuron/runtime:neuron_op_op_lib"],
)

py_library(
    name = "fuse_py",
    srcs = [
        "fuse.py",
    ],
    deps = [":neuron_op_py"],
)

py_library(
    name = "performance_py",
    srcs = [
        "performance.py",
    ],
    deps = [":neuron_op_py"],
)

py_library(
    name = "unittest_py",
    srcs = [
        "graph_util_test.py",
        "saved_model_test.py",
        "keras_test.py",
        "fuse_test.py",
        "performance_test.py",
        "op_register_test.py",
        "tfv1_keras_test.py",
    ],
    deps = [
        ":graph_util_py",
        ":saved_model_py",
        ":fuse_py"
    ],
)