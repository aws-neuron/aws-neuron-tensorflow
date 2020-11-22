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
        ":meta_graph_util_py",
        ":neuron_cc_py",
        ":neuron_op_py",
    ],
)

py_library(
    name = "utils_py",
    srcs = [
        "utils.py",
    ],
)

py_library(
    name = "neff_util_py",
    srcs = [
        "neff_util.py",
    ],
)

py_library(
    name = "graph_def_util_py",
    srcs = [
        "graph_def_util.py",
    ],
    deps = [
        ":neuron_cc_py",
        ":neff_util_py",
    ],
)

py_library(
    name = "meta_graph_util_py",
    srcs = [
        "meta_graph_util.py",
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

py_library(
    name = "saved_model_v2_py",
    srcs = [
        "saved_model_v2.py",
    ],
    deps = [
        ":graph_def_util_py",
        ":meta_graph_util_py",
        ":neuron_cc_py",
    ],
)

py_library(
    name = "neuron_cc_py",
    srcs = [
        "neuron_cc.py",
    ],
    deps = [
        ":utils_py",
    ]
)

tf_custom_op_library(
    name = "aws_neuron_plugin",
    per_os_targets = True,
    deps = [
        "//tensorflow/neuron/runtime:neuron_op_op_lib",
        "//tensorflow/neuron/runtime:neuron_op_kernel",
        "//tensorflow/neuron/grappler:passes",
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
    deps = [
        ":neuron_cc_py",
        ":neff_util_py",
        ":neuron_op_py",
    ],
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
        "tfv1_while_loop_test.py",
        "saved_model_test.py",
        "keras_test.py",
        "fuse_test.py",
        "performance_test.py",
        "op_register_test.py",
        "tfv1_keras_test.py",
        "test_early_exit.py",
        "unittest_base.py",
        "saved_model_v2_test.py",
    ],
    deps = [
        ":graph_util_py",
        ":saved_model_py",
        ":fuse_py",
        ":saved_model_v2_py",
    ],
)
