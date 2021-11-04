load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(default_visibility = ["//visibility:private"])

tf_cc_binary(
    name = "aws_neuron_tf2hlo",
    deps = [
        ":aws_neuron_tf2hlo_main",
        ":aws_neuron_custom_op",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "aws_neuron_tf2hlo_main",
    srcs = [
        "aws_neuron_tf2hlo.cc",
        "compile.h",
        "compile.cc",
        "flags.h",
        "flags.cc",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla",
        "//tensorflow/compiler/jit:xla_cpu_jit",
    ],
)

cc_library(
    name = "aws_neuron_custom_op",
    srcs = [
        "custom_op.cc",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla",
        "//tensorflow/compiler/jit:xla_cpu_jit",
    ],
    alwayslink = 1,
)
