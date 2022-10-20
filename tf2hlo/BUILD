load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(default_visibility = ["//visibility:private"])

tf_cc_binary(
    name = "aws_neuron_tf2hlo",
    deps = [
        ":aws_neuron_tf2hlo_main",
    ],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../tensorflow"],
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
        "//tensorflow/neuron/grappler/convert:tf2xla",
    ],
)
