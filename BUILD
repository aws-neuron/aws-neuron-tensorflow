package(default_visibility = ["//visibility:public"])


py_library(
    name = "neuron_py",
    srcs = [
        "__init__.py",
        "api/__init__.py",
        "tensorflow.py",
    ],
    deps = [
        "//tensorflow/neuron/python:saved_model_py",
        "//tensorflow/neuron/python:saved_model_v2_py",
        "//tensorflow/neuron/python:graph_util_py",
        "//tensorflow/neuron/python:predictor_py",
        "//tensorflow/neuron/python:neuron_op_py",
        "//tensorflow/neuron/python:fuse_py",
        "//tensorflow/neuron/python:performance_py",
        "//tensorflow/neuron/python:neuron_cc_py",
        "//tensorflow/neuron/python:unittest_py",
    ],
)

filegroup(
    name = "license",
    data = [
        "LICENSE",
        "THIRD-PARTY-LICENSES.txt",
    ],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [":neuron_py"],
    deps = [":license"],
)
