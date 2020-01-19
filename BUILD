package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")


cc_library(
    name = "all_ops",
    deps = [":neuron_op_op_lib"],
)

cc_library(
    name = "all_kernels",
    deps = [":neuron_op_kernel"],
)

py_library(
    name = "graph_util_py",
    srcs = [
        "python/graph_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python/neuron/convert:whitelist_partition_swig"],
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

py_library(
    name = "saved_model_test_py",
    srcs = [
        "python/saved_model_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":saved_model_py"],
)

py_library(
    name = "keras_test_py",
    srcs = [
        "python/keras_test.py",
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
    copts = ["-std=c++14"],
    deps = [
        "//tensorflow/python/neuron/neuron_clib:neuron_clib",
    ],
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
        ":graph_util_test_py",
        ":saved_model_test_py",
        ":keras_test_py",
        ":fuse_test_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/saved_model",
    ],
)


load(
    "//tensorflow/python/tools/api/generator:api_gen.bzl",
    "gen_api_init_files",
)
load(
    "//tensorflow/python/neuron:neuron_api_init_files.bzl",
    "TENSORFLOW_NEURON_API_INIT_FILES",
)

gen_api_init_files(
    name = "tf_neuron_python_api_gen",
    srcs = [
        "//tensorflow:api_template_v1.__init__.py",
    ],
    api_version = 1,
    output_dir = "_api/v1/",
    output_files = TENSORFLOW_NEURON_API_INIT_FILES,
    output_package = "tensorflow._api.v1",
    root_file_name = "v1.py",
    root_init_template = "//tensorflow:api_template_v1.__init__.py",
    packages = ["tensorflow.python.neuron"],
    package_deps = [":neuron_py"],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        ":neuron_py",
        ":tf_neuron_python_api_gen",
    ],
)
