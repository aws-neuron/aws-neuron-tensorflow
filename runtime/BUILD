package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_kernel_library")


cc_library(
    name = "all_ops",
    deps = [
        ":neuron_op_op_lib",
        ":neuron_op_kernel",
        ":identity_op",
        ":avgpooling_op",
        ":maxpooling_op",
        ":constant_op",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["neuron_op"],
)

tf_kernel_library(
    name = "neuron_op_kernel",
    srcs = [
        "kernels/neuron_op.cc",
    ],
    hdrs = ["kernels/neuron_op.h"],
    deps = [
        ":model",
        ":device",
        ":registration",
    ],
)

tf_kernel_library(
    name = "identity_op",
    srcs = [
        "kernels/identity_op.cc",
    ],
    deps = [
        ":device",
        ":registration",
    ],
)

tf_kernel_library(
    name = "avgpooling_op",
    srcs = [
        "kernels/avgpooling_op.cc",
    ],
    deps = [
        ":device",
        ":registration",
        ":pooling_utils",
    ],
)

tf_kernel_library(
    name = "maxpooling_op",
    srcs = [
        "kernels/maxpooling_op.cc",
    ],
    deps = [
        ":device",
        ":registration",
        ":pooling_utils",
    ],
)

tf_kernel_library(
    name = "conv_ops",
    srcs = [
        "kernels/conv_ops.cc",
    ],
    deps = [
        ":device",
        ":registration",
    ],
)

tf_kernel_library(
    name = "constant_op",
    srcs = [
        "kernels/constant_op.cc",
    ],
    deps = [
        ":device",
        ":registration",
    ],
)

cc_library(
    name = "model",
    srcs = [
        "model.cc",
        "model.h",
        "model_config.h",
    ],
    hdrs = ["model.h"],
    deps = [
        ":utils",
        ":engine",
        ":device",
    ],
)

cc_library(
    name = "engine",
    srcs = ["engine.cc"],
    hdrs = ["engine.h"],
    deps = [
        ":utils",
        ":shared_memory",
        ":runtime_grpc",
    ],
)

cc_library(
    name = "device",
    srcs = ["device.cc"],
    hdrs = ["device.h"],
    deps = [
        ":utils",
        ":shared_memory",
        ":engine",
        "@org_tensorflow//tensorflow/core:core_cpu",
    ],
    alwayslink=1,  # for linking device into tf-serving
)

cc_library(
    name = "shared_memory",
    srcs = ["shared_memory.cc"],
    hdrs = ["shared_memory.h"],
    deps = [
        ":utils",
        ":runtime_grpc",
    ],
)

cc_library(
    name = "runtime_grpc",
    srcs = [
        "runtime_grpc.cc",
        "version.h",
    ],
    hdrs = ["runtime_grpc.h"],
    deps = [
        ":utils",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "profiler.h",
        "profiler.cc",
        "tensor_util.h",
        "tensor_util.cc",
        "semaphore.h",
        "semaphore.cc",
        "env.h",
        "env.cc",
    ],
    hdrs = [
        "profiler.h",
        "tensor_util.h",
        "semaphore.h",
        "env.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        ":nmgr_service.grpc.o",
        ":macros",
    ],
)

cc_library(
    name = "macros",
    srcs = ["macros.h"],
    hdrs = ["macros.h"],
)

cc_library(
    name = "registration",
    srcs = ["kernels/registration.h"],
    hdrs = ["kernels/registration.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library(
    name = "pooling_utils",
    srcs = ["kernels/pooling_utils.h"],
    hdrs = ["kernels/pooling_utils.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

# nrt proto libraries
cc_proto_library(
    name = "nmgr_service.grpc.o",
    srcs = [
        "proto/nmgr_service.proto",
        "proto/nmgr_session_service.proto",
        "proto/nmgr.proto",
        "proto/nmetric.proto",
        "proto/common.proto",
        "proto/nerr.proto",
    ],
    protoc = "@com_google_protobuf//:protoc",
    include = "proto/",
    use_grpc_plugin = True,
    default_runtime = "@com_google_protobuf//:protobuf",
)
