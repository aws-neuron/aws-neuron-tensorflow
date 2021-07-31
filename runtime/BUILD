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
        ":init",
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
        ":direct",
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
        ":direct/linker",
    ],
)

cc_library(
    name = "init",
    srcs = ["init.cc"],
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
        "profiler.cc",
    ],
    hdrs = [
        "runtime_grpc.h",
        "profiler.h",
    ],
    deps = [
        ":utils",
        ":nmgr_service.grpc.o",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "tensor_util.h",
        "tensor_util.cc",
        "semaphore.h",
        "semaphore.cc",
        "env.h",
        "env.cc",
    ],
    hdrs = [
        "version.h",
        "tensor_util.h",
        "semaphore.h",
        "env.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        ":macros",
    ],
)

cc_library(
    name = "macros",
    srcs = ["macros.h"],
    hdrs = ["macros.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
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

# direct-link mode runtime
cc_library(
    name = "direct",
    deps = [
        ":direct/adaptor",
        ":direct/core_range",
        ":direct/executable_info",
        ":direct/host_memory",
        ":direct/executable",
        ":direct/placer",
        ":direct/function",
    ],
)

cc_library(
    name = "direct/function",
    hdrs = ["direct/function.h"],
    srcs = ["direct/function.cc"],
    deps = [
        ":direct/core_range",
        ":direct/executable",
        ":direct/executable_info",
        ":direct/host_memory",
        ":direct/placer",
        ":macros",
    ],
)

cc_library(
    name = "direct/placer",
    hdrs = ["direct/placer.h"],
    srcs = ["direct/placer.cc"],
    deps = [
        ":direct/adaptor",
        ":direct/core_range",
        ":direct/executable_info",
    ],
)

cc_library(
    name = "direct/executable",
    hdrs = ["direct/executable.h"],
    srcs = ["direct/executable.cc"],
    deps = [
        ":direct/adaptor",
        ":direct/core_range",
        ":direct/host_memory",
        ":macros",
    ],
)

cc_library(
    name = "direct/host_memory",
    hdrs = ["direct/host_memory.h"],
    srcs = ["direct/host_memory.cc"],
    deps = [
        ":direct/adaptor",
        ":direct/executable_info",
        ":macros",
    ],
)

cc_library(
    name = "direct/executable_info",
    hdrs = ["direct/executable_info.h"],
    srcs = ["direct/executable_info.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "direct/core_range",
    hdrs = ["direct/core_range.h"],
)

# tensorflow-adapted nrt library
cc_library(
    name = "direct/adaptor",
    hdrs = ["direct/adaptor.h"],
    srcs = ["direct/adaptor.cc"],
    deps = [
        ":libnrt",
        ":utils",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

# TODO: remove once nrt library is fully integrated
# dummy target to trigger nrt library linkage
cc_library(
    name = "direct/linker",
    hdrs = ["direct/linker.h"],
    srcs = ["direct/linker.cc"],
    deps = [
        ":libnrt",
    ],
)

# nrt C library
cc_library(
    name = "libnrt",
    hdrs = glob(["direct/include/nrt/*.h"]),
    srcs = [":libnrt_shared_library"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../tensorflow_neuron/runtime/direct/lib/nrt"],
    strip_include_prefix = "direct/include",
)

filegroup(
    name = "libnrt_shared_library",
    srcs = glob(["direct/lib/nrt/*.so.*", "direct/lib/nrt/fake_nrt.cc"]),
)
