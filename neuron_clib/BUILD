package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library_additional_deps",)
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")

cc_library(
    name = "neuron_clib",
    srcs = [
        "neuron_clib.cc",
        "timestamps.h",
        "profiler.h",
        "profiler.cc",
        "tensor_util.h",
        "tensor_util.cc",
        "shared_memory.h",
    ],
    hdrs = ["neuron_clib.h"],
    deps = [
        "//tensorflow/compiler/xla/python:semaphore",
        ":nmgr_service.grpc.o",
        "@grpc//:grpc++_codegen_base",
    ] + tf_custom_op_library_additional_deps(),
)

# nrt proto libraries
cc_proto_library(
    name = "nmgr_service.grpc.o",
    srcs = [
        "nrt_proto/nmgr_service.proto",
        "nrt_proto/nmgr.proto",
        "nrt_proto/nmetric.proto",
        "nrt_proto/common.proto",
        "nrt_proto/nerr.proto",
    ],
    protoc = "@com_google_protobuf//:protoc",
    include = "nrt_proto/",
    use_grpc_plugin = True,
    default_runtime = "@com_google_protobuf//:protobuf",
    visibility = ["//visibility:public"],
)
