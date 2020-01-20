package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")

cc_library(
    name = "device",
    srcs = [
        "device.cc",
        "timestamps.h",
        "profiler.h",
        "profiler.cc",
        "tensor_util.h",
        "tensor_util.cc",
        "shared_memory.h",
        "runtime_grpc.h",
        "runtime_grpc.cc",
        "semaphore.h",
        "semaphore.cc",
    ],
    hdrs = ["device.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        ":nmgr_service.grpc.o",
        "@grpc//:grpc++_codegen_base",
    ],
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
)
