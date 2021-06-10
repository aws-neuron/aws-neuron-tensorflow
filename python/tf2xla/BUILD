load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")


tf_proto_library(
    name = "tf2xla_proto",
    srcs = ["tf2xla.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"]
)
