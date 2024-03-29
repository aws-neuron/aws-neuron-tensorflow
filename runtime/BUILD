package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_kernel_library")
load(":build_defs.bzl", "if_tf_serving")


cc_library(
    name = "all_ops",
    deps = [
        ":neuron_op_op_lib",
        ":neuron_op_kernel",
        ":identity_op",
        ":avgpooling_op",
        ":maxpooling_op",
        ":constant_op",
        ":islibmodeavailable_op",
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

tf_kernel_library(
    name = "islibmodeavailable_op",
    srcs = [
        "kernels/islibmodeavailable_op.cc",
    ],
    deps = [
        ":device",
        ":registration",
        ":direct/placer",
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
    deps = [
        ":semaphore",
        ":direct/env",
        ":direct/macros",
        ":direct/tensor_util",
        ":direct/version",
    ],
)
cc_library(
    name = "semaphore",
    srcs = ["semaphore.cc"],
    hdrs = ["semaphore.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "registration",
    srcs = ["kernels/registration.h"],
    hdrs = ["kernels/registration.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
    defines = if_tf_serving(["AWS_NEURON_TFSERV"]),
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
        "nrt_proto/nmgr_service.proto",
        "nrt_proto/nmgr_session_service.proto",
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

filegroup(
    name = "build_file",
    srcs = ["BUILD"],
)

genrule(
    name = "nrt_proto",
    srcs = [":build_file"],
    outs = [
        "nrt_proto/nmgr_service.proto",
        "nrt_proto/nmgr_session_service.proto",
        "nrt_proto/nmgr.proto",
        "nrt_proto/nmetric.proto",
        "nrt_proto/common.proto",
        "nrt_proto/nerr.proto",
    ],
    cmd = """
        TARGET=$(@D)/nrt_proto
        rm -rf $$TARGET
        LOCAL_PROTO=$$(dirname $(location build_file))/proto
        if [ -d $$LOCAL_PROTO ]; then
            cp -r $$LOCAL_PROTO $$TARGET
        else
            git clone https://github.com/aws/aws-neuron-runtime-proto $$TARGET
            ( cd $$TARGET && git checkout ed0897dd3599ab3e8b63c156c88261651e3e46cf )
        fi
    """,
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
        ":direct/dynamic_batch",
        ":direct/placer",
        ":direct/function",
    ],
)

cc_library(
    name = "direct/function",
    hdrs = ["direct/function.h"],
    srcs = ["direct/function.cc"],
    deps = [
        ":direct/macros",
        ":direct/routine",
    ],
)

cc_library(
    name = "direct/routine",
    hdrs = ["direct/routine.h"],
    srcs = ["direct/routine.cc"],
    deps = [
        ":direct/core_range",
        ":direct/dynamic_batch",
        ":direct/executable",
        ":direct/executable_info",
        ":direct/host_memory",
        ":direct/macros",
        ":direct/placer",
        ":direct/tensor_util",
    ],
)

cc_library(
    name = "direct/placer",
    hdrs = ["direct/placer.h"],
    srcs = ["direct/placer.cc"],
    deps = [
        ":direct/adaptor",
        ":direct/core_range",
        ":direct/env",
        ":direct/executable_info",
        ":direct/macros",
    ],
)

cc_library(
    name = "direct/executable",
    hdrs = ["direct/executable.h"],
    srcs = ["direct/executable.cc"],
    deps = [
        ":direct/adaptor",
        ":direct/core_range",
        ":direct/executable_info",
        ":direct/host_memory",
        ":direct/macros",
        ":direct/profiler_context",
    ],
)

cc_library(
    name = "direct/host_memory",
    hdrs = ["direct/host_memory.h"],
    srcs = ["direct/host_memory.cc"],
    deps = [
        ":direct/adaptor",
        ":direct/executable_info",
        ":direct/macros",
    ],
)

cc_library(
    name = "direct/dynamic_batch",
    hdrs = ["direct/dynamic_batch.h"],
    srcs = ["direct/dynamic_batch.cc"],
    deps = [
        ":direct/executable_info",
    ],
)

# nrt profiler
cc_library(
    name = "direct/profiler_context",
    hdrs = ["direct/profiler_context.h"],
    srcs = ["direct/profiler_context.cc"],
    deps = [
        ":direct/adaptor",
        ":direct/env",
        ":direct/executable_info",
        ":direct/macros",
    ],
)

cc_library(
    name = "direct/executable_info",
    hdrs = ["direct/executable_info.h"],
    srcs = ["direct/executable_info.cc"],
    deps = [
        ":direct/node_def_keys",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "direct/tensor_util",
    srcs = ["direct/tensor_util.cc"],
    hdrs = ["direct/tensor_util.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "direct/env",
    srcs = ["direct/env.cc"],
    hdrs = ["direct/env.h"],
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
    srcs = ["direct/adaptor.cc", ":libnrt_headers"],
    deps = [
        ":libnrt",
        ":direct/macros",
        ":direct/version",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "direct/macros",
    srcs = ["direct/macros.h"],
    hdrs = ["direct/macros.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library(
    name = "direct/node_def_keys",
    hdrs = ["direct/node_def_keys.h"],
)

cc_library(
    name = "direct/version",
    hdrs = ["direct/version.h"],
)

# nrt C library
cc_library(
    name = "libnrt",
    hdrs = [":libnrt_headers"],
    srcs = [":libnrt.so.1"],
    linkopts = ["-Wl,-rpath,/opt/aws/neuron/lib:$$ORIGIN/../tensorflow_neuron/runtime/direct/nrt:$$ORIGIN/nrt"],
    strip_include_prefix = "direct",
)

filegroup(
    name = "libnrt_shared_library",
    srcs = glob(["direct/nrt/*.so.*"]),
)

# Generate a "fake" libnrt.so.1 solely for linking purpose.
cc_binary(
    name = "libnrt.so.1",
    linkshared = True,
    srcs = [":fake_libnrt_cc", ":libnrt_headers"],
    includes = ["direct"],
    linkopts = ["-Wl,-soname,libnrt.so.1"],
)

genrule(
    name = "fake_libnrt_cc",
    srcs = [":libnrt_headers", ":direct/gen_mock_nrt_cc_py"],
    outs = ["fake_libnrt.cc"],
    cmd = "python3 $(locations :direct/gen_mock_nrt_cc_py) $(locations :libnrt_headers) > $@",
)

filegroup(
    name = "direct/gen_mock_nrt_cc_py",
    srcs = ["direct/gen_mock_nrt_cc.py"],
)

genrule(
    name = "libnrt_headers",
    srcs = [":build_file"],
    outs = [
        "direct/nrt/nrt.h",
        "direct/nrt/nrt_status.h",
        "direct/nrt/nrt_profile.h",
        "direct/nrt/nrt_experimental.h",
    ],
    cmd = """
        TARGET=$(@D)/direct/nrt
        rm -rf $$TARGET
        CURRENT_DIR=$$(dirname $(location build_file))
        LOCAL_NRT_HEADERS=$$CURRENT_DIR/direct/nrt_headers
        if [ -d $$LOCAL_NRT_HEADERS ]; then
            cp -r $$LOCAL_NRT_HEADERS $$TARGET
        else
            git clone https://github.com/aws-neuron/aws-neuron-sdk $$CURRENT_DIR/aws-neuron-sdk
            ( cd $$CURRENT_DIR/aws-neuron-sdk && git checkout ceb5ba97b09d35d8abc178e4e5bef2b0467f21dd )
            cp -r $$CURRENT_DIR/aws-neuron-sdk/src/libnrt/include/nrt $$TARGET
        fi
    """,
)
