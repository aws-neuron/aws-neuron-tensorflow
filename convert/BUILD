package(default_visibility = ["//visibility:public"])


cc_library(
    name = "convert_graph",
    srcs = ["convert_graph.cc"],
    hdrs = ["convert_graph.h"],
    copts = ["-DGOOGLE_CUDA=1", "-DGOOGLE_TENSORRT=1"],
    deps = [
        ":segment",
        "//tensorflow/core:core_cpu_headers_lib",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
    ],
)

genrule(
    name = 'hacked_tf2tensorrt_segment',
    srcs = [
        "//tensorflow/compiler/tf2tensorrt:segment/segment.h",
        "//tensorflow/compiler/tf2tensorrt:segment/segment.cc",
        "//tensorflow/compiler/tf2tensorrt:segment/union_find.h",
    ],
    outs = ['segment.h', "segment.cc", "union_find.h"],
    cmd = """
        cp $(location //tensorflow/compiler/tf2tensorrt:segment/segment.h) $(@D)/segment.h
        sed 's/tensorflow\/compiler\/tf2tensorrt\/segment\///g' $(location //tensorflow/compiler/tf2tensorrt:segment/segment.cc) > $(@D)/segment.cc.0
        sed 's/not converted to TensorRT/not compiled by neuron-cc/g' $(@D)/segment.cc.0 > $(@D)/segment.cc.1
        sed 's/docs.nvidia.com\/deeplearning/github.com\/aws\/aws-neuron-sdk/g' $(@D)/segment.cc.1 > $(@D)/segment.cc.2
        sed 's/\/frameworks\/tf-trt-user-guide\/index.html#supported-ops/\/blob\/master\/release-notes\/neuron-cc-ops\/neuron-cc-ops-tensorflow.md/g' $(@D)/segment.cc.2 > $(@D)/segment.cc
        cp $(location //tensorflow/compiler/tf2tensorrt:segment/union_find.h) $(@D)/union_find.h
    """,
)

cc_library(
    name = "segment",
    srcs = [
        "segment.cc",
        "union_find.h",
    ],
    hdrs = [
        "segment.h",
    ],
    copts = ["-DGOOGLE_CUDA=1", "-DGOOGLE_TENSORRT=1"],
    deps = [
        "//tensorflow/core:core",
        "//tensorflow/core:protos_all_cc",
        "@com_google_protobuf//:protobuf_headers",
    ],
)
