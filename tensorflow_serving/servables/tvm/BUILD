# Description: Tensorflow Serving tvm servable

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["-layering_check"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

cc_library(
    name = "ndarray_util",
    srcs = ["ndarray_util.cc"],
    hdrs = ["ndarray_util.h", "tvm_loader.h",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_github_dmlc_tvm//:tvm_runtime",
        "//tensorflow_serving/apis:predict_proto",
        "//tensorflow_serving/core:servable_handle",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
    includes = ["@com_github_dmlc_tvm//3rdparty/dmlc-core/include",
	        "@com_github_dmlc_tvm//include",
	        "@com_github_dmlc_tvm//3rdparty/dlpack/include",
		"@com_github_dmlc_tvm//src/runtime"],
)

cc_library(
    name = "tvm_loader",
    srcs = ["tvm_loader.cc"],
    hdrs = ["tvm_loader.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_github_dmlc_tvm//:tvm_runtime",
        ":ndarray_util",
        "//tensorflow_serving/core:simple_loader",
    ],
    includes = ["@com_github_dmlc_tvm//3rdparty/dmlc-core/include",
	        "@com_github_dmlc_tvm//include",
	        "@com_github_dmlc_tvm//3rdparty/dlpack/include",
		"@com_github_dmlc_tvm//src/runtime"],
    alwayslink = 1,
)