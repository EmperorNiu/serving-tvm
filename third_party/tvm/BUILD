# TVM (tvm.ai) library.
# from https://github.com/dmlc/tvm.git

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache

exports_files(["LICENSE"])

cc_library(
    name = "tvm_runtime",
    srcs = [
        "src/runtime/c_runtime_api.cc",
        "src/runtime/cpu_device_api.cc",
        "src/runtime/workspace_pool.cc",
        "src/runtime/module.cc",
        "src/runtime/registry.cc",
        "src/runtime/threading_backend.cc",
        "src/runtime/thread_pool.cc",
        "src/runtime/ndarray.cc",
        "src/runtime/object.cc",
        "src/runtime/logging.cc",
        "src/runtime/file_utils.cc",
        "src/runtime/container.cc",
    ],
    hdrs = glob([
      "3rdparty/dmlc-core/include/**/*.h",
      "include/**/*.h",
      "3rdparty/dlpack/include/**/*.h",
      "src/runtime/**/*.h",
    ]),
    includes = ["3rdparty/dmlc-core/include",
	        "include",
	        "3rdparty/dlpack/include",
		"src/runtime",
    ],
    alwayslink = 1,
)

