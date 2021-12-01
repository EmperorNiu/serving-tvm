#ifndef TENSORFLOW_SERVING_SERVABLES_TVM_NDARRAY_UTIL_H_
#define TENSORFLOW_SERVING_SERVABLES_TVM_NDARRAY_UTIL_H_

#include <string>
#include <unordered_set>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/servables/tvm/tvm_loader.h"
#include "tensorflow/core/framework/tensor.h"

// TVM Headers
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>

namespace tensorflow {
namespace serving {
using tvm::runtime::NDArray;
class TVMBundle;

std::string GetNameHint(TVMBundle *bundle, int type, int index);
DataType DLTypeToDataType(NDArray &ndarray);
size_t GetNDArraySize(NDArray &ndarray);
Status CopyNDArrayFromTensorProto(NDArray &ndarray, const TensorProto &ptensor);
Status CopyTensorFromNDArray(Tensor &tensor, NDArray &ndarray);
TensorInfo MakeTensorInforFromNDArray(NDArray &ndarray, std::string name);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TVM_NDARRAY_UTIL_H_