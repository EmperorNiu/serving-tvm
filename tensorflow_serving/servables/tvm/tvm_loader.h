#ifndef TENSORFLOW_SERVING_SERVABLES_TVM_TVM_LOADER_H_
#define TENSORFLOW_SERVING_SERVABLES_TVM_TVM_LOADER_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow_serving/servables/tvm/ndarray_util.h"
// TVM Headers
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>

namespace tensorflow {
namespace serving {

struct TVMBundle
{
    ~TVMBundle(){
    }

    tvm::runtime::Module mod;
    MetaGraphDef meta_graph_def;
    TVMBundle() = default;
};

// Loads a SavedModel from the specified export directory. 
Status TVMLoadModel(const std::string& export_dir,
                  TVMBundle* const bundle);

} // serving
} // tensorflow

#endif // TENSORFLOW_SERVING_SERVABLES_TVM_TVM_LOADER_H_
