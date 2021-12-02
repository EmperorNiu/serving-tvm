#ifndef TENSORFLOW_SERVING_SERVABLES_TVM_GET_MODEL_METADATA_IMPL_H_
#define TENSORFLOW_SERVING_SERVABLES_TVM_GET_MODEL_METADATA_IMPL_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace tensorflow {
namespace serving {

Status TVMModelGetSignatureDef(
       ServerCore* core, const ModelSpec& model_spec,
       const GetModelMetadataRequest& request,
       GetModelMetadataResponse* response);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_GET_MODEL_METADATA_IMPL_H_