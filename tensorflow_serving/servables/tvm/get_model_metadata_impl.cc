#include "tensorflow_serving/servables/tvm/get_model_metadata_impl.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/tvm/tvm_loader.h"

namespace tensorflow {
namespace serving {

Status TVMModelGetSignatureDef(
       ServerCore* core, const ModelSpec& model_spec,
       const GetModelMetadataRequest& request,
       GetModelMetadataResponse* response) {
  ServableHandle<TVMBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));
  SignatureDefMap signature_def_map;
  for (const auto& signature : bundle->meta_graph_def.signature_def()) {
    (*signature_def_map.mutable_signature_def())[signature.first] =
        signature.second;
  }
  auto response_model_spec = response->mutable_model_spec();
  // TODO: name ??
  response_model_spec->set_name("tvm");
  response_model_spec->mutable_version()->set_value(bundle.id().version);

  (*response->mutable_metadata())["signature_def"].PackFrom(
      signature_def_map);
  return tensorflow::Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
