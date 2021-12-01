#include "tensorflow_serving/servables/tvm/tvm_factory.h"
#include "tensorflow_serving/servables/tvm/tvm_loader.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace serving {

Status TVMFactory::Create(
    const TVMConfig& config,
    std::unique_ptr<TVMFactory>* factory) {
  factory->reset(new TVMFactory(config));
  return Status::OK();
}

Status TVMFactory::EstimateResourceRequirement(
    const string& path, ResourceAllocation* estimate) const {
  // TODO
  Status status;
  return status;
  //return EstimateResourceFromPath(path, estimate);
}

Status TVMFactory::CreateTVM(
    const string& path, std::unique_ptr<TVMBundle>* bundle) {

  bundle->reset(new TVMBundle);
  TF_RETURN_IF_ERROR(TVMLoadModel(path, bundle->get()));

  return Status::OK();
}

TVMFactory::TVMFactory(
    const TVMConfig& config)
    : config_(config) {
    }
}  // namespace serving
}  // namespace tensorflow
