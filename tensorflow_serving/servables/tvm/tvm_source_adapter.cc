#include "tensorflow_serving/servables/tvm/tvm_source_adapter.h"

#include <stddef.h>
#include <memory>
#include <vector>

#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/resources/resource_util.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/servables/tvm/tvm_factory.h"

namespace tensorflow {
namespace serving {

// 构建TVMFactory
Status TVMSourceAdapter::Create(
    const TVMSourceAdapterConfig& config,
    std::unique_ptr<TVMSourceAdapter>* adapter) {
  std::unique_ptr<TVMFactory> bundle_factory;
  TF_RETURN_IF_ERROR(
      TVMFactory::Create(config.config(), &bundle_factory));
  adapter->reset(new TVMSourceAdapter(std::move(bundle_factory)));
  return Status::OK();
}

TVMSourceAdapter::~TVMSourceAdapter() { Detach(); }

// 构造函数
TVMSourceAdapter::TVMSourceAdapter(
    std::unique_ptr<TVMFactory> bundle_factory)
    : bundle_factory_(std::move(bundle_factory)) {}

// 创建TVMBundle，构建使用于tf-serving的SampleLoader
Status TVMSourceAdapter::Convert(const StoragePath& path,
                                           std::unique_ptr<Loader>* loader) {
  std::shared_ptr<TVMFactory> bundle_factory = bundle_factory_;
  auto servable_creator = [bundle_factory,
                           path](std::unique_ptr<TVMBundle>* bundle) {
    return bundle_factory->CreateTVM(path, bundle);
  };
  auto resource_estimator = [bundle_factory,
                             path](ResourceAllocation* estimate) {
    return bundle_factory->EstimateResourceRequirement(path, estimate);
  };
  loader->reset(
      new SimpleLoader<TVMBundle>(servable_creator, resource_estimator));
  return Status::OK();
}

std::function<Status(
    std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*)>
TVMSourceAdapter::GetCreator(
    const TVMSourceAdapterConfig& config) {
  return [config](std::unique_ptr<tensorflow::serving::SourceAdapter<
                      StoragePath, std::unique_ptr<Loader>>>* source) {
    std::unique_ptr<TVMSourceAdapter> typed_source;
    TF_RETURN_IF_ERROR(
        TVMSourceAdapter::Create(config, &typed_source));
    *source = std::move(typed_source);
    return Status::OK();
  };
}

// Register the source adapter.
class TVMSourceAdapterCreator {
 public:
  static Status Create(
      const TVMSourceAdapterConfig& config,
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*
          adapter) {
    std::unique_ptr<TVMFactory> bundle_factory;
    TF_RETURN_IF_ERROR(
        TVMFactory::Create(config.config(), &bundle_factory));
    adapter->reset(new TVMSourceAdapter(std::move(bundle_factory)));
    return Status::OK();
  }
};
REGISTER_STORAGE_PATH_SOURCE_ADAPTER(TVMSourceAdapterCreator,
                                     TVMSourceAdapterConfig);

}  // namespace serving
}  // namespace tensorflow
