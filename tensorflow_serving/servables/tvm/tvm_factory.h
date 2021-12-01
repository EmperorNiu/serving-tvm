#ifndef TENSORFLOW_SERVING_SERVABLES_TVM_TVM_FACTORY_H_
#define TENSORFLOW_SERVING_SERVABLES_TVM_TVM_FACTORY_H_

#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tvm/tvm_config.pb.h"
#include "tensorflow_serving/servables/tvm/tvm_loader.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {

// A factory that creates TVMBundle from export paths.
//
// TVMBundle holds the necessary objects of TVM runtime required for inference.
//
// The factory can also estimate the resource (e.g. RAM) requirements of a
// TVMBundle based on the export.
//
// This class is thread-safe.
class TVMFactory {
 public:
  static Status Create(const TVMConfig& config,
                       std::unique_ptr<TVMFactory>* factory);

  // Instantiates a bundle from a given export path.
  Status CreateTVM(const string& path,
                             std::unique_ptr<TVMBundle>* bundle);

  // Estimates the resources a session bundle will use once loaded, from its
  // export path.
  Status EstimateResourceRequirement(const string& path,
                                     ResourceAllocation* estimate) const;

 private:

  TVMFactory(const TVMConfig& config);
  const TVMConfig config_;

  TF_DISALLOW_COPY_AND_ASSIGN(TVMFactory);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TVM_TVM_FACTORY_H_