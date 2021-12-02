#ifndef TENSORFLOW_SERVING_SERVABLES_TVM_PREDICT_IMPL_H_
#define TENSORFLOW_SERVING_SERVABLES_TVM_PREDICT_IMPL_H_

#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tvm/tvm_loader.h"
#include "tensorflow_serving/apis/predict.pb.h"

namespace tensorflow {
namespace serving {

// Utility methods for implementation of predict.
// Initialized and called from Tensorflow servable.
class TVMPredictor {
 public:
  TVMPredictor() = default;

  Status Predict(ServerCore* core, const ModelSpec& model_spec,
                 const PredictRequest& request, PredictResponse* response);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TVM_PREDICT_IMPL_H_