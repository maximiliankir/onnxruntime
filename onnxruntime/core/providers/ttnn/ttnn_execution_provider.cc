#include "core/providers/ttnn/ttnn_execution_provider.h"

#include "core/common/logging/logging.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/graph/constants.h"
#include "core/providers/ttnn/ttnn_operators.h"
#include "core/providers/ttnn/ttnn_fwd.h"

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  onnxruntime::Status st;
};
}  // namespace

namespace onnxruntime {

TTNNExecutionProvider::TTNNExecutionProvider() : IExecutionProvider{onnxruntime::kTtnnExecutionProvider} {

}

TTNNExecutionProvider::~TTNNExecutionProvider() {}

namespace contrib {
namespace ttnn {

// Forward declarations of op kernels
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTtnnExecutionProvider, kOnnxDomain, 11, Conv);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterTTNNKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTtnnExecutionProvider, kOnnxDomain, 11, Conv)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

}  // namespace ttnn
}  // namespace contrib

static KernelRegistryAndStatus GetTTNNKernelRegistry() {
    KernelRegistryAndStatus ret;
    // Register the TTNN kernels
    ret.st = ::onnxruntime::contrib::ttnn::RegisterTTNNKernels(*ret.kernel_registry);
    ORT_THROW_IF_ERROR(ret.st);
    return ret;
}

std::shared_ptr<KernelRegistry> TTNNExecutionProvider::GetKernelRegistry() const {
    static KernelRegistryAndStatus ret = GetTTNNKernelRegistry();
    // throw if the registry failed to initialize
    ORT_THROW_IF_ERROR(ret.st);
    return ret.kernel_registry;
}

}   // namespace onnxruntime
