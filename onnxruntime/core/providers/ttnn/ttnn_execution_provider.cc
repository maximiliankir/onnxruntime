#include "core/providers/ttnn/ttnn_execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

TTNNExecutionProvider::TTNNExecutionProvider() : IExecutionProvider{onnxruntime::kTtnnExecutionProvider} {

}

TTNNExecutionProvider::~TTNNExecutionProvider() {}

std::shared_ptr<KernelRegistry> TTNNExecutionProvider::GetKernelRegistry() const {
    return nullptr;
}

std::unique_ptr<IDataTransfer> TTNNExecutionProvider::GetDataTransfer() const {
    return nullptr;
}

std::vector<AllocatorPtr> TTNNExecutionProvider::CreatePreferredAllocators() {
    return {};
}

}   // namespace onnxruntime
