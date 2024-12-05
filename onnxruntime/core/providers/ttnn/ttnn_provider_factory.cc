#include "core/providers/ttnn/ttnn_provider_factory.h"

#include <memory>

#include "core/providers/ttnn/ttnn_execution_provider.h"
#include "core/providers/ttnn/ttnn_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct TTNNProviderFactory : IExecutionProviderFactory {
    TTNNProviderFactory() = default;
    ~TTNNProviderFactory() override = default;
    std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> TTNNProviderFactory::CreateProvider() {
    return std::make_unique<TTNNExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory> TTNNProviderFactoryCreator::Create() {
    return std::make_shared<TTNNProviderFactory>();
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_TTNN, _In_ OrtSessionOptions* options) {
    options->provider_factories.push_back(std::make_shared<onnxruntime::TTNNProviderFactory>());
    return nullptr;
}
