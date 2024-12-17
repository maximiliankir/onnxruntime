#pragma once

#include "core/framework/execution_provider.h"

namespace onnxruntime {

class TTNNExecutionProvider : public IExecutionProvider {
 public:
    explicit TTNNExecutionProvider();
    virtual ~TTNNExecutionProvider();

    std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace onnxruntime
