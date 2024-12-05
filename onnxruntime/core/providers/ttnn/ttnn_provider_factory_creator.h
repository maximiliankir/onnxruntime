#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
struct TTNNProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create();
};
}  // namespace onnxruntime
