// TODO Structure this differently with a superclass and seprarate files for each operator
// Look at the CUDA provider for an example
#include "core/providers/ttnn/ttnn_operators.h"
#include "core/providers/ttnn/ttnn_fwd.h"

namespace onnxruntime {
namespace contrib {
namespace ttnn {

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    11,
    kTtnnExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraints<float>()),
    ConvKernel);

}  // namespace ttnn
}  // namespace contrib
}  // namespace onnxruntime
