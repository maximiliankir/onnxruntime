// TODO Structure this differently with a superclass and seprarate files for each operator
// Look at the CUDA provider for an example
#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include <iostream>


namespace onnxruntime {
namespace contrib {
namespace ttnn {

class ConvKernel : public OpKernel {
 public:
  explicit ConvKernel(const OpKernelInfo& info) : OpKernel(info) {

  }

  Status Compute(OpKernelContext* context) const override {
    context = context;

    std::cout << "ConvKernel::Compute" << std::endl;
    // TODO: Implement the Conv operator here

    return Status::OK();

  }

};

}  // namespace ttnn
}  // namespace contrib
}  // namespace onnxruntime
