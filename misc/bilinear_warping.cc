#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("BilinearWarping")
  .Input("x: float")
  .Input("flow: float")
  .Output("y: float");

REGISTER_OP("BilinearWarpingGrad")
  .Input("grad_y: float")
  .Input("x: float")
  .Input("flow: float")
  .Output("grad_x: float")
  .Output("grad_flow: float");

void BilinearWarpingLauncher(const float* input, const float* flow, float* out, 
    const int count, const int channels, const int height, const int width);

class BilinearWarpingGPUOp : public OpKernel {
public:
  explicit BilinearWarpingGPUOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    const Tensor& flow_tensor = context->input(1);
    auto flow = flow_tensor.flat<float>();

    OP_REQUIRES(context, input_tensor.dims() == 4, errors::InvalidArgument("input dim != 4"));
    OP_REQUIRES(context, flow_tensor.dims() == 4, errors::InvalidArgument("flow dim != 4"));

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<float>();

    const int input_dims = input_tensor.dims();
    OP_REQUIRES(context, input_dims == 4, errors::InvalidArgument("input dim != 4"));
    OP_REQUIRES(context, flow_tensor.dims() == 4, errors::InvalidArgument("flow dim != 4"));
    OP_REQUIRES(context, flow_tensor.dim_size(0) == input_tensor.dim_size(0), errors::InvalidArgument("flow dim 0 != input dim 0"));
    OP_REQUIRES(context, flow_tensor.dim_size(1) == input_tensor.dim_size(1), errors::InvalidArgument("flow dim 1 != input dim 1"));
    OP_REQUIRES(context, flow_tensor.dim_size(2) == input_tensor.dim_size(2), errors::InvalidArgument("flow dim 2 != input dim 2"));
    OP_REQUIRES(context, flow_tensor.dim_size(3) == 2, errors::InvalidArgument("Flow dim 3 != 2"));

    const int count = input_tensor.NumElements();
    const int channels = input_tensor.dim_size(3);
    const int height = input_tensor.dim_size(1);
    const int width = input_tensor.dim_size(2);
    BilinearWarpingLauncher(input.data(), flow.data(), output.data(), count, channels, height, width);
  }
};

void BilinearWarpingGradLauncher(const float* grad_y, const float* input, 
    const float* flow, float* grad_x, float* grad_flow, 
    const int count, const int channels, const int height, const int width);

class BilinearWarpingGradGPUOp : public OpKernel {
public:
  explicit BilinearWarpingGradGPUOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_y_tensor = context->input(0);
    auto grad_y = grad_y_tensor.flat<float>();

    const Tensor& input_tensor = context->input(1);
    auto input = input_tensor.flat<float>();

    const Tensor& flow_tensor = context->input(2);
    auto flow = flow_tensor.flat<float>();

    OP_REQUIRES(context, input_tensor.dims() == 4, errors::InvalidArgument("input dim != 4"));
    OP_REQUIRES(context, flow_tensor.dims() == 4, errors::InvalidArgument("flow dim != 4"));

    Tensor* grad_x_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &grad_x_tensor));
    auto grad_x = grad_x_tensor->flat<float>();

    Tensor* grad_flow_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, flow_tensor.shape(), &grad_flow_tensor));
    auto grad_flow = grad_flow_tensor->flat<float>();

    const int input_dims = input_tensor.dims();
    OP_REQUIRES(context, input_dims == 4, errors::InvalidArgument("input dim != 4"));
    OP_REQUIRES(context, flow_tensor.dims() == 4, errors::InvalidArgument("flow dim != 4"));
    OP_REQUIRES(context, flow_tensor.dim_size(0) == input_tensor.dim_size(0), errors::InvalidArgument("flow dim 0 != input dim 0"));
    OP_REQUIRES(context, flow_tensor.dim_size(1) == input_tensor.dim_size(1), errors::InvalidArgument("flow dim 1 != input dim 1"));
    OP_REQUIRES(context, flow_tensor.dim_size(2) == input_tensor.dim_size(2), errors::InvalidArgument("flow dim 2 != input dim 2"));
    OP_REQUIRES(context, flow_tensor.dim_size(3) == 2, errors::InvalidArgument("Flow dim 3 != 2"));

    const int count = input_tensor.NumElements();
    const int channels = input_tensor.dim_size(3);
    const int height = input_tensor.dim_size(1);
    const int width = input_tensor.dim_size(2);
    BilinearWarpingGradLauncher(grad_y.data(), input.data(), flow.data(), grad_x.data(), grad_flow.data(), count, channels, height, width);
  }
};

REGISTER_KERNEL_BUILDER(Name("BilinearWarping").Device(DEVICE_GPU), BilinearWarpingGPUOp);
REGISTER_KERNEL_BUILDER(Name("BilinearWarpingGrad").Device(DEVICE_GPU), BilinearWarpingGradGPUOp);