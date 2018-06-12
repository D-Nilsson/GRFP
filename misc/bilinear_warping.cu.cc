#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void set_zero(const int N, float* in) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < N) {
    in[index] = 0.;
  }
}

__global__ void kernel_warping_bilinear_forward(const int N, int channels, int width, 
    int height, const float* in, const float* flow, float* out) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < N) {
    int c = index % channels;
    int w = (index / channels) % width;
    int h = (index / channels / width) % height;
    int n = index / channels / width / height;
    
    float flow_i = flow[((n*height + h)*width + w)*2 + 0];
    float flow_j = flow[((n*height + h)*width + w)*2 + 1];
    
    int s_low = ceil(h + flow_i - 1);
    int s_high = s_low + 1;
    int t_low = ceil(w + flow_j - 1);
    int t_high = t_low + 1;

    out[index] = 0.;
    for(int s=s_low; s<=s_high; s++) {
      for(int t=t_low; t<=t_high; t++) {
        if(s < 0 || s >= height || t < 0 || t >= width)
          continue;

        float di = h + flow_i - s;
        float dj = w + flow_j - t;
        
        float weight = (1. - abs(di)) * (1 - abs(dj));
        out[index] += weight*in[((n*height + s)*width + t)*channels + c];
      }
    }
  }
}

__global__ void kernel_warping_bilinear_backward(const int N, int channels, int width, 
    int height, const float* grad_y, const float* input, 
    const float* flow, float* grad_x, float* grad_flow) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < N) {
    int c = index % channels;
    int w = (index / channels) % width;
    int h = (index / channels / width) % height;
    int n = index / channels / width / height;
    
    float flow_i = flow[((n*height + h)*width + w)*2 + 0];
    float flow_j = flow[((n*height + h)*width + w)*2 + 1];
    
    int s_low = ceil(h + flow_i - 1);
    int s_high = s_low + 1;
    int t_low = ceil(w + flow_j - 1);
    int t_high = t_low + 1;

    bool prop_to_data = true, prop_to_flow = true;
    
    for(int s=s_low; s<=s_high; s++) {
      for(int t=t_low; t<=t_high; t++) {
        if(s < 0 || s >= height || t < 0 || t >= width)
          continue;
        float di = h + flow_i - s;
        float dj = w + flow_j - t;
        
        float weight = (1-fabs(di))*(1-fabs(dj));
        
        if(prop_to_data) {
          atomicAdd(&grad_x[((n*height + s)*width + t)*channels + c], 
            grad_y[((n*height + h)*width + w)*channels + c]*weight);
        }
        if(prop_to_flow) {
          float weight_0 = (dj > 0) ? -(1-fabs(di)) : (1-fabs(di));
          atomicAdd(&grad_flow[((n*height + h)*width + w)*2 + 1],
            grad_y[((n*height + h)*width + w)*channels + c]*
            input[((n*height + s)*width + t)*channels + c]*
            weight_0);

          float weight_1 = (di > 0) ? -(1-fabs(dj)) : (1-fabs(dj));
          atomicAdd(&grad_flow[((n*height + h)*width + w)*2 + 0],
            grad_y[((n*height + h)*width + w)*channels + c]*
            input[((n*height + s)*width + t)*channels + c]*
            weight_1);
        }
      }
    }
  }
}

void BilinearWarpingLauncher(const float* input, const float* flow, float* out, 
    const int count, const int channels, const int height, const int width) {
  
  const int kThreadsPerBlock = 1024;
  kernel_warping_bilinear_forward<<<(count + kThreadsPerBlock - 1)/kThreadsPerBlock,
      kThreadsPerBlock>>>(count, channels, width, height,
        input, flow, out);
}

void BilinearWarpingGradLauncher(const float* grad_y, const float* input, 
    const float* flow, float* grad_x, float* grad_flow, 
    const int count, const int channels, const int height, const int width) {
  
  const int kThreadsPerBlock = 1024;
  set_zero<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock,
          kThreadsPerBlock>>>(count, grad_x);

  set_zero<<<(count / channels * 2 + kThreadsPerBlock - 1) / kThreadsPerBlock,
          kThreadsPerBlock>>>(count / channels * 2, grad_flow);


  kernel_warping_bilinear_backward<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock,
          kThreadsPerBlock>>>(count, channels, width, 
    height, grad_y,  input, flow, grad_x, grad_flow);
}


#endif