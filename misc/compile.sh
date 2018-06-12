# compile the bilinear warping operator

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

nvcc -std=c++11 -c -o bilinear_warping.cu.o bilinear_warping.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

g++ -std=c++11 -shared -o bilinear_warping.so bilinear_warping.cc \
	bilinear_warping.cu.o -I $TF_INC -fPIC -L  /usr/local/cuda/lib64/ -lcudart
