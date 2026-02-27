NVCC ?= nvcc
ARCH ?= -arch=sm_89
CXX = g++
CXXFLAGS ?= -O3 -std=c++17

all: spmm_baseline spmm_opt

# Removed spmm_ref.cpp from the command line because it is already #included in the .cu file
spmm_baseline: spmm_baseline.cu
	$(NVCC) $(ARCH) $(CXXFLAGS) spmm_baseline.cu -o spmm_baseline

# Removed spmm_ref.cpp from the command line
spmm_opt: spmm_opt.cu
	$(NVCC) $(ARCH) $(CXXFLAGS) spmm_opt.cu -o spmm_opt

clean:
	rm -f spmm_baseline spmm_opt