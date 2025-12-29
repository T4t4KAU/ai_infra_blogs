ARCH  ?= 120
BUILD ?= release

CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
NCU  := $(CUDA_HOME)/bin/ncu
CUTLASS := ./cutlass

BUILD_DIR := build
NCU_DIR   := ncu-reports

COMMON_FLAGS  := -std=c++17 -arch=sm_$(ARCH)
RELEASE_FLAGS := -O3 --use_fast_math
DEBUG_FLAGS   := -O3 -g -G

ifeq ($(BUILD),debug)
  NVCC_FLAGS := $(COMMON_FLAGS) $(DEBUG_FLAGS)
else
  NVCC_FLAGS := $(COMMON_FLAGS) $(RELEASE_FLAGS)
endif

# ============================================================
# Nsight Compute flags
# ============================================================

NCU_FLAGS ?= \
	--set full \
	--target-processes all \
	--launch-skip 0 \
	--launch-count 1

# ============================================================
# Build rules
# ============================================================

matmul_v1: matmul_v1.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@

matmul_v2: matmul_v2.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@

matmul_v3: matmul_v3.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@

matmul_v4: matmul_v4.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@

matmul_v5: matmul_v5.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $(BUILD_DIR)/$@

cutlass_sgemm_bench: cutlass_sgemm_bench.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -O3 -std=c++17 \
		--expt-relaxed-constexpr \
		-I$(CUTLASS)/include \
		-I$(CUTLASS)/tools/util/include \
		$< -o $(BUILD_DIR)/$@

# ============================================================
# NCU profiling rule (generic)
# ============================================================

%_ncu: %
	@mkdir -p $(NCU_DIR)
	@echo "==> Profiling $< with Nsight Compute"
	$(NCU) $(NCU_FLAGS) \
		-o $(NCU_DIR)/$< \
		$(BUILD_DIR)/$<

# ============================================================
# Clean
# ============================================================

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(NCU_DIR)
