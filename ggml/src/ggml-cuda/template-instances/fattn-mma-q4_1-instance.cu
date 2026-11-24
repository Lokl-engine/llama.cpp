// MMA FlashAttention kernels that read a q4_1 KV cache directly, see GGML_CUDA_FATTN_MMA_KV_TYPE in fattn-mma-f16.cuh.
// The list of cases must match DECL_FATTN_MMA_Q4_CASE in fattn.cu.

#include "../common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#define GGML_CUDA_FATTN_MMA_KV_TYPE GGML_TYPE_Q4_1
#define ggml_cuda_flash_attn_ext_mma_f16_case ggml_cuda_flash_attn_ext_mma_q4_1_case

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_F16_CASE(256, 256, 1, 8); // GQA 5-8, e.g. Qwen3.x 27B
DECL_FATTN_MMA_F16_CASE(512, 512, 1, 8); // GQA 5-8, e.g. the global layers of Gemma 4 31B

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
