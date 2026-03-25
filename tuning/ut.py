
import sys
from triton.testing import runtime
import torch
import triton
from aiter.ops.triton.gemm_afp4wfp4 import (
    gemm_afp4wfp4,
    gemm_afp4wfp4_preshuffled_scales,
    gemm_afp4wfp4_preshuffle,
)
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs, run_torch

M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

config = None
if len(sys.argv) >= 5:
    config={
        "BLOCK_SIZE_M": int(sys.argv[4]),
        "BLOCK_SIZE_N": int(sys.argv[5]),
        "BLOCK_SIZE_K": int(sys.argv[6]),
        "GROUP_SIZE_M": int(sys.argv[7]),
        "num_warps": int(sys.argv[8]),
        "num_stages": int(sys.argv[9]),
        "waves_per_eu": int(sys.argv[10]),
        "matrix_instr_nonkdim": int(sys.argv[11]),
        "cache_modifier": ".cg" if int(sys.argv[12]) == 0 else None,
        "NUM_KSPLIT": int(sys.argv[13]),
    }
    print(config)

di = runtime.driver.active.get_device_interface()
cache = runtime.driver.active.get_empty_cache_for_benchmark()
dtype=torch.bfloat16
shuffle=True
print(M, N, K)
x, w, w_triton, x_scales, w_scales, x_scales_triton, w_scales_triton, out_dtype, y = (
    generate_gemm_afp4wfp4_inputs(M, N, K, dtype, output=True, shuffle_scales_fg=shuffle, shuffle_weight_fg=shuffle)
)
torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)
for _ in range(250):
    cache.zero_()
    di.synchronize()
    if shuffle == False:
        triton_out = gemm_afp4wfp4(
            x, w_triton, x_scales_triton, w_scales_triton, dtype, y, config=config
        )
    else:
        triton_out = gemm_afp4wfp4_preshuffle(
            x, w_triton, x_scales_triton, w_scales_triton, dtype, y, config=config
        )
    di.synchronize()

torch.testing.assert_close(torch_out, triton_out)
        

                