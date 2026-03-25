
from itertools import product
import os
import sys

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
G = int(sys.argv[4])
NUM_KSPLITs = [1]

possible_split = [3, 4, 7, 8, 14, 16]
if K == 7168:
    possible_split = [7]
if K == 1536:
    possible_split = []
for a_possible_split in possible_split:
    if K % a_possible_split == 0:
        NUM_KSPLITs.append(a_possible_split)

# if M >= 1024:
#     Ms = [32, 64, 128, 256]
# elif M >= 256:
#     Ms = [32, 64, 128]
# elif M >= 32:
#     Ms = [32]

Ns = [32, 64, 128]

if K == 7168:
    Ks = [128]
if K == 1536:
    Ks = [128]

parms = {
    "BLOCK_SIZE_M": Ms,
    "BLOCK_SIZE_N": Ns,
    "BLOCK_SIZE_K": Ks,
    "GROUP_SIZE_M": [1, 4, 8],
    "num_warps": [2, 4, 8],
    "num_stages": [1, 2, 3],
    "waves_per_eu": [1, 2, 4, 6, 8],
    "matrix_instr_nonkdim": [16],
    "cache_modifier": [0, 1],
    "NUM_KSPLIT": NUM_KSPLITs,
}
comb = list(product(*parms.values()))
comb_p = []
for a_comb in comb:
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim, cache_modifier, NUM_KSPLIT = a_comb
    # skip cases
    if NUM_KSPLIT > 1 and GROUP_SIZE_M > 1:
        continue
    if BLOCK_SIZE_K > K // NUM_KSPLIT:
        continue
    if BLOCK_SIZE_K == K // NUM_KSPLIT and num_stages != 1: # k_itr == 1 case
        continue
    if BLOCK_SIZE_K < K // NUM_KSPLIT and num_stages == 1: # k_itr > 1 case
        continue
    comb_p.append(a_comb)
comb = comb_p
filename = f"screen-{M}-{N}-{K}.txt"
s = " ".join([str(v) for v in parms.keys()])
os.popen(f"echo 'Number of combinations = {len(comb)}' > {filename}")
os.popen(f"echo '{s}' >> {filename}")
for a_comb in comb:
    s = " ".join([str(v) for v in a_comb])
    os.popen(f"echo 'screencase {s}' >> {filename}").read()
    cmd=f"""HIP_VISIBLE_DEVICES={G} rocprofv3 --kernel-trace -f csv -o res-{M}-{N}-{K} -- python3 ut.py {M} {N} {K} {s} >> {filename}"""
    cmd_rprof=f"""python3 rprof.py res-{M}-{N}-{K}_kernel_trace.csv -k gemm >> {filename}"""
    os.popen(cmd).read()
    os.popen(cmd_rprof).read()