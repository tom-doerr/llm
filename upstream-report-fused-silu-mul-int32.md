# Draft bug report (NOT filed) — vLLM: int32 overflow in fused silu_and_mul block-quant kernel

**Title:** `[Bug] silu_and_mul_per_block_quant`: int32 token-offset overflow → illegal memory
access above ~2^31/(2*intermediate_size) batched tokens (Qwen3.8-27B-FP8, `--max-num-batched-tokens 65536`)

**Environment:** vLLM 0.26.1rc1.dev1105+g040700aaa (main, Aug 22 2026), torch 2.13.0+cu130,
NVIDIA DGX Spark GB10 (sm_121a, aarch64, 128 GB unified memory), model `Qwen/Qwen3.8-27B-FP8`
(hidden 5120, intermediate 17408, FP8 block 128×128), TP=1, flashinfer, prefix caching, MTP.

**Symptom:** startup dies deterministically in memory profiling with
`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`
(`profile_run` → `_dummy_run` → inductor autotune `benchmark_all_configs` → `synchronize`);
kernel log shows `NVRM: Xid 31 ... MMU Fault ... FAULT_PTE ACCESS_TYPE_VIRT_READ`.
Reproduces on every start with `--max-num-batched-tokens 65536`; never with 32768.

**Analysis:** the reported inductor kernel is only the sync point. The custom fusion pass
`act_quant` (log: `Enabled custom fusions: norm_quant, act_quant`) replaces SiluAndMul +
block quant with `torch.ops._C.silu_and_mul_per_block_quant`
(`csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu`), whose
token offsets are computed in 32-bit:

    int const token_idx = blockIdx.x;
    int const input_stride = hidden_size * 2;            // 34816 for this model
    scalar_t const* token_input_gate = input + token_idx * input_stride + group_start;

`token_idx * input_stride` overflows int32 at token 61,681 (2^31-1 / 34816), so a 65,536-token
dummy run reads from a wrapped negative offset. The unfused `silu_and_mul`
(`activation_kernels.cu`) and the `norm_quant` path (`layernorm_utils.cuh`,
`fused_layernorm_dynamic_per_token_quant.cu`) already use `int64_t` offsets — only this fused
kernel does not.

**Confirmed by experiment (Aug 22 2026, this box):**
| max-num-batched-tokens | fusions | result |
|---|---|---|
| 65536 | norm_quant + act_quant (default) | Xid 31, illegal read, engine dies |
| 65536 | `-cc.pass_config.fuse_act_quant=false` | **starts and serves normally** |
| 32768 | default (both fusions) | starts and serves normally |

Disabling exactly the suspected pass fixes it; disabling nothing else was needed.

**Suggested fix:** cast to `int64_t` in `fused_silu_mul_block_quant.cu` (as in the sibling
kernels), e.g. `int64_t const token_offset = static_cast<int64_t>(token_idx) * input_stride;`.
Precedent: the same class of bug was fixed in vLLM's Mamba kernel (PR #35275).

**Possibly the same root cause:** vllm#13824 (closed stale) — IMA in `per_token_group_quant_fp8`
above ~40k batched tokens on DeepSeek-R1.
