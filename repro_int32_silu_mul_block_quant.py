"""Standalone repro: int32 token-offset overflow in silu_and_mul_per_block_quant.

Grid is (num_tokens, num_groups); the kernel computes the input pointer as
    int const token_idx = blockIdx.x;
    int const input_stride = hidden_size * 2;
    input + token_idx * input_stride + group_start;      // all int32
so `token_idx * input_stride` wraps once it exceeds 2**31-1.
"""
import sys, torch, vllm  # noqa: F401  (importing vllm registers torch.ops._C)

op = torch.ops._C.silu_and_mul_per_block_quant
print("schema:", op.default._schema, flush=True)

H = 17408          # Qwen3.8-27B intermediate_size; input last dim = 2*H = 34816
G = 128            # block quant group size
STRIDE = 2 * H
MAX_IDX=(2**31-1)//STRIDE
print(f"hidden={H} input_stride={STRIDE}: last int32-safe token_idx = {MAX_IDX} "
      f"-> last safe num_tokens = {MAX_IDX+1}", flush=True)

def run(n_tokens):
    inp = torch.randn(n_tokens, STRIDE, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(n_tokens, H, dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty(n_tokens, H // G, dtype=torch.float32, device="cuda")
    top = (n_tokens - 1) * STRIDE
    print(f"\n--- num_tokens={n_tokens}: max token_idx*input_stride = {top:,} "
          f"({'fits' if top <= 2**31-1 else 'OVERFLOWS'} int32 [max 2,147,483,647])", flush=True)
    try:
        op(out, inp, scales, G, None, False)
        torch.cuda.synchronize()
        print(f"    OK  out.isfinite? {out.to(torch.float32).isfinite().all().item()}", flush=True)
        return True
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e).splitlines()[0][:140]}", flush=True)
        return False
    finally:
        del inp, out, scales
        torch.cuda.empty_cache()

for n in [int(x) for x in sys.argv[1:]] or [61681, 61682]:
    if not run(n):
        break   # CUDA context is poisoned after an IMA; nothing after it is meaningful
