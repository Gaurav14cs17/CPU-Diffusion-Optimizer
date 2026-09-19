# CPU Optimization Notes

Diffusion inference on CPU is often **memory-bandwidth bound**, not just FLOP bound.

Considerations designed into this project:

- x86 AVX2 / AVX-512 detection
- ARM NEON / ASIMD detection where practical
- Peak RAM and RSS tracking
- Tensor copies and layout
- Cache locality (prefer reuse when read is cheaper than recompute)
- Threading (`torch.set_num_threads`, OMP)
- Operator fusion (planned)
- INT8 (planned), INT4 (future)

## Quantization path

```
FP32 → FP16/BF16 (when CPU supports) → INT8 → INT4 (future)
```

Unsafe fake quantization that does not change kernels is not treated as a real optimization.

## Benchmark methodology

Every claim must come from measured runs:

1. Warmup
2. Repeated timed measurements
3. Quality metrics (MSE / PSNR / optional SSIM)
4. Accept only if quality gates pass
