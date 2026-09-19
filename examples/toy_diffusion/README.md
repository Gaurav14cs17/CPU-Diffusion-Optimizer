# Toy Diffusion Model

Minimal deterministic DiT-like diffusion model for CPU tests and Phase 1 demos.

No pretrained weights. Fast enough for unit tests and local profiling.

```python
from examples.toy_diffusion import build_toy_model

model = build_toy_model(height=16, width=16, num_blocks=4)
out = model.sample(num_steps=8, seed=0)
print(out["output"].shape)
```
