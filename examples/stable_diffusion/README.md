# FastSDCPU / Stable Diffusion CPU baseline

Adapter: `engine/model/adapters/fastsdcpu.py` (`kind: fastsdcpu` or `openvino`).

Aligned with [FastSDCPU](https://github.com/rupeshs/fastsdcpu) **model choices**
(SD Turbo / OpenVINO exports), not a fork of that app.

```bash
pip install -e ".[fastsdcpu]"
# optional: force Diffusers CPU without OpenVINO
# export CDO_FASTSDCPU_BACKEND=pytorch

python -m cli experiment --config configs/fastsdcpu.yaml
```

Default OpenVINO id: `rupeshs/sd-turbo-openvino`. Override with
`path: "openvino:rupeshs/..."` or `path: "pytorch:stabilityai/sd-turbo"`.

Also see `engine/model/adapters/diffusers.py` for the generic Diffusers path.
