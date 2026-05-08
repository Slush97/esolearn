# scry-diffusion test assets

The model weights and tokenizer files for Stable Diffusion 1.5 (and any
later SDXL work) live under `crates/scry-diffusion/.assets/`. The
directory is gitignored — every contributor populates it locally from
HuggingFace. None of these files are checked into the repo; only this
README and the `.gitignore` rule are.

## SD 1.5

Layout expected by the loaders (M3-M8):

```text
crates/scry-diffusion/.assets/sd-1-5/
├── tokenizer/
│   ├── vocab.json                                      # ~862 KB
│   └── merges.txt                                      # ~525 KB
├── text_encoder/
│   └── model.safetensors                               # ~492 MB (CLIP-L, fp16/fp32)
├── unet/
│   └── diffusion_pytorch_model.safetensors             # ~3.4 GB (fp16) / ~6.8 GB (fp32)
└── vae/
    └── diffusion_pytorch_model.safetensors             # ~167 MB (fp16) / ~335 MB (fp32)
```

### Download

Requires `huggingface-cli` (ships with the `huggingface_hub` Python
package). The repo is gated, so you may need to `huggingface-cli login`
first and accept the
[CreativeML Open RAIL-M license](https://huggingface.co/runwayml/stable-diffusion-v1-5)
on the model page.

```bash
# From the repo root.
mkdir -p crates/scry-diffusion/.assets/sd-1-5
huggingface-cli download runwayml/stable-diffusion-v1-5 \
    --local-dir crates/scry-diffusion/.assets/sd-1-5 \
    --include 'tokenizer/*' \
              'text_encoder/model.safetensors' \
              'unet/diffusion_pytorch_model.safetensors' \
              'vae/diffusion_pytorch_model.safetensors'
```

The full repo also ships `feature_extractor/`, `safety_checker/`,
`scheduler/`, and config JSONs — none of which we need, since we
implement the scheduler / pipeline ourselves.

### What the loaders expect

- **Tokenizer** ([`Tokenizer::from_dir`](src/tokenizer.rs)) reads
  `tokenizer/vocab.json` + `tokenizer/merges.txt`.
- **Text encoder** (M3) reads `text_encoder/model.safetensors`.
- **VAE decoder** (M4) reads `vae/diffusion_pytorch_model.safetensors`.
- **UNet** (M5/M6) reads `unet/diffusion_pytorch_model.safetensors`.

The pipeline driver (`examples/txt2img.rs`, M8) takes a single
`--root <path>` arg pointing at `.assets/sd-1-5/` and walks these
subdirectories itself.

## Numerical references

Per HANDOFF.md, each milestone validates against HF `diffusers`. Generate
references locally via `crates/scry-diffusion/bench_pytorch.py` (lands in
M9) or by hand in a Python REPL. Reference numpy dumps go under
`.assets/refs/` and stay gitignored too.

## SDXL (M10)

When SDXL work begins, mirror this structure under
`crates/scry-diffusion/.assets/sdxl-base-1.0/` from
`stabilityai/stable-diffusion-xl-base-1.0`. SDXL ships *two* text
encoders (`text_encoder/` and `text_encoder_2/`) and a different VAE
scaling factor — see `UnetConfig::sdxl_base()` and
`VaeDecoderConfig::sdxl()` for the config differences.

## License

These weights are not Apache/MIT — they are governed by the
CreativeML Open RAIL-M license that ships with the upstream HF repo.
We never ship the weights ourselves; consult the upstream model card
for terms that apply to your use of the trained outputs.
