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

Requires the `hf` CLI (ships with the `huggingface_hub` Python package
1.14+; the older `huggingface-cli` binary still works on 1.13 but has
been removed in 1.14). The repo is gated, so you may need to
`hf auth login` first and accept the
[CreativeML Open RAIL-M license](https://huggingface.co/runwayml/stable-diffusion-v1-5)
on the model page.

```bash
# From the repo root. Two passes — the new `hf` CLI's --include doesn't
# expand globs the way the old `huggingface-cli` did, so we list the
# tokenizer files explicitly. The bulk safetensors fetch is in pass 1
# (it does still understand the per-file --include forms below).
mkdir -p crates/scry-diffusion/.assets/sd-1-5

hf download runwayml/stable-diffusion-v1-5 \
    --local-dir crates/scry-diffusion/.assets/sd-1-5 \
    --include 'text_encoder/model.safetensors' \
              'unet/diffusion_pytorch_model.safetensors' \
              'vae/diffusion_pytorch_model.safetensors'

hf download runwayml/stable-diffusion-v1-5 tokenizer/vocab.json \
    --local-dir crates/scry-diffusion/.assets/sd-1-5
hf download runwayml/stable-diffusion-v1-5 tokenizer/merges.txt \
    --local-dir crates/scry-diffusion/.assets/sd-1-5
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

## SD 1.5 inpainting (M11)

Layout under `crates/scry-diffusion/.assets/sd-1-5-inpainting/`:

```text
crates/scry-diffusion/.assets/sd-1-5-inpainting/
├── tokenizer/
│   ├── vocab.json                                      # ~1.1 MB
│   └── merges.txt                                      # ~513 KB
└── unet/
    └── diffusion_pytorch_model.fp16.safetensors        # ~1.7 GB (fp16)
```

The text encoder and VAE are byte-identical to the base SD 1.5 ones, so
the inpaint example points back at `sd-1-5/text_encoder/` and
`sd-1-5/vae/` rather than duplicating the bytes locally.

### Download

The original `runwayml/stable-diffusion-inpainting` HF repo was pulled in
mid-2024; the community org `stable-diffusion-v1-5/` mirrors it,
ungated:

```bash
mkdir -p crates/scry-diffusion/.assets/sd-1-5-inpainting

hf download stable-diffusion-v1-5/stable-diffusion-inpainting unet/diffusion_pytorch_model.fp16.safetensors --local-dir crates/scry-diffusion/.assets/sd-1-5-inpainting

hf download stable-diffusion-v1-5/stable-diffusion-inpainting tokenizer/vocab.json --local-dir crates/scry-diffusion/.assets/sd-1-5-inpainting
hf download stable-diffusion-v1-5/stable-diffusion-inpainting tokenizer/merges.txt --local-dir crates/scry-diffusion/.assets/sd-1-5-inpainting
```

`hf`'s positional filename form is the reliable way — `--include` with
multiple file globs silently drops the include flag on `huggingface_hub
1.13`.

### What changes vs base SD 1.5

The only structural difference is the UNet's `conv_in.weight`, which
widens from `[320, 4, 3, 3]` to `[320, 9, 3, 3]` — 4 noisy-latent
channels + 1 mask channel + 4 masked-latent channels, in that order.
Every other tensor matches the base checkpoint at 1e-7. The loader
picks up the wider shape via `UnetConfig::sd_1_5_inpainting()`; nothing
else in the model code knows or cares.

## LCM-Dreamshaper-v7 (M12 visual gate)

Full SD 1.5-compatible LCM-distilled checkpoint from
[`SimianLuo/LCM_Dreamshaper_v7`](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7).
Pairs with the base SD 1.5 tokenizer / text encoder / VAE — those three
are byte-identical between the two snapshots, so only the LCM UNet
(~3.4 GB) is fetched fresh; the example symlinks the rest from
`.assets/sd-1-5/`.

```text
crates/scry-diffusion/.assets/lcm-dreamshaper-v7/
├── tokenizer/       -> ../sd-1-5/tokenizer
├── text_encoder/    -> ../sd-1-5/text_encoder
├── vae/             -> ../sd-1-5/vae
└── unet/
    ├── config.json
    └── diffusion_pytorch_model.safetensors             # ~3.4 GB
```

### Download

```bash
mkdir -p crates/scry-diffusion/.assets/lcm-dreamshaper-v7
hf download SimianLuo/LCM_Dreamshaper_v7 \
    unet/diffusion_pytorch_model.safetensors unet/config.json \
    --local-dir crates/scry-diffusion/.assets/lcm-dreamshaper-v7

cd crates/scry-diffusion/.assets/lcm-dreamshaper-v7
ln -sf ../sd-1-5/tokenizer ./tokenizer
ln -sf ../sd-1-5/text_encoder ./text_encoder
ln -sf ../sd-1-5/vae ./vae
```

### What changes vs base SD 1.5

The UNet adds `time_embedding.cond_proj.weight` — a bias-free
`Linear(256, 320)` that projects a 256-d sinusoidal embedding of the
guidance scale `w` into the time embedding (HF's "guidance-distilled"
trick that lets LCM run at `cfg=1.0` while still behaving as if CFG were
active). Loader picks it up automatically when present; the caller wires
`w` via [`Unet::set_guidance_scale`] (LCM-Dreamshaper was distilled with
`w = guidance_scale − 1 = 7.0`). Vanilla SD checkpoints leave this key
absent and `cached_cond_emb` stays `None`, so the path is a no-op for
non-LCM models.

Trained at 768×768; runs at 512×512 too but visibly higher quality at
the trained resolution. See `examples/txt2img_lcm.rs`.

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
