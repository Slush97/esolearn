# esoc-color

Perceptual color math for data visualization. OKLab/OKLCH conversions, palettes, color scales, color-vision-deficiency simulation, WCAG contrast, and gamut clipping. Zero external dependencies.

The core type is `Color` — `#[repr(C)]` linear RGBA in f32, designed to upload directly to GPU buffers without conversion. All perceptual operations (mixing, gradients, distance) happen in OKLab space, so blends look the way humans perceive them rather than the way sRGB averages them.

## Install

```toml
[dependencies]
esoc-color = "0.1"
```

## What's in here

- **OKLab / OKLCH** — round-trip conversions following Björn Ottosson's reference matrices.
- **Palettes** — built-in `tab10` (categorical), `viridis` (sequential), `rdbu` (diverging), plus `sequential` / `diverging` / `categorical` builders for custom maps.
- **`ColorScale`** — `[0, 1] → Color` lookup with helpers for generating GPU 1D textures.
- **CVD simulation** — protanopia, deuteranopia, tritanopia (Viénot/Brettel/Mollon matrices), with severity interpolation.
- **WCAG contrast** — `contrast_ratio`, `meets_aa`, `meets_aaa`, and a `text_color_on(bg)` helper that picks black or white for legibility.
- **Gamut clipping** — keep OKLCH colors inside the sRGB gamut without hue shifts.

## Example

```rust
use esoc_color::{Color, ColorScale};
use esoc_color::contrast::text_color_on;

let scale = ColorScale::viridis();
let mid = scale.map(0.5);

let bg = Color::from_srgb8(0x1f, 0x77, 0xb4);
let label = text_color_on(bg); // Color::WHITE or Color::BLACK
```

## Design notes

Linear RGB is the storage format because it's what shaders want. The sRGB encode/decode helpers are explicit (`Color::from_srgb8`, `Color::to_srgb8`) so it's always obvious where the gamma boundary is. `Palette::sample` interpolates in OKLab, not RGB — straight-line interpolation in linear or sRGB produces muddy mid-tones, which is why most matplotlib-style colormaps look wrong when naively lerped.

## License

MIT OR Apache-2.0
