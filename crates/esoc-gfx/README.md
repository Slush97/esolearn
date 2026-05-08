# esoc-gfx

Renders an [`esoc-scene`](https://crates.io/crates/esoc-scene) `SceneGraph` to SVG. PNG output is available behind a feature flag via `resvg`.

This crate is the rendering backend for [`esoc-chart`](https://crates.io/crates/esoc-chart).

## Install

```toml
[dependencies]
esoc-gfx = "0.2"

# PNG output (pulls in resvg + tiny-skia)
esoc-gfx = { version = "0.2", features = ["png"] }
```

## Render a scene graph

```rust
use esoc_gfx::{render_scene_svg, save_scene_svg};
use esoc_scene::SceneGraph;

let scene: SceneGraph = build_my_scene();
let svg: String = render_scene_svg(&scene, 800.0, 600.0)?;
save_scene_svg(&scene, 800.0, 600.0, "out.svg")?;
```

With the `png` feature:

```rust
use esoc_gfx::save_scene_png;
save_scene_png(&scene, 800.0, 600.0, "out.png")?;
```

## Design notes

SVG-first because the target audience is ML/data folks producing reports — vector output renders crisply in notebooks, papers, and slide decks without resolution decisions. PNG is opt-in because pulling in `resvg` more than triples the dependency tree, and many users never need it.

When `png` is enabled, system fonts are loaded via `usvg` so the rasterized output matches the SVG.

The crate denies `unsafe_code` — everything is safe Rust.

## License

MIT OR Apache-2.0
