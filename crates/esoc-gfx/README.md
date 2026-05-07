# esoc-gfx

Low-level 2D vector graphics for the esoc charting stack. Outputs SVG by default with zero non-workspace dependencies; PNG output is available behind a feature flag via `resvg`.

The crate has two entry points. The high-level path consumes an `esoc_scene::SceneGraph` and emits SVG (`render_scene_svg`) — this is what `esoc-chart` uses. The lower-level `Canvas` API lets you push individual draw elements with explicit layers, gradients, and clip paths if you need finer control than the scene graph offers.

## Install

```toml
[dependencies]
esoc-gfx = "0.1"

# PNG output (pulls in resvg + tiny-skia)
esoc-gfx = { version = "0.1", features = ["png"] }
```

## Render a scene graph

```rust
use esoc_gfx::scene_svg::{render_scene_svg, save_scene_svg};
use esoc_scene::SceneGraph;

let scene: SceneGraph = build_my_scene();
let svg: String = render_scene_svg(&scene, 800.0, 600.0)?;
save_scene_svg(&scene, 800.0, 600.0, "out.svg")?;
```

With the `png` feature:

```rust
use esoc_gfx::scene_svg::save_scene_png;
save_scene_png(&scene, 800.0, 600.0, "out.png")?;
```

## Canvas API

```rust
use esoc_gfx::prelude::*;

let mut canvas = Canvas::new(400.0, 300.0);
// push DrawElements, gradients, clips...
let svg = render_svg(&canvas)?;
save_svg(&canvas, "out.svg")?;
```

## Design notes

SVG-first because the target audience is ML/data folks producing reports — vector output renders crisply in notebooks, papers, and slide decks without resolution decisions. PNG is opt-in because pulling in `resvg` more than triples the dependency tree, and many users never need it.

Text measurement uses `HeuristicTextMeasurer` by default (font-metric estimates, no font files). When `png` is enabled, system fonts are loaded via `usvg` so the rasterized output matches the SVG.

The crate denies `unsafe_code` — everything is safe Rust.

## License

MIT OR Apache-2.0
