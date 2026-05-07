# esoc-scene

Arena-allocated scene graph with typed visual marks. This is the intermediate representation that sits between chart logic (which decides *what* to draw) and rendering backends (which decide *how*). Producers build a `SceneGraph`; backends like `esoc-gfx` walk it.

## Install

```toml
[dependencies]
esoc-scene = "0.1"
```

## Marks

Nine primitive mark types cover the data-visualization vocabulary:

`LineMark`, `RectMark`, `PointMark`, `AreaMark`, `TextMark`, `ArcMark`, `RuleMark`, `PathMark`, `ImageMark`.

A `Mark` enum wraps any one of them; `MarkBatch` holds a homogeneous run for instanced rendering. Each batch attribute can be a single value or per-instance — the backend decides whether to expand or instance.

## Nodes and the arena

`SceneGraph` is a generational-index arena. `NodeId` carries an index plus generation counter, so freed slots can't be confused with reused ones (ABA safety). Nodes have a parent, children, local `Affine2D` transform, opacity, z-order, optional clip rect, and either `Container`, `Mark`, or `Batch` content.

Optional `MarkKey` per node identifies marks across scene diffs — useful when an animation layer needs enter/update/exit sets between frames.

## Why an arena?

Tree-of-`Box`/`Rc` scene graphs fragment memory and make traversal cache-unfriendly. An arena keeps nodes contiguous, lets `NodeId` be `Copy`, and gives O(1) insert/remove without unsafe code or refcounting.

## Example

```rust
use esoc_scene::{SceneGraph, Node, NodeContent, Mark};
use esoc_scene::mark::{PointMark, MarkerShape};

let mut sg = SceneGraph::with_root();
let root = sg.root().unwrap();

let dot = sg.insert(Node {
    parent: Some(root),
    children: vec![],
    transform: Default::default(),
    clip: false,
    z_order: 0,
    opacity: 1.0,
    content: NodeContent::Mark(Mark::Point(PointMark {
        center: [100.0, 50.0],
        size: 6.0,
        shape: MarkerShape::Circle,
        fill: Default::default(),
        stroke: Default::default(),
    })),
    key: None,
});
sg.add_child(root, dot);
```

## License

MIT OR Apache-2.0
