# GPU-Accelerated Chart Rendering — Deep Research

## Reference Implementations

### Datoviz (Vulkan, C/Python)
- **Layered architecture**: vklite (Vulkan wrapper) → Renderer → Intermediate Protocol → Scene API
- **Visual pipeline**: Array (CPU) → Dual (CPU↔GPU link) → Baker (preprocess to vertex buffers) → Visual → Shader
- **MSDF text**: Multi-channel Signed Distance Fields via msdf-atlas-gen for high-quality GPU text
- **Batch rendering**: Collections of similar marks (points, markers, glyphs) batched for GPU performance
- **Custom memory allocator**: Few large shared buffers with sub-region management (avoids many small Vulkan buffers)
- **Request-based architecture**: All GPU ops are `DvzRequest` structs processed by renderer — enables batching and async
- **Shaders by Nicolas Rougier**: Active research in GPU-accelerated graphical primitive rendering

### Avenger (Rust, wgpu)
- Vega-compatible renderer in Rust using wgpu
- Defines 2D scenegraph representation tailored to InfoVis
- Crates: avenger-scenegraph, avenger-wgpu, avenger-scales, avenger-guides, avenger-geometry
- Significantly faster than Canvas renderer for marks with many instances (large scatter plots)
- Bottleneck is often Vega scenegraph generation, not rendering

### re_renderer (Rust, wgpu) — from Rerun
- **Three primitives**: Lines (LineDrawableBuilder), Points (PointCloudBuilder), Meshes
- **DrawPhaseManager**: Transparent phases sort far-to-near; opaque phases bundle by DrawData type
- **Resource management**: CpuWriteGpuReadBelt for bulk buffer writes, pipeline/shader pooling
- **PickingLayerProcessor**: GPU-side picking/hit testing with readback
- Uses WGSL shaders, live-reloadable in debug mode

### ChartGPU (WebGPU, TypeScript)
- 35M points at ~72fps in benchmark mode
- **LTTB downsampling as compute shader** — decimation runs on GPU
- **GPU-accelerated tooltips and hit testing**
- Smooth pan/zoom at 60fps with 1M data points

## GPU Rendering Techniques for Chart Elements

### Lines (Instanced Quads)
From Tyro's instanced line rendering (https://wwwtyro.net/2019/11/18/instanced-lines.html):

**Geometry**: Each line segment = 2 triangles (6 vertices) forming a quad
- Instance geometry: (0, -0.5) to (1, 0.5)
- X-component = distance along segment, Y = distance along width

**Vertex shader math**:
```
xBasis = normalize(pointB - pointA)  // line direction
yBasis = vec2(-xBasis.y, xBasis.x)  // perpendicular
position = pointA + xBasis * vertex.x + yBasis * width * vertex.y
```

**Join types**:
- Miter: `normalize(normalize(C-B) + normalize(B-A))`, two triangles
- Round: `GL_TRIANGLE_FAN` with 16-wedge circle at joints
- Bevel: Single triangle from perpendiculars of adjacent segments

**End caps**: Round (circle at endpoints), Square (oriented rectangles)

**Buffer layout**: Interleaved `[x0,y0,x1,y1,x2,y2...]` with offset/stride
- Separate x/y buffers allow reusing x-coords across multiple y datasets (multi-line plots)
- Instance divisor=0 for geometry, divisor=1 for per-instance data

**Screen-space projection** (for consistent width regardless of depth):
```
screenPos = resolution * (0.5 * clipCoord / clipCoord.w + 0.5)
// Expand line in screen space after perspective division
```

### Nicolas Rougier's GLSL Antialiased Rendering
(JCGT 2013: "Shader-Based Antialiased, Dashed, Stroked Polylines")

**Approach**: Tessellate polyline into triangles on CPU, use signed distance fields in fragment shader for:
- Antialiasing via distance-to-edge (smooth alpha falloff)
- Dash patterns (distance along line parameterization)
- Line joins and caps (SDF composition)

This is what Datoviz uses internally. The SDF approach gives pixel-perfect AA at any zoom level.

### Scatter Plots (Instanced Quads / Point Sprites)
- Each point = instanced quad with per-instance position, color, size
- Fragment shader computes circle SDF: `distance(fragCoord, center) - radius`
- Discard fragments outside circle, smooth alpha for AA
- For millions of points: use compute shader for frustum culling / LOD

### Text on GPU
**Multi-channel Signed Distance Fields (MSDF)**:
- Pre-rasterize font glyphs into MSDF textures using msdfgen
- 3-channel (RGB) SDF encodes sharp corners that single-channel SDF loses
- Fragment shader: `median(r, g, b)` → distance → smooth step for alpha
- Sharp at any scale, single texture atlas for all sizes
- esocidae already uses glyph atlas (ShelfAllocator) — can extend for chart text

### Transparency / Blending
**Weighted Blended OIT** (McGuire & Bavoil 2013):
- Single geometry pass, no depth sorting required
- Additive blend with weight function that falls off with depth
- Good enough for charts (overlapping translucent areas, scatter with alpha)
- Much simpler than depth peeling or linked lists

## Data Pipeline to GPU

### Buffer Strategies
**Frequency-based selection**:
- USAGE_STATIC: Constant data (grid lines, axis marks)
- USAGE_DEFAULT + UpdateData: Occasional updates (theme change)
- USAGE_DYNAMIC + mapping: Every-frame updates (streaming data, animations)

**wgpu simplification**: `queue.write_buffer()` handles synchronization automatically — no manual ring buffers needed (unlike raw Vulkan)

**Streaming buffer pattern**:
1. Oversized dynamic buffer
2. Write at current offset with MAP_FLAG_DO_NOT_SYNCHRONIZE
3. When full, MAP_FLAG_DISCARD and reset offset

### Level-of-Detail / Decimation
**Largest-Triangle-Three-Buckets (LTTB)** — Sveinn Steinarsson 2013:
- Preserves visual shape better than min/max or every-Nth-point
- Divide data into buckets, select point in each bucket that forms largest triangle with neighbors
- ChartGPU runs this as a **compute shader on GPU**
- O(n) complexity, can run at 60fps for millions of points

**Min-max decimation**: For each pixel column, compute min and max of data points — produces 2 values per pixel. Simple but loses shape detail between extremes.

### Spatial Indexing for Interaction
**GPU picking** (re_renderer approach):
- Render scene to offscreen "picking" texture with object IDs as colors
- Read back pixel under cursor → O(1) hit test
- No CPU-side spatial index needed

**CPU-side alternatives**:
- R-tree / k-d tree for point queries
- Grid-based spatial hash for uniform density data

## Hybrid Rendering Architecture

**Key insight**: Charts have two types of content:
1. **Data-heavy** (scatter points, lines, areas) → GPU excels
2. **Text-heavy** (labels, annotations, legends, titles) → CPU/vector excels

### Approach: GPU data layer + CPU annotation layer
1. GPU renders data marks to offscreen texture (or directly to framebuffer)
2. CPU computes label positions (anti-collision), renders text via MSDF atlas
3. Composite in final pass

### Alternatively: Full GPU with MSDF text
- esocidae already does this for terminal text
- Use glyph atlas for chart labels too
- Limitation: harder to get pixel-perfect kerning, no complex text layout (RTL, ligatures)

## Relevance to esocidae + esoc-chart

esocidae already has:
- Arena scene graph with flat `Vec<Option<Node>>` + free list
- Instanced quad rendering (`QuadInstance`, `#[repr(C)]`, Pod)
- Custom shader pipelines (WGSL via `PipelineRegistry`)
- Damage tracking (avoid redundant GPU uploads)
- Dual atlas allocators (ShelfAllocator for glyphs, SlabAllocator for images)
- Offscreen post-processing passes

**What needs to be added for charting**:
- Line rendering pipeline (instanced quads with miter/round joins)
- SDF-based markers (circle, square, triangle, diamond, cross, star)
- Area fill rendering (tessellated polygons via Lyon crate)
- Axis/grid rendering (thin line instances)
- Color scale textures (1D texture lookup for heatmaps/contours)
- Picking/hit-test render pass (color-encoded offscreen framebuffer)
- LTTB compute shader for large dataset decimation

## Additional Findings

### Area Fills — Lyon (Rust CPU Tessellation)
- `lyon` crate: CPU-side path tessellation → triangle mesh for GPU
- Outputs vertex + index buffer consumed directly by wgpu
- Fill tessellator 2x faster than libtess2
- For chart areas: tessellate once (top line + baseline → triangles), upload, single draw call
- Used by ggez and other Rust graphics projects

### SDF Marker Shapes (Fragment Shader)
From Rougier's research (HAL hal-01081592):
- **CSG composition**: Union=`min(d1,d2)`, Difference=`max(d1,-d2)`, Intersection=`max(d1,d2)`
- Circle: `length(P) - radius`
- Square: `max(abs(P.x), abs(P.y)) - size`
- Diamond: `abs(P.x) + abs(P.y) - size`
- Triangle, cross, star, plus: composable from CSG ops
- AA via `smoothstep` on signed distance, ~1px edge

### Anti-Aliasing Hierarchy for Charts
1. **SDF per-primitive** (best for charts): Fragment shader smoothstep on signed distance. Pixel-perfect for lines, markers. What Datoviz/GLMakie use.
2. **FXAA post-process**: Cheap but blurs text. GLMakie uses as supplement.
3. **MSAA**: Hardware multi-sample. Expensive for fill-rate-bound scenes. Overkill for 2D charts.

**Recommendation**: SDF-based per-primitive AA is strongly preferred for charting.

### Almar Klein's Line Rendering (pygfx/wgpu)
- Triangle strip topology, 6 virtual vertices per node pair
- Fragment shader converts to `dist_to_stroke` (signed distance)
- AA: 1px wide edge at stroke boundary, `alpha²` as pragmatic trick for thin lines
- Dashing: cumulative distances computed on CPU, uploaded, fragment shader uses modulo

### GPU Picking (deck.gl / Rerun pattern)
- Render to offscreen framebuffer with object IDs encoded as colors
- `R + G×256 + B×65536 = object_id`
- Read single pixel under cursor → O(1) hit test
- Works with millions of elements
- ChartGPU runs hit testing as compute shader instead

### Recommended Component Selection

| Component | Technique | Reference |
|---|---|---|
| Lines | Instanced quads, screen-space expand, SDF AA fragment | regl-gpu-lines, Almar Klein |
| Area fills | Lyon CPU tessellation → GPU triangle mesh | Lyon crate |
| Scatter | Instanced billboard quads + SDF markers in fragment | Datoviz, GLMakie |
| Text | MSDF glyph atlas (msdf-atlas-gen) + instanced quads | Datoviz, msdfgen |
| Decimation | LTTB as compute shader | ChartGPU |
| Picking | Color-encoded offscreen framebuffer | Rerun, deck.gl |
| Transparency | Weighted Blended OIT | GLMakie |
| Buffers | Double-buffered, streaming with partial updates | Vulkan best practices |
| Scene graph | Canvas > Figure > Panel > Visual | Datoviz, Makie |
