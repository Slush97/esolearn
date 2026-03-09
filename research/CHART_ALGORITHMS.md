# Chart Type Algorithms — Deep Research

## 1. Statistical Charts

### Kernel Density Estimation (Foundation for violin, raincloud, ridgeline)
```
f̂(x) = (1/nh) Σᵢ K((x - xᵢ) / h)
```
- K = Gaussian kernel: `K(u) = (1/√(2π)) e^(-u²/2)`
- **Silverman bandwidth**: `h = 0.9 × min(σ, IQR/1.34) × n^(-1/5)` (guards against oversmoothing multimodal)
- **Scott bandwidth**: `h = 1.06 × σ × n^(-1/5)` (simpler, assumes unimodal)
- Evaluate on grid of 200-512 points, map density to width

### Raincloud Plots (Allen et al. 2019)
Three staggered layers:
1. **Half-violin**: One-sided KDE area (clip or draw one side only)
2. **Jittered strip**: `jitter_pos = center + (rand() - 0.5) × jitter_width`
3. **Boxplot**: Tukey (median, Q1, Q3, whiskers at 1.5×IQR) between violin and strip

### Beeswarm Plots
**Force simulation** (D3-force, velocity Verlet):
```
Each tick:
  velocity += acceleration × dt
  position += velocity × dt
  Apply collision force (quadtree O(n log n) neighbor detection)
  Apply x-force: velocity += (target_x - current_x) × strength × alpha
```
Collision: if `distance(a,b) < r_a + r_b`, push apart by `(r_combined - distance)/distance × 0.5`

**Exact dodge** (Wilkinson 1999): Sort by primary axis, scan placed points, find minimum perpendicular offset avoiding collisions. O(n²) but deterministic.

### Ridgeline / Joy Plots
1. KDE per group (same bandwidth for comparability)
2. Vertical baseline: `baseline_i = i × Δy`
3. Render back-to-front (painter's algorithm)
4. Each ridge: filled area with background color for occlusion

### Estimation Plots (Bootstrap CI)
```
For b in 1..B (B=5000-10000):
  sample_b = random WITH replacement, size n
  θ̂_b = statistic(sample_b)
Sort θ̂₁..θ̂_B
Percentile CI: [θ̂_(α/2·B), θ̂_((1-α/2)·B)]
BCa (bias-corrected): adjusts for bias z₀ and acceleration a
```
Layout: Raw data (left) + bootstrap distribution with CI bar (right) + connecting line

## 2. Hierarchical / Relational

### Treemap — Squarified Algorithm (Bruls et al. 2000)
```
squarify(items, row, rectangle):
  if items empty: layoutRow(row, rectangle); return
  candidate = row + [items[0]]
  if worst_ratio(candidate, rect.shortSide) <= worst_ratio(row, rect.shortSide):
    squarify(items[1:], candidate, rectangle)  // add to row
  else:
    layoutRow(row, rectangle)                  // finalize row
    squarify(items, [], remaining_rectangle)   // start new row

worst_ratio(row, w):
  s = Σ areas in row
  for each area r: ratio = max(w²r/s², s²/(w²r))
  return max(all ratios)
```
Row layout: strip width = s/w, each item height = aᵢ × w/s

### Sankey Diagrams
**Node positioning**: BFS for horizontal (column = max(source columns) + 1)

**Vertical — Gauss-Seidel relaxation**:
```
for iteration in 1..max:
  for each node (L→R):
    weighted_center = Σ(source.y × link.value) / Σ(link.value)
    node.y = weighted_center - height/2
  resolve_collisions(column)  // sort by y, push apart
  // Repeat R→L with targets
  decrease α (damping)
```

**Link rendering — cubic bezier ribbons**:
```
mid_x = (source_x + target_x) / 2
Top: M source_x,y0_src  C mid_x,y0_src  mid_x,y0_tgt  target_x,y0_tgt
Bottom: L target_x,y1_tgt  C mid_x,y1_tgt  mid_x,y1_src  source_x,y1_src  Z
```

### Chord Diagrams
- Group angular span: `θᵢ = (gᵢ/G) × (2π - n×padding)`
- Ribbons: quadratic beziers through center between source/target arcs
- `Q center, point_on_target_arc` creates characteristic curved shape

### Sunburst Charts
Radial partition layout:
```
partition(node, x0, x1, depth):
  node.startAngle = x0 × 2π
  node.endAngle = x1 × 2π
  node.innerRadius = depth × radius_per_level
  node.outerRadius = (depth+1) × radius_per_level
  offset = x0
  for child: child_span = (x1-x0) × child.value/node.value
```

### Circle Packing — Front-Chain Algorithm (Wang et al. 2006)
1. Sort siblings descending by radius
2. Maintain front chain (convex-hull-like boundary)
3. For each new circle: find closest adjacent pair, compute tangent position (law of cosines), check overlaps, splice chain
4. **Enclosing circle**: Welzl's algorithm — randomized, expected O(n)

## 3. Specialized Charts

### Horizon Charts — Band Slicing
```
For k bands (typically 3-4), height h = (max-baseline)/k:
  band_value_i = clamp(value - baseline - i×h, 0, h)
  color_i = increasing intensity for higher bands
Negative: mirror above baseline, different hue
All bands share same vertical space → k× compression
```

### Contour Plots — Marching Squares
```
For each 2×2 cell with corners [TL,TR,BR,BL]:
  index = (TL>t?8:0)|(TR>t?4:0)|(BR>t?2:0)|(BL>t?1:0)  // 16 configs
  // Lookup table → edge intersection pattern
  // Linear interpolation: fraction = (t-vA)/(vB-vA)
  // Saddle disambiguation (index 5,10): use corner average
```
Trace connected segments into polylines/polygons.

For contour density: first apply 2D KDE (grid of Gaussian weights), then marching squares.

### Clustered Heatmap with Dendrograms
**Agglomerative clustering**:
```
while |clusters| > 1:
  (a,b) = argmin linkage_distance(cluster_i, cluster_j)
  merge at height = distance
```
Linkage methods: Single (min), Complete (max), Average (mean), **Ward's** (minimize within-cluster variance: `d(A,B) = |A||B|/(|A|+|B|) × ||μ_A-μ_B||²`)

**Optimal leaf ordering** (Bar-Joseph 2001): DP in O(n³) to minimize Σ adjacent leaf distances from 2^(n-1) possible orderings.

**Dendrogram rendering**: U-shaped connectors at each merge height.

### Parallel Coordinates
- Each dimension → vertical axis at `x_i = i × spacing`
- Each data point → polyline: `dimensions.map((dim,i) => [x_i, scale_dim(d[dim])])`
- Brushing: highlight if point passes through ALL brush selections
- For >10K lines: Canvas rendering or edge bundling

## 4. Domain-Specific

### Forest Plots (Meta-Analysis)
```
Per study: effect size (point, size ∝ weight) + CI (horizontal line)
Weight: w_i = 1/σ_i² (fixed) or 1/(σ_i²+τ²) (random)

Pooled (DerSimonian-Laird):
  θ̂_FE = Σ(w_i·θ_i)/Σ(w_i)
  Q = Σ w_i(θ_i - θ̂_FE)²
  τ² = max(0, (Q-(k-1))/(Σw_i - Σw_i²/Σw_i))
  w*_i = 1/(σ_i²+τ²)
  θ̂_RE = Σ(w*_i·θ_i)/Σ(w*_i)
```
Diamond at pooled estimate, reference line at null effect (0 or 1 for ratios).

### Kaplan-Meier Survival Curves
```
At each event time t_j:
  n_j = number at risk, d_j = events
  Ŝ(t) = Π_{t_j ≤ t} (1 - d_j/n_j)

Greenwood variance: Var = Ŝ(t)² × Σ d_j/(n_j(n_j-d_j))
```
Step-after interpolation. Censoring marks (+). Risk table below chart. Log-rank test for group comparison.

### Bland-Altman
```
x_axis = (x_i + y_i) / 2   // mean of two methods
y_axis = x_i - y_i          // difference
Bias = mean(differences)
LoA = bias ± 1.96 × SD(differences)
```
Three horizontal reference lines + optional CI bands.

### Candlestick (OHLC)
```
body_top = max(Open, Close), body_bottom = min(Open, Close)
Body: rect(x-w/2, body_bottom, w, body_top-body_bottom), fill=green/red
Upper wick: line(x, body_top, x, High)
Lower wick: line(x, body_bottom, x, Low)
```

### Smith Charts
Moebius transform: `Γ = (z-1)/(z+1)` where z = Z/Z₀ (normalized impedance)
- Constant-resistance circles: center=(r/(r+1), 0), radius=1/(r+1)
- Constant-reactance arcs: center=(1, 1/x), radius=1/|x|, clipped to unit circle

### Ternary Plots (Barycentric Coordinates)
```
// Components (a,b,c) where a+b+c=1
x = b + c/2
y = c × √3/2
```
Grid lines: straight lines parallel to opposite triangle edge.

## 5. Layout Algorithms

### Smart Label Anti-Collision

**Bitmap-based** (D3-cloud pattern):
- Render label to hidden canvas → extract pixel mask
- Spiral outward from preferred position
- Check overlap via bitwise AND with occupied mask
- Place when AND=0, OR into occupied mask

**Simulated annealing** (ggrepel pattern):
```
E = Σ label-label overlap + Σ label-point overlap + Σ distance-from-anchor + Σ leader-line-crossing
For step in 1..max:
  Perturb random label by Δx, Δy scaled by T
  ΔE = energy_after - energy_before
  Accept if ΔE<0, or with probability exp(-ΔE/T)
  T *= cooling_rate (0.99-0.999)
```

**Force-based**: d3-force with box collision (not circle), attractive force to anchor, boundary containment.

### Force-Directed Graph Layout (D3-force)
Barnes-Hut approximation: quadtree reduces O(n²) to O(n log n).
```
if cell_size/distance < θ (0.9): treat as single body
else: recurse into children
```
Link force: spring with bias inversely proportional to degree.
~300 ticks to convergence (alpha: 1 → alphaMin via exponential decay).

### Faceting / Small Multiples
```
cols = ceil(sqrt(n)), rows = ceil(n/cols)
panel_x = margin.left + col × (panel_width + spacing)
panel_y = margin.top + row × (panel_height + spacing)
```
Scale sync options: shared (global domain), free, shared-within-row/col.
Axis label dedup: y-labels on leftmost panels only, x-labels on bottom only.

## Complexity Summary

| Algorithm | Time | Space |
|---|---|---|
| KDE | O(n × grid) | O(grid) |
| Beeswarm (force) | O(n log n) per tick | O(n) |
| Treemap squarified | O(n) | O(n) |
| Sankey relaxation | O(iterations × n) | O(n + edges) |
| Circle packing | O(n log n) | O(n) |
| Marching squares | O(grid_w × grid_h) | O(grid) |
| Hierarchical clustering | O(n² log n) | O(n²) |
| Optimal leaf order | O(n³) | O(n²) |
| Barnes-Hut force | O(n log n) per tick | O(n) |
| Label SA | O(n² × steps) | O(n) |
| Welzl enclosure | O(n) expected | O(n) |
