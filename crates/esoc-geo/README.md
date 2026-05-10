# esoc-geo

> **Workspace-internal crate.** Not published to crates.io (`publish = false`).
> Provides geographic primitives for the rest of the `esoc-*` visualization
> stack — its public API is not stable and may change to suit consumers
> inside this workspace.

Geographic types, map projections, and spatial utilities used by
`esoc-chart` for map visualizations.

## Scope

- Geometry primitives (point, line, polygon, multi-*)
- Map projections (Mercator, Equal Earth, Natural Earth, Albers USA)
- Polygon simplification (Visvalingam-Whyatt, Douglas-Peucker)
- GeoJSON parsing (feature: `geojson`)
- Bundled world / US geometries (feature: `bundled`)

## Features

| Feature   | Pulls in              | Purpose                                  |
| --------- | --------------------- | ---------------------------------------- |
| `geojson` | `serde`, `serde_json` | Parse GeoJSON `FeatureCollection` input  |
| `bundled` | `geojson`, `zstd`     | Embedded zstd-compressed world / US data |

## Status

Functionally complete for the visualization use cases that drive it.
Tests cover projection round-trips, polygon ops, and GeoJSON parsing.
Because the crate is workspace-internal, breaking changes land without
deprecation cycles — depend on it from another `esoc-*` crate, not from
external code.
