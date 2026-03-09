// SPDX-License-Identifier: MIT OR Apache-2.0
//! Statistical transform resolution: Layer → ResolvedLayer.

use crate::compile::stat_aggregate;
use crate::compile::stat_bin;
use crate::compile::stat_boxplot::{self, BoxPlotSummary};
use crate::compile::stat_smooth;
use crate::error::{ChartError, Result};
use crate::grammar::layer::{Layer, MarkType};
use crate::grammar::position::Position;
use crate::grammar::stat::Stat;

/// Intermediate representation after stat transforms, before mark generation.
#[derive(Clone, Debug)]
pub struct ResolvedLayer {
    /// Mark type to render.
    pub mark: MarkType,
    /// X data (post-transform).
    pub x_data: Vec<f64>,
    /// Y data (post-transform).
    pub y_data: Vec<f64>,
    /// Category labels (post-transform).
    pub categories: Option<Vec<String>>,
    /// Y baseline for stacking (populated by position adjustments in 4A).
    pub y_baseline: Option<Vec<f64>>,
    /// Box plot summaries (only for Stat::BoxPlot).
    pub boxplot: Option<Vec<BoxPlotSummary>>,
    /// Inner radius fraction for donut charts (0.0 = pie, >0 = donut).
    pub inner_radius_fraction: f32,
    /// Position adjustment from the grammar layer.
    pub position: Position,
    /// Whether this layer was produced by a binning stat (histogram).
    pub is_binned: bool,
    /// Per-row facet assignment (from layer).
    pub facet_values: Option<Vec<String>>,
    /// Index of this layer in the original chart.
    pub layer_idx: usize,
    /// Heatmap data (row-major 2D matrix).
    pub heatmap_data: Option<Vec<Vec<f64>>>,
    /// Row labels for heatmap.
    pub row_labels: Option<Vec<String>>,
    /// Column labels for heatmap.
    pub col_labels: Option<Vec<String>>,
    /// Whether to annotate heatmap cells with values.
    pub annotate_cells: bool,
    /// Human-readable label for this layer (used in legends).
    pub label: Option<String>,
    /// Bar width in data units, set by dodge position adjustment.
    pub dodge_width: Option<f64>,
}

/// Resolve a grammar Layer by applying its stat transform.
pub fn resolve_layer(layer: &Layer, layer_idx: usize) -> Result<ResolvedLayer> {
    match &layer.stat {
        Stat::Identity => resolve_identity(layer, layer_idx),
        Stat::Bin { bins } => resolve_bin(layer, layer_idx, *bins),
        Stat::BoxPlot => resolve_boxplot(layer, layer_idx),
        Stat::Aggregate { func } => resolve_aggregate(layer, layer_idx, *func),
        Stat::Smooth { bandwidth } => resolve_smooth(layer, layer_idx, *bandwidth),
    }
}

fn resolve_identity(layer: &Layer, layer_idx: usize) -> Result<ResolvedLayer> {
    Ok(ResolvedLayer {
        mark: layer.mark,
        x_data: layer.x_data.clone().unwrap_or_default(),
        y_data: layer.y_data.clone().unwrap_or_default(),
        categories: layer.categories.clone(),
        y_baseline: None,
        boxplot: None,
        inner_radius_fraction: layer.inner_radius_fraction,
        position: layer.position,
        is_binned: false,
        facet_values: layer.facet_values.clone(),
        layer_idx,
        heatmap_data: layer.heatmap_data.clone(),
        row_labels: layer.row_labels.clone(),
        col_labels: layer.col_labels.clone(),
        annotate_cells: layer.annotate_cells,
        label: layer.label.clone(),
        dodge_width: None,
    })
}

fn resolve_bin(layer: &Layer, layer_idx: usize, bins: usize) -> Result<ResolvedLayer> {
    let x_data = layer.x_data.as_ref().ok_or(ChartError::EmptyData)?;

    let bin_count = if bins == 0 {
        stat_bin::sturges_bins(x_data.len())
    } else {
        bins
    };

    let (centers, counts) = stat_bin::compute_bins(x_data, bin_count);

    Ok(ResolvedLayer {
        mark: MarkType::Bar, // Histograms render as bars
        x_data: centers,
        y_data: counts,
        categories: None,
        y_baseline: None,
        boxplot: None,
        inner_radius_fraction: 0.0,
        position: layer.position,
        is_binned: true,
        facet_values: layer.facet_values.clone(),
        layer_idx,
        heatmap_data: None,
        row_labels: None,
        col_labels: None,
        annotate_cells: false,
        label: layer.label.clone(),
        dodge_width: None,
    })
}

fn resolve_boxplot(layer: &Layer, layer_idx: usize) -> Result<ResolvedLayer> {
    let categories = layer
        .categories
        .as_ref()
        .ok_or_else(|| ChartError::InvalidParameter("boxplot requires categories".into()))?;
    let y_data = layer.y_data.as_ref().ok_or(ChartError::EmptyData)?;

    let summaries = stat_boxplot::compute_boxplot(categories, y_data)?;

    // Generate x positions for categories and y bounds for data
    let x_data: Vec<f64> = (0..summaries.len()).map(|i| i as f64).collect();
    let cat_labels: Vec<String> = summaries.iter().map(|s| s.category.clone()).collect();

    // Y data represents the range for scale computation
    let mut all_y = Vec::new();
    for s in &summaries {
        all_y.push(s.whisker_lo);
        all_y.push(s.whisker_hi);
        all_y.extend_from_slice(&s.outliers);
    }

    Ok(ResolvedLayer {
        mark: MarkType::Bar, // Will be special-cased in mark_gen
        x_data,
        y_data: all_y,
        categories: Some(cat_labels),
        y_baseline: None,
        boxplot: Some(summaries),
        inner_radius_fraction: 0.0,
        position: layer.position,
        is_binned: false,
        facet_values: layer.facet_values.clone(),
        layer_idx,
        heatmap_data: None,
        row_labels: None,
        col_labels: None,
        annotate_cells: false,
        label: layer.label.clone(),
        dodge_width: None,
    })
}

fn resolve_aggregate(
    layer: &Layer,
    layer_idx: usize,
    func: crate::grammar::stat::AggregateFunc,
) -> Result<ResolvedLayer> {
    let categories = layer
        .categories
        .as_ref()
        .ok_or_else(|| ChartError::InvalidParameter("aggregate requires categories".into()))?;
    let y_data = layer.y_data.as_ref().ok_or(ChartError::EmptyData)?;

    let result = stat_aggregate::compute_aggregate(categories, y_data, func)?;

    Ok(ResolvedLayer {
        mark: layer.mark,
        x_data: result.x_data,
        y_data: result.y_data,
        categories: Some(result.categories),
        y_baseline: None,
        boxplot: None,
        inner_radius_fraction: 0.0,
        position: layer.position,
        is_binned: false,
        facet_values: layer.facet_values.clone(),
        layer_idx,
        heatmap_data: None,
        row_labels: None,
        col_labels: None,
        annotate_cells: false,
        label: layer.label.clone(),
        dodge_width: None,
    })
}

fn resolve_smooth(layer: &Layer, layer_idx: usize, bandwidth: f64) -> Result<ResolvedLayer> {
    let x_data = layer.x_data.as_ref().ok_or(ChartError::EmptyData)?;
    let y_data = layer.y_data.as_ref().ok_or(ChartError::EmptyData)?;

    let (x_smooth, y_smooth) = stat_smooth::compute_loess(x_data, y_data, bandwidth);

    Ok(ResolvedLayer {
        mark: layer.mark,
        x_data: x_smooth,
        y_data: y_smooth,
        categories: layer.categories.clone(),
        y_baseline: None,
        boxplot: None,
        inner_radius_fraction: 0.0,
        position: layer.position,
        is_binned: false,
        facet_values: layer.facet_values.clone(),
        layer_idx,
        heatmap_data: None,
        row_labels: None,
        col_labels: None,
        annotate_cells: false,
        label: layer.label.clone(),
        dodge_width: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::stat::AggregateFunc;

    #[test]
    fn identity_passthrough() {
        let layer = Layer::new(MarkType::Point)
            .with_x(vec![1.0, 2.0, 3.0])
            .with_y(vec![4.0, 5.0, 6.0]);
        let resolved = resolve_layer(&layer, 0).unwrap();
        assert_eq!(resolved.x_data, vec![1.0, 2.0, 3.0]);
        assert_eq!(resolved.y_data, vec![4.0, 5.0, 6.0]);
        assert!(!resolved.is_binned);
        assert!(resolved.boxplot.is_none());
    }

    #[test]
    fn bin_produces_bars() {
        let layer = Layer::new(MarkType::Bar)
            .with_x(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            .stat(Stat::Bin { bins: 5 });
        let resolved = resolve_layer(&layer, 0).unwrap();
        assert_eq!(resolved.x_data.len(), 5);
        assert_eq!(resolved.y_data.len(), 5);
        assert!(resolved.is_binned);
        assert!(matches!(resolved.mark, MarkType::Bar));
    }

    #[test]
    fn boxplot_produces_summaries() {
        let layer = Layer::new(MarkType::Bar)
            .with_y(vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .with_categories(vec!["A".into(); 5])
            .stat(Stat::BoxPlot);
        let resolved = resolve_layer(&layer, 0).unwrap();
        assert!(resolved.boxplot.is_some());
        let summaries = resolved.boxplot.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].category, "A");
    }

    #[test]
    fn aggregate_routing() {
        let layer = Layer::new(MarkType::Bar)
            .with_y(vec![10.0, 20.0, 30.0])
            .with_categories(vec!["A".into(), "B".into(), "A".into()])
            .stat(Stat::Aggregate { func: AggregateFunc::Sum });
        let resolved = resolve_layer(&layer, 0).unwrap();
        assert_eq!(resolved.categories.as_ref().unwrap(), &["A", "B"]);
        assert!((resolved.y_data[0] - 40.0).abs() < 1e-10); // A: 10+30
        assert!((resolved.y_data[1] - 20.0).abs() < 1e-10); // B: 20
    }

    #[test]
    fn smooth_produces_output() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let layer = Layer::new(MarkType::Line)
            .with_x(x.clone())
            .with_y(y)
            .stat(Stat::Smooth { bandwidth: 0.5 });
        let resolved = resolve_layer(&layer, 0).unwrap();
        assert_eq!(resolved.x_data.len(), 20);
        assert_eq!(resolved.y_data.len(), 20);
    }
}
