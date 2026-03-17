// SPDX-License-Identifier: MIT OR Apache-2.0
//! Position adjustments: stack, dodge, fill, jitter.

use crate::compile::stat_transform::ResolvedLayer;
use crate::error::Result;
use crate::grammar::position::Position;

/// Apply position adjustments across resolved layers.
pub fn apply_positions(layers: &mut [ResolvedLayer]) -> Result<()> {
    use crate::error::ChartError;
    use crate::grammar::layer::MarkType;

    // Validate position/mark-type compatibility
    for layer in layers.iter() {
        match layer.position {
            Position::Stack | Position::Fill => {
                if !matches!(layer.mark, MarkType::Bar | MarkType::Area) {
                    return Err(ChartError::InvalidParameter(format!(
                        "Stack/Fill position is only valid for Bar and Area marks, got {:?}",
                        layer.mark
                    )));
                }
            }
            Position::Dodge => {
                if !matches!(layer.mark, MarkType::Bar | MarkType::Point) {
                    return Err(ChartError::InvalidParameter(format!(
                        "Dodge position is only valid for Bar and Point marks, got {:?}",
                        layer.mark
                    )));
                }
            }
            Position::Jitter { .. } => {
                if !matches!(layer.mark, MarkType::Point | MarkType::Line) {
                    return Err(ChartError::InvalidParameter(format!(
                        "Jitter position is only valid for Point and Line marks, got {:?}",
                        layer.mark
                    )));
                }
            }
            Position::Identity => {}
        }
    }

    // Check if any layers need position adjustment
    let positions: Vec<Position> = layers.iter().map(|l| l.position).collect();

    // Apply stack/fill
    let has_stack = positions.iter().any(|p| matches!(p, Position::Stack));
    let has_fill = positions.iter().any(|p| matches!(p, Position::Fill));
    if has_stack || has_fill {
        apply_stack(layers);
        if has_fill {
            apply_fill_normalize(layers);
        }
    }

    // Apply dodge
    let has_dodge = positions.iter().any(|p| matches!(p, Position::Dodge));
    if has_dodge {
        apply_dodge(layers);
    }

    // Apply jitter
    for layer in layers.iter_mut() {
        if let Position::Jitter { x_amount, y_amount } = layer.position {
            apply_jitter(layer, x_amount, y_amount);
        }
    }

    Ok(())
}

/// Stack: for layers with Position::Stack/Fill, accumulate y-values at shared x positions.
///
/// Uses key-based alignment: builds a union of all x-values across stackable layers,
/// then reorders/fills each layer to match the union, ensuring correct stacking
/// even when layers have sparse or differently-ordered x-values.
fn apply_stack(layers: &mut [ResolvedLayer]) {
    use std::collections::HashMap;

    let stackable: Vec<usize> = layers
        .iter()
        .enumerate()
        .filter(|(_, l)| matches!(l.position, Position::Stack | Position::Fill))
        .map(|(i, _)| i)
        .collect();

    if stackable.len() < 2 {
        // Still set baseline for single stackable layer
        if stackable.len() == 1 {
            let idx = stackable[0];
            let n = layers[idx].x_data.len().min(layers[idx].y_data.len());
            layers[idx].y_baseline = Some(vec![0.0; n]);
        }
        return;
    }

    // Build union of all x-values preserving first-seen order
    // Use bit representation for exact f64 equality (NaN already filtered out upstream)
    let mut union_x: Vec<f64> = Vec::new();
    let mut seen_bits: Vec<u64> = Vec::new();
    for &si in &stackable {
        let layer = &layers[si];
        let n = layer.x_data.len().min(layer.y_data.len());
        for j in 0..n {
            let bits = layer.x_data[j].to_bits();
            if !seen_bits.contains(&bits) {
                seen_bits.push(bits);
                union_x.push(layer.x_data[j]);
            }
        }
    }

    // Rewrite each stackable layer to the union x-order, inserting 0.0 for missing keys
    for &si in &stackable {
        let layer = &layers[si];
        let n = layer.x_data.len().min(layer.y_data.len());
        let mut x_to_y: HashMap<u64, f64> = HashMap::new();
        for j in 0..n {
            x_to_y.insert(layer.x_data[j].to_bits(), layer.y_data[j]);
        }
        let new_x: Vec<f64> = union_x.clone();
        let new_y: Vec<f64> = union_x.iter().map(|k| *x_to_y.get(&k.to_bits()).unwrap_or(&0.0)).collect();
        let layer = &mut layers[si];
        layer.x_data = new_x;
        layer.y_data = new_y;
    }

    let n = union_x.len();

    // Now apply index-based stacking (safe because all layers are aligned)
    // Maintain diverging baselines: positive values stack up, negative values stack down
    let mut pos_baseline = vec![0.0_f64; n];
    let mut neg_baseline = vec![0.0_f64; n];

    for &si in &stackable {
        let layer = &mut layers[si];
        let mut baseline = vec![0.0_f64; n];
        for i in 0..n {
            if layer.y_data[i] >= 0.0 {
                baseline[i] = pos_baseline[i];
                pos_baseline[i] += layer.y_data[i];
                layer.y_data[i] = pos_baseline[i];
            } else {
                baseline[i] = neg_baseline[i];
                neg_baseline[i] += layer.y_data[i];
                layer.y_data[i] = neg_baseline[i];
            }
        }
        layer.y_baseline = Some(baseline);
    }
}

/// Fill: normalize stacked columns so each x-position sums to 1.0.
fn apply_fill_normalize(layers: &mut [ResolvedLayer]) {
    let fillable: Vec<usize> = layers
        .iter()
        .enumerate()
        .filter(|(_, l)| matches!(l.position, Position::Fill))
        .map(|(i, _)| i)
        .collect();

    if fillable.is_empty() {
        return;
    }

    // Find the length of data
    let n = fillable
        .iter()
        .map(|&i| layers[i].x_data.len().min(layers[i].y_data.len()))
        .min()
        .unwrap_or(0);

    // Compute column totals (from the top of the last stacked layer)
    let last = *fillable.last().unwrap();
    let totals: Vec<f64> = (0..n)
        .map(|i| {
            if i < layers[last].y_data.len() {
                layers[last].y_data[i]
            } else {
                1.0
            }
        })
        .collect();

    // Normalize all fillable layers
    for &li in &fillable {
        let layer = &mut layers[li];
        for i in 0..n.min(layer.y_data.len()) {
            if totals[i].abs() < 1e-15 {
                // Zero-sum column: set to 0 rather than dividing by pseudo-1
                layer.y_data[i] = 0.0;
                if let Some(ref mut baseline) = layer.y_baseline {
                    if i < baseline.len() {
                        baseline[i] = 0.0;
                    }
                }
            } else {
                layer.y_data[i] /= totals[i];
                if let Some(ref mut baseline) = layer.y_baseline {
                    if i < baseline.len() {
                        baseline[i] /= totals[i];
                    }
                }
            }
        }
    }
}

/// Dodge: offset x-positions so grouped bars sit side by side.
fn apply_dodge(layers: &mut [ResolvedLayer]) {
    let dodgeable: Vec<usize> = layers
        .iter()
        .enumerate()
        .filter(|(_, l)| matches!(l.position, Position::Dodge))
        .map(|(i, _)| i)
        .collect();

    let n_groups = dodgeable.len();
    if n_groups < 2 {
        return;
    }

    // Estimate spacing from first layer's x data
    let spacing = if !layers[dodgeable[0]].x_data.is_empty() && layers[dodgeable[0]].x_data.len() > 1 {
        (layers[dodgeable[0]].x_data[1] - layers[dodgeable[0]].x_data[0]).abs()
    } else {
        1.0
    };

    let sub_width = spacing * 0.8 / n_groups as f64;

    for (group_idx, &layer_idx) in dodgeable.iter().enumerate() {
        let offset = (group_idx as f64 - (n_groups as f64 - 1.0) / 2.0) * sub_width;
        let layer = &mut layers[layer_idx];
        layer.dodge_width = Some(sub_width);
        for x in &mut layer.x_data {
            *x += offset;
        }
    }
}

/// Jitter: add deterministic pseudo-random displacement.
fn apply_jitter(layer: &mut ResolvedLayer, x_amount: f64, y_amount: f64) {
    for (i, x) in layer.x_data.iter_mut().enumerate() {
        *x += deterministic_noise(i, 0) * x_amount;
    }
    for (i, y) in layer.y_data.iter_mut().enumerate() {
        *y += deterministic_noise(i, 1) * y_amount;
    }
}

/// Simple deterministic pseudo-random value in [-0.5, 0.5] based on index and seed.
fn deterministic_noise(index: usize, seed: usize) -> f64 {
    // Simple hash-based PRNG
    let mut h = index.wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(340_573_321));
    h ^= h >> 16;
    h = h.wrapping_mul(0x045d_9f3b);
    h ^= h >> 16;
    (h & 0xFFFF) as f64 / 65536.0 - 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::layer::MarkType;

    fn make_layer(y: Vec<f64>, pos: Position, idx: usize) -> ResolvedLayer {
        let n = y.len();
        ResolvedLayer {
            mark: MarkType::Bar,
            x_data: (0..n).map(|i| i as f64).collect(),
            y_data: y,
            categories: None,
            y_baseline: None,
            boxplot: None,
            inner_radius_fraction: 0.0,
            position: pos,
            is_binned: false,
            facet_values: None,
            layer_idx: idx,
            heatmap_data: None,
            row_labels: None,
            col_labels: None,
            annotate_cells: false,
            label: None,
            dodge_width: None,
        }
    }

    #[test]
    fn stack_two_layers() {
        let mut layers = vec![
            make_layer(vec![10.0, 20.0], Position::Stack, 0),
            make_layer(vec![5.0, 15.0], Position::Stack, 1),
        ];
        apply_positions(&mut layers).unwrap();

        // First layer baseline is 0
        assert_eq!(layers[0].y_baseline.as_ref().unwrap(), &vec![0.0, 0.0]);
        // Second layer baseline is first layer's y
        assert_eq!(layers[1].y_baseline.as_ref().unwrap(), &vec![10.0, 20.0]);
        // Second layer y is sum
        assert!((layers[1].y_data[0] - 15.0).abs() < 1e-10);
        assert!((layers[1].y_data[1] - 35.0).abs() < 1e-10);
    }

    #[test]
    fn dodge_two_layers() {
        let mut layers = vec![
            make_layer(vec![10.0, 20.0], Position::Dodge, 0),
            make_layer(vec![5.0, 15.0], Position::Dodge, 1),
        ];
        let orig_x0 = layers[0].x_data.clone();
        let orig_x1 = layers[1].x_data.clone();
        apply_positions(&mut layers).unwrap();

        // X positions should be offset symmetrically
        for i in 0..2 {
            assert!(layers[0].x_data[i] < orig_x0[i]); // shifted left
            assert!(layers[1].x_data[i] > orig_x1[i]); // shifted right
        }
    }

    #[test]
    fn fill_normalizes() {
        let mut layers = vec![
            make_layer(vec![25.0, 50.0], Position::Fill, 0),
            make_layer(vec![75.0, 50.0], Position::Fill, 1),
        ];
        apply_positions(&mut layers).unwrap();

        // Column totals should be 1.0
        for i in 0..2 {
            let total = layers[1].y_data[i]; // Top of stack = total
            assert!((total - 1.0).abs() < 1e-10, "Column {i} total = {total}");
        }
    }

    #[test]
    fn jitter_displaces() {
        let mut layer = make_layer(vec![1.0, 2.0, 3.0], Position::Jitter { x_amount: 0.1, y_amount: 0.1 }, 0);
        layer.mark = MarkType::Point; // Jitter is only valid for Point/Line
        let mut layers = vec![layer];
        let orig_x = layers[0].x_data.clone();
        apply_positions(&mut layers).unwrap();

        // Points should be displaced
        for i in 0..3 {
            assert!((layers[0].x_data[i] - orig_x[i]).abs() <= 0.05 + 1e-10);
        }
    }

    #[test]
    fn stack_three_layers() {
        let mut layers = vec![
            make_layer(vec![10.0, 20.0], Position::Stack, 0),
            make_layer(vec![5.0, 10.0], Position::Stack, 1),
            make_layer(vec![3.0, 7.0], Position::Stack, 2),
        ];
        apply_positions(&mut layers).unwrap();

        // Layer 0: baseline = [0, 0], y_data = [10, 20]
        assert_eq!(layers[0].y_baseline.as_ref().unwrap(), &vec![0.0, 0.0]);
        assert!((layers[0].y_data[0] - 10.0).abs() < 1e-10);
        assert!((layers[0].y_data[1] - 20.0).abs() < 1e-10);
        // Layer 1: baseline = [10, 20], y_data = [15, 30]
        assert!((layers[1].y_data[0] - 15.0).abs() < 1e-10);
        assert!((layers[1].y_data[1] - 30.0).abs() < 1e-10);
        // Layer 2: baseline = [15, 30], y_data = [18, 37]
        assert!((layers[2].y_data[0] - 18.0).abs() < 1e-10);
        assert!((layers[2].y_data[1] - 37.0).abs() < 1e-10);
    }

    #[test]
    fn single_layer_dodge_is_noop() {
        let mut layers = vec![make_layer(vec![10.0, 20.0], Position::Dodge, 0)];
        let orig_x = layers[0].x_data.clone();
        apply_positions(&mut layers).unwrap();
        // Single dodged layer should not be offset
        assert_eq!(layers[0].x_data, orig_x);
    }

    #[test]
    fn stack_sparse_groups() {
        // Layer 0 has x=[0,1,2], layer 1 has x=[1,2,3] (sparse overlap)
        let mut l0 = make_layer(vec![10.0, 20.0, 30.0], Position::Stack, 0);
        l0.x_data = vec![0.0, 1.0, 2.0];
        let mut l1 = make_layer(vec![5.0, 15.0, 25.0], Position::Stack, 1);
        l1.x_data = vec![1.0, 2.0, 3.0];
        let mut layers = vec![l0, l1];
        apply_positions(&mut layers).unwrap();

        // After key-based alignment, both layers should have x=[0,1,2,3]
        assert_eq!(layers[0].x_data.len(), 4);
        assert_eq!(layers[1].x_data.len(), 4);
        // Layer 0 y at x=3 should be 0 (missing)
        assert!((layers[0].y_data[3] - 0.0).abs() < 1e-10);
        // Layer 1 y at x=0 should be 0 (missing), stacked on baseline
        // Layer 1 baseline at x=0 should be layer 0's value (10)
        assert!((layers[1].y_baseline.as_ref().unwrap()[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn diverging_stack_mixed_positive_negative() {
        let mut layers = vec![
            make_layer(vec![10.0, -5.0], Position::Stack, 0),
            make_layer(vec![-3.0, 8.0], Position::Stack, 1),
        ];
        apply_positions(&mut layers).unwrap();

        // Layer 0: positive 10 stacks up from 0, negative -5 stacks down from 0
        assert!((layers[0].y_baseline.as_ref().unwrap()[0] - 0.0).abs() < 1e-10);
        assert!((layers[0].y_data[0] - 10.0).abs() < 1e-10);
        assert!((layers[0].y_baseline.as_ref().unwrap()[1] - 0.0).abs() < 1e-10);
        assert!((layers[0].y_data[1] - (-5.0)).abs() < 1e-10);

        // Layer 1: -3 stacks down from neg_baseline[0]=0, 8 stacks up from pos_baseline[1]=0
        assert!((layers[1].y_baseline.as_ref().unwrap()[0] - 0.0).abs() < 1e-10);
        assert!((layers[1].y_data[0] - (-3.0)).abs() < 1e-10);
        assert!((layers[1].y_baseline.as_ref().unwrap()[1] - 0.0).abs() < 1e-10);
        assert!((layers[1].y_data[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn fill_zero_sum_column() {
        let mut layers = vec![
            make_layer(vec![0.0, 10.0], Position::Fill, 0),
            make_layer(vec![0.0, 20.0], Position::Fill, 1),
        ];
        apply_positions(&mut layers).unwrap();

        // Column 0: both zero → should remain 0, not 0.5
        assert!((layers[0].y_data[0]).abs() < 1e-10);
        assert!((layers[1].y_data[0]).abs() < 1e-10);
        // Column 1: should normalize to 1.0
        assert!((layers[1].y_data[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn invalid_position_mark_combo_errors() {
        // Jitter + Heatmap should error
        let mut layer = make_layer(vec![1.0], Position::Jitter { x_amount: 0.1, y_amount: 0.1 }, 0);
        layer.mark = MarkType::Heatmap;
        let result = apply_positions(&mut [layer]);
        assert!(result.is_err());
    }

    #[test]
    fn dodge_sets_dodge_width() {
        let mut layers = vec![
            make_layer(vec![10.0, 20.0], Position::Dodge, 0),
            make_layer(vec![5.0, 15.0], Position::Dodge, 1),
        ];
        apply_positions(&mut layers).unwrap();
        // Both layers should have dodge_width set
        assert!(layers[0].dodge_width.is_some());
        assert!(layers[1].dodge_width.is_some());
        // dodge_width should be spacing * 0.8 / n_groups = 1.0 * 0.8 / 2 = 0.4
        let dw = layers[0].dodge_width.unwrap();
        assert!((dw - 0.4).abs() < 1e-10);
    }
}
