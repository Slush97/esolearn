// SPDX-License-Identifier: MIT OR Apache-2.0
//! Position adjustments: stack, dodge, fill, jitter.

use crate::compile::stat_transform::ResolvedLayer;
use crate::error::Result;
use crate::grammar::position::Position;

/// Apply position adjustments across resolved layers.
pub fn apply_positions(layers: &mut [ResolvedLayer]) -> Result<()> {
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
fn apply_stack(layers: &mut [ResolvedLayer]) {
    let stackable: Vec<usize> = layers
        .iter()
        .enumerate()
        .filter(|(_, l)| matches!(l.position, Position::Stack | Position::Fill))
        .map(|(i, _)| i)
        .collect();

    if stackable.len() < 2 {
        return;
    }

    // For each stackable layer (in order), set baseline to sum of all prior layers
    // We match by x-value index position
    for si in 1..stackable.len() {
        let layer_idx = stackable[si];
        let n = layers[layer_idx].x_data.len().min(layers[layer_idx].y_data.len());
        let mut baseline = vec![0.0_f64; n];

        // Sum y-values from all prior stackable layers at each position
        for &prev_idx in &stackable[..si] {
            let prev = &layers[prev_idx];
            let prev_n = prev.x_data.len().min(prev.y_data.len());
            for (b, &y) in baseline.iter_mut().zip(prev.y_data.iter()).take(n.min(prev_n)) {
                *b += y;
            }
        }

        let layer = &mut layers[layer_idx];
        // Adjust y_data to represent the top of this segment
        for (y, &b) in layer.y_data.iter_mut().zip(baseline.iter()).take(n) {
            *y += b;
        }
        layer.y_baseline = Some(baseline);
    }

    // First stackable layer gets zero baseline
    let first = stackable[0];
    let n = layers[first].x_data.len().min(layers[first].y_data.len());
    layers[first].y_baseline = Some(vec![0.0; n]);
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
            let total = if totals[i].abs() < 1e-15 { 1.0 } else { totals[i] };
            layer.y_data[i] /= total;
            if let Some(ref mut baseline) = layer.y_baseline {
                if i < baseline.len() {
                    baseline[i] /= total;
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
        let mut layers = vec![make_layer(vec![1.0, 2.0, 3.0], Position::Jitter { x_amount: 0.1, y_amount: 0.1 }, 0)];
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

        // Layer 0: y_data unchanged, baseline = [0, 0]
        assert_eq!(layers[0].y_baseline.as_ref().unwrap(), &vec![0.0, 0.0]);
        // Layer 1: y_data = [5+10, 10+20] = [15, 30], baseline = [10, 20]
        assert!((layers[1].y_data[0] - 15.0).abs() < 1e-10);
        assert!((layers[1].y_data[1] - 30.0).abs() < 1e-10);
        // Layer 2: baseline = sum of modified layer0 + layer1 y_data
        // baseline = [10+15, 20+30] = [25, 50]
        // y_data = [3+25, 7+50] = [28, 57]
        assert!((layers[2].y_data[0] - 28.0).abs() < 1e-10);
        assert!((layers[2].y_data[1] - 57.0).abs() < 1e-10);
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
