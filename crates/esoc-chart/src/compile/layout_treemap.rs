// SPDX-License-Identifier: MIT OR Apache-2.0
//! Squarified treemap layout algorithm (Bruls et al. 2000).

use esoc_scene::bounds::BoundingBox;

/// A single cell in the treemap layout.
pub struct TreemapCell {
    /// Index into the original data arrays.
    pub index: usize,
    /// Pixel-space rectangle.
    pub bounds: BoundingBox,
}

/// Compute a squarified treemap layout.
///
/// `values` are the data values (must be non-negative). Zero values are filtered out.
/// `container` is the available pixel-space rectangle.
///
/// Returns cells positioned within `container`, sized proportionally to values.
pub fn squarified_layout(values: &[f64], container: BoundingBox) -> Vec<TreemapCell> {
    // Pair each value with its original index, filter out zeros
    let mut items: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| (i, v))
        .collect();

    if items.is_empty() {
        return Vec::new();
    }

    // Sort by value descending
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Scale values so they sum to the container area
    let total: f64 = items.iter().map(|(_, v)| v).sum();
    let total_area = f64::from(container.w) * f64::from(container.h);
    let scale = total_area / total;
    let areas: Vec<f64> = items.iter().map(|(_, v)| v * scale).collect();

    let mut cells = Vec::with_capacity(items.len());
    let mut remaining = BoundingBox::new(container.x, container.y, container.w, container.h);
    let mut idx = 0;

    while idx < items.len() {
        let shorter_side = f64::from(remaining.w.min(remaining.h));
        if shorter_side <= 0.0 {
            break;
        }

        // Greedily add items to the current strip
        let mut strip = vec![idx];
        let mut strip_area = areas[idx];
        let mut best_worst = worst_aspect_ratio(&[areas[idx]], shorter_side);

        let mut next = idx + 1;
        while next < items.len() {
            let mut candidate_areas: Vec<f64> = strip.iter().map(|&i| areas[i]).collect();
            candidate_areas.push(areas[next]);
            let candidate_worst = worst_aspect_ratio(&candidate_areas, shorter_side);
            if candidate_worst > best_worst {
                // Adding this item worsens the aspect ratio — stop
                break;
            }
            best_worst = candidate_worst;
            strip.push(next);
            strip_area += areas[next];
            next += 1;
        }

        // Layout this strip
        let strip_items: Vec<(usize, f64)> = strip
            .iter()
            .map(|&i| (items[i].0, areas[i]))
            .collect();
        let new_cells = layout_strip(&strip_items, strip_area, &mut remaining);
        cells.extend(new_cells);

        idx = next;
    }

    cells
}

/// Compute the worst aspect ratio for a strip of items along the given side length.
fn worst_aspect_ratio(areas: &[f64], side: f64) -> f64 {
    let total: f64 = areas.iter().sum();
    if total <= 0.0 || side <= 0.0 {
        return f64::INFINITY;
    }
    // The strip occupies total/side along the shorter dimension
    let strip_length = total / side;
    let mut worst = 0.0_f64;
    for &a in areas {
        let cell_side = a / strip_length;
        let ratio = if strip_length > cell_side {
            strip_length / cell_side
        } else {
            cell_side / strip_length
        };
        worst = worst.max(ratio);
    }
    worst
}

/// Layout a finalized strip within the remaining rectangle, shrinking it.
fn layout_strip(
    items: &[(usize, f64)],
    strip_area: f64,
    remaining: &mut BoundingBox,
) -> Vec<TreemapCell> {
    let mut cells = Vec::with_capacity(items.len());
    let horizontal = remaining.w <= remaining.h;

    if horizontal {
        // Strip fills from the top, height = strip_area / width
        let strip_h = (strip_area / f64::from(remaining.w)) as f32;
        let strip_h = strip_h.min(remaining.h);
        let mut x = remaining.x;
        for &(index, area) in items {
            let cell_w = if strip_h > 0.0 {
                (area as f32 / strip_h).min(remaining.x + remaining.w - x)
            } else {
                0.0
            };
            cells.push(TreemapCell {
                index,
                bounds: BoundingBox::new(x, remaining.y, cell_w, strip_h),
            });
            x += cell_w;
        }
        remaining.y += strip_h;
        remaining.h -= strip_h;
        remaining.h = remaining.h.max(0.0);
    } else {
        // Strip fills from the left, width = strip_area / height
        let strip_w = (strip_area / f64::from(remaining.h)) as f32;
        let strip_w = strip_w.min(remaining.w);
        let mut y = remaining.y;
        for &(index, area) in items {
            let cell_h = if strip_w > 0.0 {
                (area as f32 / strip_w).min(remaining.y + remaining.h - y)
            } else {
                0.0
            };
            cells.push(TreemapCell {
                index,
                bounds: BoundingBox::new(remaining.x, y, strip_w, cell_h),
            });
            y += cell_h;
        }
        remaining.x += strip_w;
        remaining.w -= strip_w;
        remaining.w = remaining.w.max(0.0);
    }

    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let cells = squarified_layout(&[], BoundingBox::new(0.0, 0.0, 400.0, 300.0));
        assert!(cells.is_empty());
    }

    #[test]
    fn all_zeros() {
        let cells = squarified_layout(&[0.0, 0.0, 0.0], BoundingBox::new(0.0, 0.0, 400.0, 300.0));
        assert!(cells.is_empty());
    }

    #[test]
    fn single_item_fills_container() {
        let container = BoundingBox::new(10.0, 20.0, 400.0, 300.0);
        let cells = squarified_layout(&[100.0], container);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].index, 0);
        assert!((cells[0].bounds.x - 10.0).abs() < 1.0);
        assert!((cells[0].bounds.y - 20.0).abs() < 1.0);
        assert!((cells[0].bounds.w - 400.0).abs() < 1.0);
        assert!((cells[0].bounds.h - 300.0).abs() < 1.0);
    }

    #[test]
    fn total_area_matches_container() {
        let container = BoundingBox::new(0.0, 0.0, 400.0, 300.0);
        let values = vec![30.0, 20.0, 15.0, 10.0, 5.0];
        let cells = squarified_layout(&values, container);
        assert_eq!(cells.len(), 5);
        let total: f32 = cells.iter().map(|c| c.bounds.w * c.bounds.h).sum();
        let expected = 400.0 * 300.0;
        assert!(
            (total - expected).abs() < expected * 0.01,
            "total area {total} should be close to {expected}"
        );
    }

    #[test]
    fn no_zero_dimensions() {
        let container = BoundingBox::new(0.0, 0.0, 400.0, 300.0);
        let values = vec![30.0, 20.0, 15.0, 10.0, 5.0];
        let cells = squarified_layout(&values, container);
        for cell in &cells {
            assert!(cell.bounds.w > 0.0, "cell {} has zero width", cell.index);
            assert!(cell.bounds.h > 0.0, "cell {} has zero height", cell.index);
        }
    }

    #[test]
    fn indices_preserved() {
        let container = BoundingBox::new(0.0, 0.0, 400.0, 300.0);
        let values = vec![10.0, 0.0, 30.0, 5.0];
        let cells = squarified_layout(&values, container);
        // Zero value should be filtered out
        assert_eq!(cells.len(), 3);
        let indices: Vec<usize> = cells.iter().map(|c| c.index).collect();
        assert!(indices.contains(&0));
        assert!(!indices.contains(&1)); // zero value
        assert!(indices.contains(&2));
        assert!(indices.contains(&3));
    }

    #[test]
    fn two_equal_items() {
        let container = BoundingBox::new(0.0, 0.0, 200.0, 100.0);
        let cells = squarified_layout(&[50.0, 50.0], container);
        assert_eq!(cells.len(), 2);
        let a0 = cells[0].bounds.w * cells[0].bounds.h;
        let a1 = cells[1].bounds.w * cells[1].bounds.h;
        assert!(
            (a0 - a1).abs() < 1.0,
            "equal values should have equal areas: {a0} vs {a1}"
        );
    }

    #[test]
    fn many_items() {
        let container = BoundingBox::new(0.0, 0.0, 800.0, 600.0);
        let values: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let cells = squarified_layout(&values, container);
        assert_eq!(cells.len(), 50);
    }
}
