// SPDX-License-Identifier: MIT OR Apache-2.0
//! Connected component labeling using two-pass union-find.
//!
//! Labels each connected foreground region in a binary image with a unique
//! integer ID and computes per-component statistics (area, bounding box,
//! centroid).

use crate::error::Result;
use crate::image::{Gray, ImageBuf};

/// Pixel connectivity mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Connectivity {
    /// 4-connected: up, down, left, right.
    Four,
    /// 8-connected: includes diagonals.
    Eight,
}

/// Per-component statistics.
#[derive(Clone, Debug)]
pub struct ComponentStats {
    /// Number of pixels in the component.
    pub area: u32,
    /// Bounding box: (x_min, y_min, x_max, y_max) inclusive.
    pub bbox: (u32, u32, u32, u32),
    /// Centroid (mean x, mean y).
    pub centroid: (f64, f64),
}

/// Result of connected component labeling.
#[derive(Clone, Debug)]
pub struct ConnectedComponents {
    /// Per-pixel label (0 = background, 1.. = component IDs). Row-major.
    pub labels: Vec<u32>,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Number of foreground labels (component IDs are 1..=num_labels).
    pub num_labels: u32,
    /// Statistics for each component (index 0 = label 1, etc.).
    pub stats: Vec<ComponentStats>,
}

/// Label connected foreground components in a binary image.
///
/// Input pixel values > 0.5 are treated as foreground.
///
/// # Example
///
/// ```
/// use scry_cv::prelude::*;
/// use scry_cv::components::labeling::{connected_components, Connectivity};
///
/// let mut data = vec![0.0f32; 16 * 16];
/// // Small blob
/// data[3 * 16 + 3] = 1.0;
/// data[3 * 16 + 4] = 1.0;
/// data[4 * 16 + 3] = 1.0;
/// let img = GrayImageF::from_vec(data, 16, 16).unwrap();
/// let cc = connected_components(&img, Connectivity::Four).unwrap();
/// assert_eq!(cc.num_labels, 1);
/// ```
pub fn connected_components(
    img: &ImageBuf<f32, Gray>,
    connectivity: Connectivity,
) -> Result<ConnectedComponents> {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let data = img.as_slice();

    let mut labels = vec![0u32; w * h];
    let mut uf = UnionFind::new(w * h / 4 + 16); // rough capacity
    let mut next_label = 1u32;

    // --- Pass 1: assign provisional labels ---
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if data[idx] <= 0.5 {
                continue;
            }

            // Collect labels of already-visited foreground neighbors
            let mut neighbors = [0u32; 4];
            let mut nn = 0usize;

            // Left
            if x > 0 && labels[idx - 1] > 0 {
                neighbors[nn] = labels[idx - 1];
                nn += 1;
            }
            // Above
            if y > 0 && labels[idx - w] > 0 {
                neighbors[nn] = labels[idx - w];
                nn += 1;
            }

            if connectivity == Connectivity::Eight {
                // Above-left
                if x > 0 && y > 0 && labels[(y - 1) * w + x - 1] > 0 {
                    neighbors[nn] = labels[(y - 1) * w + x - 1];
                    nn += 1;
                }
                // Above-right
                if x + 1 < w && y > 0 && labels[(y - 1) * w + x + 1] > 0 {
                    neighbors[nn] = labels[(y - 1) * w + x + 1];
                    nn += 1;
                }
            }

            if nn == 0 {
                // New label
                uf.ensure_capacity(next_label as usize + 1);
                labels[idx] = next_label;
                next_label += 1;
            } else {
                // Find minimum neighbor label and union all
                let mut min_label = neighbors[0];
                for i in 1..nn {
                    if neighbors[i] < min_label {
                        min_label = neighbors[i];
                    }
                }
                labels[idx] = min_label;
                for i in 0..nn {
                    uf.union(min_label as usize, neighbors[i] as usize);
                }
            }
        }
    }

    // --- Pass 2: resolve labels to canonical IDs ---
    let mut canonical = vec![0u32; next_label as usize];
    let mut final_label = 0u32;
    for l in 1..next_label {
        let root = uf.find(l as usize) as u32;
        if canonical[root as usize] == 0 {
            final_label += 1;
            canonical[root as usize] = final_label;
        }
        canonical[l as usize] = canonical[root as usize];
    }

    for l in &mut labels {
        if *l > 0 {
            *l = canonical[*l as usize];
        }
    }

    // --- Pass 3: compute stats ---
    let num_labels = final_label;
    let mut stats: Vec<ComponentStats> = (0..num_labels)
        .map(|_| ComponentStats {
            area: 0,
            bbox: (u32::MAX, u32::MAX, 0, 0),
            centroid: (0.0, 0.0),
        })
        .collect();

    for y in 0..h {
        for x in 0..w {
            let l = labels[y * w + x];
            if l == 0 {
                continue;
            }
            let s = &mut stats[(l - 1) as usize];
            s.area += 1;
            s.bbox.0 = s.bbox.0.min(x as u32);
            s.bbox.1 = s.bbox.1.min(y as u32);
            s.bbox.2 = s.bbox.2.max(x as u32);
            s.bbox.3 = s.bbox.3.max(y as u32);
            s.centroid.0 += x as f64;
            s.centroid.1 += y as f64;
        }
    }

    for s in &mut stats {
        if s.area > 0 {
            s.centroid.0 /= f64::from(s.area);
            s.centroid.1 /= f64::from(s.area);
        }
    }

    Ok(ConnectedComponents {
        labels,
        width: img.width(),
        height: img.height(),
        num_labels,
        stats,
    })
}

// ---------- Union-Find (disjoint set) ----------

struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(capacity: usize) -> Self {
        let mut parent = Vec::with_capacity(capacity);
        let mut rank = Vec::with_capacity(capacity);
        for i in 0..capacity {
            parent.push(i as u32);
            rank.push(0);
        }
        Self { parent, rank }
    }

    fn ensure_capacity(&mut self, n: usize) {
        while self.parent.len() <= n {
            let i = self.parent.len() as u32;
            self.parent.push(i);
            self.rank.push(0);
        }
    }

    fn find(&mut self, x: usize) -> usize {
        self.ensure_capacity(x);
        if self.parent[x] != x as u32 {
            self.parent[x] = self.find(self.parent[x] as usize) as u32; // path compression
        }
        self.parent[x] as usize
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Union by rank
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb as u32,
            std::cmp::Ordering::Greater => self.parent[rb] = ra as u32,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra as u32;
                self.rank[ra] += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_blob() {
        let mut data = vec![0.0f32; 16 * 16];
        for y in 2..6 {
            for x in 2..6 {
                data[y * 16 + x] = 1.0;
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, 16, 16).unwrap();
        let cc = connected_components(&img, Connectivity::Four).unwrap();
        assert_eq!(cc.num_labels, 1);
        assert_eq!(cc.stats[0].area, 16);
    }

    #[test]
    fn two_separated_blobs() {
        let mut data = vec![0.0f32; 32 * 32];
        // Blob 1: top-left
        for y in 1..4 {
            for x in 1..4 {
                data[y * 32 + x] = 1.0;
            }
        }
        // Blob 2: bottom-right
        for y in 20..23 {
            for x in 20..23 {
                data[y * 32 + x] = 1.0;
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, 32, 32).unwrap();
        let cc = connected_components(&img, Connectivity::Four).unwrap();
        assert_eq!(cc.num_labels, 2);
        assert_eq!(cc.stats[0].area, 9);
        assert_eq!(cc.stats[1].area, 9);
    }

    #[test]
    fn diagonal_four_vs_eight() {
        // Two pixels touching diagonally
        let mut data = vec![0.0f32; 8 * 8];
        data[2 * 8 + 2] = 1.0;
        data[3 * 8 + 3] = 1.0;
        let img = ImageBuf::<f32, Gray>::from_vec(data, 8, 8).unwrap();

        let cc4 = connected_components(&img, Connectivity::Four).unwrap();
        assert_eq!(cc4.num_labels, 2, "4-conn: diagonal pixels are separate");

        let cc8 = connected_components(&img, Connectivity::Eight).unwrap();
        assert_eq!(cc8.num_labels, 1, "8-conn: diagonal pixels are connected");
    }

    #[test]
    fn empty_image_returns_zero_labels() {
        let img = ImageBuf::<f32, Gray>::new(16, 16).unwrap();
        let cc = connected_components(&img, Connectivity::Four).unwrap();
        assert_eq!(cc.num_labels, 0);
        assert!(cc.stats.is_empty());
    }

    #[test]
    fn centroid_is_correct() {
        // 3x3 block at (4,4)-(6,6)
        let mut data = vec![0.0f32; 16 * 16];
        for y in 4..7 {
            for x in 4..7 {
                data[y * 16 + x] = 1.0;
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, 16, 16).unwrap();
        let cc = connected_components(&img, Connectivity::Four).unwrap();
        assert_eq!(cc.num_labels, 1);
        let c = &cc.stats[0].centroid;
        assert!((c.0 - 5.0).abs() < 1e-9, "centroid x = {}", c.0);
        assert!((c.1 - 5.0).abs() < 1e-9, "centroid y = {}", c.1);
    }
}
