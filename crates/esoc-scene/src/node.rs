// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scene graph node types.

use crate::mark::{Mark, MarkBatch, MarkKey};
use crate::transform::Affine2D;

/// A handle to a node in the scene graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId {
    /// Slot index in the arena.
    pub index: u32,
    /// Generation counter for ABA safety.
    pub generation: u32,
}

/// What a node contains.
pub enum NodeContent {
    /// A grouping container (no visual output).
    Container,
    /// A single visual mark.
    Mark(Mark),
    /// A batch of homogeneous marks (for instanced rendering).
    Batch(MarkBatch),
}

/// A node in the scene graph.
pub struct Node {
    /// Parent node.
    pub parent: Option<NodeId>,
    /// Child nodes.
    pub children: Vec<NodeId>,
    /// Local transform.
    pub transform: Affine2D,
    /// Whether this node clips its children to its bounds.
    pub clip: bool,
    /// Z-order for sibling sorting.
    pub z_order: i32,
    /// Opacity multiplier `[0, 1]`.
    pub opacity: f32,
    /// The visual content of this node.
    pub content: NodeContent,
    /// Optional key for scene diffing (transitions).
    pub key: Option<MarkKey>,
}

impl Node {
    /// Create a container node.
    pub fn container() -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            transform: Affine2D::IDENTITY,
            clip: false,
            z_order: 0,
            opacity: 1.0,
            content: NodeContent::Container,
            key: None,
        }
    }

    /// Create a node wrapping a single mark.
    pub fn with_mark(mark: Mark) -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            transform: Affine2D::IDENTITY,
            clip: false,
            z_order: 0,
            opacity: 1.0,
            content: NodeContent::Mark(mark),
            key: None,
        }
    }

    /// Create a node wrapping a mark batch.
    pub fn with_batch(batch: MarkBatch) -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            transform: Affine2D::IDENTITY,
            clip: false,
            z_order: 0,
            opacity: 1.0,
            content: NodeContent::Batch(batch),
            key: None,
        }
    }

    /// Set the transform and return self.
    pub fn transform(mut self, t: Affine2D) -> Self {
        self.transform = t;
        self
    }

    /// Set `z_order` and return self.
    pub fn z_order(mut self, z: i32) -> Self {
        self.z_order = z;
        self
    }

    /// Set the key and return self.
    pub fn key(mut self, k: MarkKey) -> Self {
        self.key = Some(k);
        self
    }
}
