// SPDX-License-Identifier: MIT OR Apache-2.0
//! Layered drawing surface that collects elements and defs.

use crate::element::{ClipDef, DrawElement, LinearGradient};
use crate::layer::Layer;

/// A layered drawing surface that accumulates elements.
///
/// Elements are sorted by layer when rendered, ensuring correct z-ordering.
#[derive(Clone, Debug)]
pub struct Canvas {
    /// Canvas width in pixels.
    pub width: f64,
    /// Canvas height in pixels.
    pub height: f64,
    /// Accumulated drawing elements.
    elements: Vec<DrawElement>,
    /// Gradient definitions.
    gradients: Vec<LinearGradient>,
    /// Clip path definitions.
    clips: Vec<ClipDef>,
    /// Counter for generating unique IDs.
    id_counter: usize,
}

impl Canvas {
    /// Create a new canvas with the given dimensions.
    pub fn new(width: f64, height: f64) -> Self {
        Self {
            width,
            height,
            elements: Vec::new(),
            gradients: Vec::new(),
            clips: Vec::new(),
            id_counter: 0,
        }
    }

    /// Add an element to the canvas.
    pub fn add(&mut self, element: DrawElement) {
        self.elements.push(element);
    }

    /// Add a linear gradient definition and return its ID.
    pub fn add_gradient(&mut self, mut gradient: LinearGradient) -> String {
        if gradient.id.is_empty() {
            gradient.id = self.next_id("grad");
        }
        let id = gradient.id.clone();
        self.gradients.push(gradient);
        id
    }

    /// Add a clip rectangle and return its ID.
    pub fn add_clip(&mut self, mut clip: ClipDef) -> String {
        if clip.id.is_empty() {
            clip.id = self.next_id("clip");
        }
        let id = clip.id.clone();
        self.clips.push(clip);
        id
    }

    /// Get all elements sorted by layer (bottom-to-top).
    pub fn elements_sorted(&self) -> Vec<&DrawElement> {
        let mut elems: Vec<&DrawElement> = self.elements.iter().collect();
        elems.sort_by_key(|e| e.layer);
        elems
    }

    /// Get all gradient definitions.
    pub fn gradients(&self) -> &[LinearGradient] {
        &self.gradients
    }

    /// Get all clip definitions.
    pub fn clips(&self) -> &[ClipDef] {
        &self.clips
    }

    /// Generate a unique ID with the given prefix.
    fn next_id(&mut self, prefix: &str) -> String {
        self.id_counter += 1;
        format!("{prefix}_{}", self.id_counter)
    }

    /// Bulk-add elements from another canvas (used for compositing).
    pub fn merge(&mut self, other: &Self) {
        self.elements.extend(other.elements.iter().cloned());
        self.gradients.extend(other.gradients.iter().cloned());
        self.clips.extend(other.clips.iter().cloned());
    }

    /// Convenience: add an element on the specified layer.
    pub fn draw(&mut self, kind: crate::element::Element, layer: Layer) {
        self.add(DrawElement::new(kind, layer));
    }
}
