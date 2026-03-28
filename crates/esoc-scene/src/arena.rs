// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generational-index arena scene graph.

use crate::node::{Node, NodeContent, NodeId};
use crate::transform::Affine2D;

/// Slot in the arena: either occupied or free.
#[allow(clippy::large_enum_variant)]
enum Slot {
    Occupied(Node),
    Free { next_free: Option<u32> },
}

/// Arena-based scene graph with generational indices.
pub struct SceneGraph {
    slots: Vec<Slot>,
    generation: Vec<u32>,
    free_head: Option<u32>,
    root: Option<NodeId>,
}

impl SceneGraph {
    /// Create an empty scene graph.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            generation: Vec::new(),
            free_head: None,
            root: None,
        }
    }

    /// Create a scene graph with a root container node.
    pub fn with_root() -> Self {
        let mut sg = Self::new();
        let root = sg.insert(Node::container());
        sg.root = Some(root);
        sg
    }

    /// The root node, if any.
    pub fn root(&self) -> Option<NodeId> {
        self.root
    }

    /// Set the root node.
    pub fn set_root(&mut self, id: NodeId) {
        self.root = Some(id);
    }

    /// Insert a node into the arena, returning its ID.
    pub fn insert(&mut self, node: Node) -> NodeId {
        if let Some(idx) = self.free_head {
            let i = idx as usize;
            if let Slot::Free { next_free } = self.slots[i] {
                self.free_head = next_free;
            }
            self.generation[i] += 1;
            self.slots[i] = Slot::Occupied(node);
            NodeId {
                index: idx,
                generation: self.generation[i],
            }
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Slot::Occupied(node));
            self.generation.push(0);
            NodeId {
                index: idx,
                generation: 0,
            }
        }
    }

    /// Remove a node by ID. Returns the node if it existed.
    pub fn remove(&mut self, id: NodeId) -> Option<Node> {
        let i = id.index as usize;
        if i >= self.slots.len() || self.generation[i] != id.generation {
            return None;
        }
        if let Slot::Occupied(_) = &self.slots[i] {
            let old = std::mem::replace(
                &mut self.slots[i],
                Slot::Free {
                    next_free: self.free_head,
                },
            );
            self.free_head = Some(id.index);
            match old {
                Slot::Occupied(node) => Some(node),
                Slot::Free { .. } => None,
            }
        } else {
            None
        }
    }

    /// Get a reference to a node.
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        let i = id.index as usize;
        if i >= self.slots.len() || self.generation[i] != id.generation {
            return None;
        }
        match &self.slots[i] {
            Slot::Occupied(node) => Some(node),
            Slot::Free { .. } => None,
        }
    }

    /// Get a mutable reference to a node.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        let i = id.index as usize;
        if i >= self.slots.len() || self.generation[i] != id.generation {
            return None;
        }
        match &mut self.slots[i] {
            Slot::Occupied(node) => Some(node),
            Slot::Free { .. } => None,
        }
    }

    /// Add a child to a parent node.
    pub fn add_child(&mut self, parent: NodeId, child: NodeId) {
        // Set parent on child
        if let Some(c) = self.get_mut(child) {
            c.parent = Some(parent);
        }
        // Add to parent's children list
        if let Some(p) = self.get_mut(parent) {
            p.children.push(child);
        }
    }

    /// Insert a node as a child of `parent`.
    pub fn insert_child(&mut self, parent: NodeId, node: Node) -> NodeId {
        let id = self.insert(node);
        self.add_child(parent, id);
        id
    }

    /// Number of live nodes.
    pub fn len(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, Slot::Occupied(_)))
            .count()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all live (`NodeId`, `&Node`) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied(node) => Some((
                    NodeId {
                        index: i as u32,
                        generation: self.generation[i],
                    },
                    node,
                )),
                Slot::Free { .. } => None,
            })
    }

    /// Compute the world transform for a node by walking up the parent chain.
    pub fn world_transform(&self, id: NodeId) -> Affine2D {
        let mut chain = Vec::new();
        let mut current = Some(id);
        while let Some(cid) = current {
            if let Some(node) = self.get(cid) {
                chain.push(node.transform);
                current = node.parent;
            } else {
                break;
            }
        }
        let mut result = Affine2D::IDENTITY;
        for t in chain.into_iter().rev() {
            result = result.then(t);
        }
        result
    }

    /// Collect all leaf marks in z-order (depth-first, sorted by `z_order` within siblings).
    pub fn collect_marks(&self) -> Vec<(NodeId, Affine2D)> {
        let mut result = Vec::new();
        if let Some(root) = self.root {
            self.collect_marks_recursive(root, Affine2D::IDENTITY, &mut result);
        }
        result
    }

    fn collect_marks_recursive(
        &self,
        id: NodeId,
        parent_transform: Affine2D,
        result: &mut Vec<(NodeId, Affine2D)>,
    ) {
        let Some(node) = self.get(id) else { return };
        let world = parent_transform.then(node.transform);

        match &node.content {
            NodeContent::Container => {}
            NodeContent::Mark(_) | NodeContent::Batch(_) => {
                result.push((id, world));
            }
        }

        // Sort children indices by z_order without cloning
        let mut indices: Vec<usize> = (0..node.children.len()).collect();
        indices.sort_by_key(|&i| self.get(node.children[i]).map_or(0, |n| n.z_order));

        for i in indices {
            // Re-fetch node since we can't hold a reference across recursive calls
            let child_id = self.get(id).map(|n| n.children[i]);
            if let Some(cid) = child_id {
                self.collect_marks_recursive(cid, world, result);
            }
        }
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mark::Mark;
    use crate::style::StrokeStyle;

    #[test]
    fn insert_and_get() {
        let mut sg = SceneGraph::new();
        let id = sg.insert(Node::container());
        assert!(sg.get(id).is_some());
        assert_eq!(sg.len(), 1);
    }

    #[test]
    fn remove_and_reuse() {
        let mut sg = SceneGraph::new();
        let id1 = sg.insert(Node::container());
        sg.remove(id1);
        assert!(sg.get(id1).is_none());
        assert_eq!(sg.len(), 0);

        // Next insert reuses the slot
        let id2 = sg.insert(Node::container());
        assert_eq!(id2.index, id1.index);
        assert_ne!(id2.generation, id1.generation);
    }

    #[test]
    fn parent_child() {
        let mut sg = SceneGraph::with_root();
        let root = sg.root().unwrap();
        let child = sg.insert_child(root, Node::container());

        let root_node = sg.get(root).unwrap();
        assert_eq!(root_node.children.len(), 1);
        assert_eq!(root_node.children[0], child);

        let child_node = sg.get(child).unwrap();
        assert_eq!(child_node.parent, Some(root));
    }

    #[test]
    fn collect_marks_ordering() {
        let mut sg = SceneGraph::with_root();
        let root = sg.root().unwrap();

        let mut n1 = Node::with_mark(Mark::Rule(crate::mark::RuleMark {
            segments: vec![],
            stroke: StrokeStyle::default(),
        }));
        n1.z_order = 2;

        let mut n2 = Node::with_mark(Mark::Rule(crate::mark::RuleMark {
            segments: vec![],
            stroke: StrokeStyle::default(),
        }));
        n2.z_order = 1;

        sg.insert_child(root, n1);
        sg.insert_child(root, n2);

        let marks = sg.collect_marks();
        assert_eq!(marks.len(), 2);
        // z_order=1 should come first
        let first = sg.get(marks[0].0).unwrap();
        let second = sg.get(marks[1].0).unwrap();
        assert!(first.z_order <= second.z_order);
    }
}
