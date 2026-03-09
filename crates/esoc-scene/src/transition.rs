// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scene diffing for transitions (enter/update/exit).

use crate::mark::MarkKey;
use crate::node::NodeId;

/// An easing function for animated transitions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Easing {
    /// Linear interpolation.
    #[default]
    Linear,
    /// Ease in (slow start).
    EaseIn,
    /// Ease out (slow end).
    EaseOut,
    /// Ease in and out.
    EaseInOut,
}

impl Easing {
    /// Apply the easing function to a `t` in `[0, 1]`.
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => t * (2.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
        }
    }
}

/// Result of diffing two scenes by `MarkKey`.
#[derive(Clone, Debug, Default)]
pub struct SceneDiff {
    /// Nodes that appear only in the new scene (new data).
    pub enter: Vec<NodeId>,
    /// Nodes that exist in both scenes (matched by key).
    pub update: Vec<(NodeId, NodeId)>,
    /// Nodes that appear only in the old scene (removed data).
    pub exit: Vec<NodeId>,
}

/// Diff two sets of keyed node IDs.
pub fn diff_by_key(
    old: &[(MarkKey, NodeId)],
    new: &[(MarkKey, NodeId)],
) -> SceneDiff {
    use std::collections::HashMap;

    let old_map: HashMap<&MarkKey, NodeId> = old.iter().map(|(k, id)| (k, *id)).collect();
    let new_map: HashMap<&MarkKey, NodeId> = new.iter().map(|(k, id)| (k, *id)).collect();

    let mut diff = SceneDiff::default();

    for (key, new_id) in new {
        if let Some(&old_id) = old_map.get(key) {
            diff.update.push((old_id, *new_id));
        } else {
            diff.enter.push(*new_id);
        }
    }

    for (key, old_id) in old {
        if !new_map.contains_key(key) {
            diff.exit.push(*old_id);
        }
    }

    diff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn easing_endpoints() {
        for easing in [Easing::Linear, Easing::EaseIn, Easing::EaseOut, Easing::EaseInOut] {
            assert!((easing.apply(0.0)).abs() < 1e-6, "{easing:?} at 0");
            assert!((easing.apply(1.0) - 1.0).abs() < 1e-6, "{easing:?} at 1");
        }
    }

    #[test]
    fn diff_enter_update_exit() {
        let old = vec![
            (MarkKey::Index(0), NodeId { index: 0, generation: 0 }),
            (MarkKey::Index(1), NodeId { index: 1, generation: 0 }),
            (MarkKey::Index(2), NodeId { index: 2, generation: 0 }),
        ];
        let new = vec![
            (MarkKey::Index(1), NodeId { index: 10, generation: 0 }),
            (MarkKey::Index(2), NodeId { index: 11, generation: 0 }),
            (MarkKey::Index(3), NodeId { index: 12, generation: 0 }),
        ];

        let diff = diff_by_key(&old, &new);
        assert_eq!(diff.enter.len(), 1); // Index(3) is new
        assert_eq!(diff.update.len(), 2); // Index(1), Index(2) matched
        assert_eq!(diff.exit.len(), 1); // Index(0) removed
    }
}
