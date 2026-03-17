// SPDX-License-Identifier: MIT OR Apache-2.0
//! Feature properties: scalar key-value pairs.

use std::collections::HashMap;

/// A scalar property value.
///
/// Nested objects/arrays from `GeoJSON` are serialized back to JSON strings
/// to avoid recursive types.
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyValue {
    /// String value.
    String(String),
    /// Numeric value.
    Number(f64),
    /// Boolean value.
    Bool(bool),
    /// Null / missing value.
    Null,
}

impl PropertyValue {
    /// Get as string, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as number, if it is one.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as bool, if it is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Check if null.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

/// Feature properties: a map from string keys to scalar values.
pub type Properties = HashMap<String, PropertyValue>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_value_accessors() {
        let s = PropertyValue::String("hello".into());
        assert_eq!(s.as_str(), Some("hello"));
        assert_eq!(s.as_f64(), None);

        let n = PropertyValue::Number(42.0);
        assert_eq!(n.as_f64(), Some(42.0));
        assert_eq!(n.as_str(), None);

        let b = PropertyValue::Bool(true);
        assert_eq!(b.as_bool(), Some(true));

        let null = PropertyValue::Null;
        assert!(null.is_null());
        assert_eq!(null.as_str(), None);
    }
}
