// SPDX-License-Identifier: MIT OR Apache-2.0
//! Encoding: maps data fields to visual channels.

/// A data field type.
#[derive(Clone, Debug)]
pub enum FieldType {
    /// Continuous numeric data.
    Quantitative,
    /// Categorical/nominal data.
    Nominal,
    /// Ordered categorical data.
    Ordinal,
    /// Temporal data.
    Temporal,
}

/// A visual channel.
#[derive(Clone, Debug)]
pub enum Channel {
    /// X position.
    X,
    /// Y position.
    Y,
    /// Fill color.
    Color,
    /// Marker size.
    Size,
    /// Marker shape (not yet supported).
    #[doc(hidden)]
    Shape,
    /// Opacity (not yet supported).
    #[doc(hidden)]
    Opacity,
    /// Text content (not yet supported).
    #[doc(hidden)]
    Text,
}

/// An encoding maps a data field to a visual channel.
#[derive(Clone, Debug)]
pub struct Encoding {
    /// Visual channel.
    pub channel: Channel,
    /// Field type.
    pub field_type: FieldType,
    /// Data accessor (column index or field name).
    pub field: FieldAccessor,
    /// Optional title for this encoding (used in legends/axis labels).
    pub title: Option<String>,
}

/// How to access a data field.
#[derive(Clone, Debug)]
pub enum FieldAccessor {
    /// Column index.
    Index(usize),
    /// Field name.
    Name(String),
}

/// Convenience constructors.
pub fn quantitative(index: usize) -> Encoding {
    Encoding {
        channel: Channel::X,
        field_type: FieldType::Quantitative,
        field: FieldAccessor::Index(index),
        title: None,
    }
}

/// Create a nominal encoding.
pub fn nominal(index: usize) -> Encoding {
    Encoding {
        channel: Channel::X,
        field_type: FieldType::Nominal,
        field: FieldAccessor::Index(index),
        title: None,
    }
}

impl Encoding {
    /// Set the channel.
    pub fn channel(mut self, ch: Channel) -> Self {
        self.channel = ch;
        self
    }

    /// Set a title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}
