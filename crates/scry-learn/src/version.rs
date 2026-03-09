// SPDX-License-Identifier: MIT OR Apache-2.0
//! Model schema versioning for serde serialization compatibility.

/// Current schema version for serde model serialization.
pub const SCHEMA_VERSION: u32 = 1;

/// Called after deserialization to verify version compatibility.
pub fn check_schema_version(found: u32) -> crate::error::Result<()> {
    if found != SCHEMA_VERSION {
        return Err(crate::error::ScryLearnError::InvalidParameter(format!(
            "model schema version mismatch: expected {SCHEMA_VERSION}, found {found} \
             — model was serialized with an incompatible scry-learn version"
        )));
    }
    Ok(())
}
