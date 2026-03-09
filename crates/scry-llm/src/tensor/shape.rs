use arrayvec::ArrayVec;

use crate::error::{Result, ScryLlmError};

/// Maximum number of dimensions supported.
pub const MAX_DIMS: usize = 6;

/// Compact shape stored on the stack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: ArrayVec<usize, MAX_DIMS>,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        let mut d = ArrayVec::new();
        for &dim in dims {
            d.push(dim);
        }
        Self { dims: d }
    }

    pub fn scalar() -> Self {
        Self::new(&[1])
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Compute contiguous row-major strides.
    pub fn strides(&self) -> ArrayVec<usize, MAX_DIMS> {
        let n = self.dims.len();
        let mut s = ArrayVec::new();
        if n == 0 {
            return s;
        }
        for _ in 0..n {
            s.push(0);
        }
        s[n - 1] = 1;
        for i in (0..n - 1).rev() {
            s[i] = s[i + 1] * self.dims[i + 1];
        }
        s
    }

    /// Broadcast two shapes together, returning the output shape.
    ///
    /// # Errors
    ///
    /// Returns [`ScryLlmError::BroadcastIncompatible`] if the shapes cannot be broadcast together.
    pub fn broadcast(a: &Shape, b: &Shape) -> Result<Shape> {
        let max_ndim = a.ndim().max(b.ndim());
        let mut result = ArrayVec::new();

        for i in 0..max_ndim {
            let da = if i < a.ndim() {
                a.dims[a.ndim() - 1 - i]
            } else {
                1
            };
            let db = if i < b.ndim() {
                b.dims[b.ndim() - 1 - i]
            } else {
                1
            };

            if da == db {
                result.push(da);
            } else if da == 1 {
                result.push(db);
            } else if db == 1 {
                result.push(da);
            } else {
                return Err(ScryLlmError::BroadcastIncompatible {
                    a: a.dims.to_vec(),
                    b: b.dims.to_vec(),
                });
            }
        }

        result.reverse();
        Ok(Shape { dims: result })
    }

    /// Compute broadcast strides: dimensions that are 1 in the original but
    /// larger in the broadcast result get stride 0.
    pub fn broadcast_strides(&self, target: &Shape) -> ArrayVec<usize, MAX_DIMS> {
        let base_strides = self.strides();
        let mut result = ArrayVec::new();
        let offset = target.ndim() - self.ndim();

        for i in 0..target.ndim() {
            if i < offset {
                // This dimension was prepended (doesn't exist in self)
                result.push(0);
            } else {
                let self_idx = i - offset;
                if self.dims[self_idx] == 1 && target.dims[i] != 1 {
                    result.push(0);
                } else {
                    result.push(base_strides[self_idx]);
                }
            }
        }
        result
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_2d() {
        let s = Shape::new(&[3, 4]);
        let strides = s.strides();
        assert_eq!(&strides[..], &[4, 1]);
    }

    #[test]
    fn strides_3d() {
        let s = Shape::new(&[2, 3, 4]);
        let strides = s.strides();
        assert_eq!(&strides[..], &[12, 4, 1]);
    }

    #[test]
    fn broadcast_same() {
        let a = Shape::new(&[3, 4]);
        let b = Shape::new(&[3, 4]);
        let c = Shape::broadcast(&a, &b).unwrap();
        assert_eq!(c.dims(), &[3, 4]);
    }

    #[test]
    fn broadcast_scalar() {
        let a = Shape::new(&[3, 4]);
        let b = Shape::new(&[1]);
        let c = Shape::broadcast(&a, &b).unwrap();
        assert_eq!(c.dims(), &[3, 4]);
    }

    #[test]
    fn broadcast_row_col() {
        let a = Shape::new(&[3, 1]);
        let b = Shape::new(&[1, 4]);
        let c = Shape::broadcast(&a, &b).unwrap();
        assert_eq!(c.dims(), &[3, 4]);
    }

    #[test]
    fn broadcast_incompatible() {
        let a = Shape::new(&[3, 4]);
        let b = Shape::new(&[3, 5]);
        assert!(Shape::broadcast(&a, &b).is_err());
    }

    #[test]
    fn broadcast_strides_expand() {
        let a = Shape::new(&[1, 4]);
        let target = Shape::new(&[3, 4]);
        let strides = a.broadcast_strides(&target);
        // dim 0: was 1, target 3 → stride 0
        // dim 1: was 4, target 4 → stride 1
        assert_eq!(&strides[..], &[0, 1]);
    }

    #[test]
    fn numel() {
        assert_eq!(Shape::new(&[2, 3, 4]).numel(), 24);
        assert_eq!(Shape::scalar().numel(), 1);
    }
}
