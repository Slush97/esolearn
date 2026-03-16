//! Named constants replacing magic numbers throughout scry-learn.
//!
//! Grouped by purpose. Each constant documents where it is used.

// ─── Near-zero thresholds ────────────────────────────────────────────────────

/// Threshold below which a value is treated as effectively zero.
/// Used for: column norm checks (lasso, elastic_net), probability clamping
/// (gradient boosting log-loss), PCA variance guards, Jacobi element skipping,
/// isotonic regression interpolation.
pub(crate) const NEAR_ZERO: f64 = 1e-15;

/// Threshold for singular matrix detection in Gauss-Jordan elimination
/// and Newton-Raphson Hessian denominators.
pub(crate) const SINGULAR_THRESHOLD: f64 = 1e-12;

/// Hessian regularization constant for Platt scaling Newton steps.
pub(crate) const PLATT_HESSIAN_REG: f64 = 1e-12;

/// Singular determinant threshold for Platt scaling 2×2 system.
pub(crate) const PLATT_SINGULAR_DET: f64 = 1e-20;

/// Minimum step size for Platt scaling Newton line search.
pub(crate) const PLATT_MIN_STEP: f64 = 1e-10;

/// Convergence threshold for Platt scaling parameter updates.
pub(crate) const PLATT_CONVERGENCE: f64 = 1e-9;

/// Probability clamping bounds for gradient boosting prior/leaf values.
pub(crate) const GBT_PROB_CLAMP: f64 = 1e-7;

// ─── Convergence tolerances ──────────────────────────────────────────────────

/// Default convergence tolerance for coordinate descent (Lasso, ElasticNet,
/// LinearSVC, LinearSVR).
pub(crate) const DEFAULT_TOL: f64 = 1e-4;

/// Stricter convergence tolerance for L-BFGS and logistic regression.
pub(crate) const STRICT_TOL: f64 = 1e-6;

/// Jacobi eigendecomposition convergence tolerance (off-diagonal Frobenius norm).
pub(crate) const JACOBI_TOL: f64 = 1e-12;

/// Maximum Jacobi sweeps in PCA eigendecomposition.
pub(crate) const JACOBI_MAX_SWEEPS: usize = 100;

// ─── Line search constants ──────────────────────────────────────────────────

/// Armijo sufficient decrease constant (c₁ in Wolfe conditions).
pub(crate) const ARMIJO_C: f64 = 1e-4;

/// Strong Wolfe curvature condition constant (c₂).
pub(crate) const WOLFE_C2: f64 = 0.9;

/// Backtracking factor for line search step reduction.
pub(crate) const LINE_SEARCH_BACKTRACK: f64 = 0.5;

/// Maximum line search iterations before giving up.
pub(crate) const LINE_SEARCH_MAX_ITER: usize = 20;

/// Steepest descent fallback step size scaling factor.
pub(crate) const STEEPEST_DESCENT_SCALE: f64 = 0.01;

// ─── L-BFGS constants ───────────────────────────────────────────────────────

/// Curvature condition threshold for L-BFGS history update.
/// Only add correction pair if s·y > this value.
pub(crate) const LBFGS_CURVATURE_THRESH: f64 = 1e-16;

// ─── Optimizer defaults ─────────────────────────────────────────────────────

/// Default momentum coefficient for SGD with momentum.
pub(crate) const SGD_MOMENTUM: f64 = 0.9;

/// Default Adam first moment decay rate (β₁).
pub(crate) const ADAM_BETA1: f64 = 0.9;

/// Default Adam second moment decay rate (β₂).
pub(crate) const ADAM_BETA2: f64 = 0.999;

/// Default Adam numerical stability constant (ε).
pub(crate) const ADAM_EPSILON: f64 = 1e-8;

// ─── Pegasos (SVM) constants ────────────────────────────────────────────────

/// Learning rate decay constant for Pegasos SGD: lr = 1/(C * (1 + DECAY * epoch)).
pub(crate) const PEGASOS_LR_DECAY: f64 = 0.01;

// ─── SMO (Kernel SVM) constants ─────────────────────────────────────────────

/// Default convergence tolerance for SMO (KKT violation threshold).
/// Matches sklearn's `SVC(tol=1e-3)` default.
#[cfg(feature = "experimental")]
pub(crate) const SMO_TOL: f64 = 1e-3;

/// Threshold for treating SMO box constraint bounds L and H as equal.
/// When |L − H| < this value the pair is skipped.
#[cfg(feature = "experimental")]
pub(crate) const SMO_BOUNDS_EQ: f64 = 1e-8;

/// Minimum change in alpha_j to accept an SMO step.
/// Steps smaller than this are treated as no-ops.
#[cfg(feature = "experimental")]
pub(crate) const SMO_ALPHA_CHANGE_THRESH: f64 = 1e-5;

/// Alpha threshold for identifying support vectors.
/// Points with alpha > this value are retained after SMO.
#[cfg(feature = "experimental")]
pub(crate) const SV_ALPHA_THRESH: f64 = 1e-8;

// ─── Parallelism thresholds ───────────────────────────────────────────────

/// Minimum n×m product to enable rayon parallelism in logistic regression
/// feature gradient computation. Below this, rayon spawn overhead (~2-5 µs)
/// exceeds the parallel speedup.
pub(crate) const LOGREG_PAR_THRESHOLD: usize = 5_000;

/// Minimum query×train×features product to parallelize KNN brute-force predict.
pub(crate) const KNN_PAR_THRESHOLD: usize = 10_000;

/// Minimum n×k product to parallelize K-Means assignment step.
pub(crate) const KMEANS_PAR_THRESHOLD: usize = 5_000;

/// Minimum n² kernel matrix size to parallelize SVM kernel computation.
#[cfg(feature = "experimental")]
pub(crate) const SVM_KERNEL_PAR_THRESHOLD: usize = 10_000;

