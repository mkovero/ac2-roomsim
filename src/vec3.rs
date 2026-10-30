//! Three-vectors as plain arrays: the geometry needs a handful of operations, not a library.

pub(crate) type V3 = [f64; 3];

pub(crate) fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn add_scaled(a: V3, s: f64, d: V3) -> V3 {
    [a[0] + s * d[0], a[1] + s * d[1], a[2] + s * d[2]]
}

pub(crate) fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
