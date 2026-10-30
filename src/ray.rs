//! Stochastic ray tracing for the late tail (Krokstad, Strøm & Sørsdal 1968), with
//! scattering after Lambert's cosine law and detection in receiver spheres.
//!
//! Each ray carries an equal share of the source's energy and loses the absorbed share at
//! every reflection (the expected value, instead of killing rays at random, which would add
//! variance). At a reflection the ray is scattered with probability s, otherwise reflected
//! specularly; a scattered ray leaves in a direction drawn from the cosine law, the angular
//! distribution of an ideally diffusing surface.
//!
//! A receiver is a sphere of radius r. A ray crossing it along a chord of length ℓ adds
//! E·4π·ℓ/V_sphere to the echogram bin of its crossing time: averaged over rays this is the
//! sphere's mean energy density times 4π/c, which makes the direct sound at distance d come
//! out as 1/d², the same scale the image sources use.
//!
//! Image sources already give every purely specular path up to their order exactly, so a ray
//! still on such a path is not detected; it counts once it has been scattered or has more
//! specular reflections than the image-source order. Together the two methods count each
//! path once.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::band::PerBand;
use crate::rng::Rng;
use crate::room::Prism;
use crate::vec3::{V3, add_scaled, dot, sub};

/// What every ray of a run shares.
pub(crate) struct RayCtx<'a> {
    pub prism: &'a Prism,
    pub source: V3,
    pub receivers: &'a [V3],
    pub radius: f64,
    pub image_order: u8,
    pub absorption_scale: PerBand,
    pub air: PerBand,
    pub c: f64,
    pub bin_s: f64,
    pub bins: usize,
    pub seed: u64,
}

/// One work item's result: per receiver, unnormalised energy per bin (one band).
pub(crate) struct ChunkOut {
    pub hist: Vec<Vec<f64>>,
    pub lost: u64,
}

/// A ray is dropped once its energy has fallen this far (−90 dB) below its start: the
/// echogram's dynamic range ends well above that.
const ENERGY_FLOOR: f64 = 1e-9;

/// Trace rays `first..first + count` of `band`.
pub(crate) fn trace_chunk(
    ctx: &RayCtx<'_>,
    band: usize,
    first: u64,
    count: u64,
    cancel: Option<&AtomicBool>,
) -> Option<ChunkOut> {
    let mut hist = vec![vec![0.0; ctx.bins]; ctx.receivers.len()];
    let mut lost = 0;
    let max_len = ctx.bins as f64 * ctx.bin_s * ctx.c;
    let r2 = ctx.radius * ctx.radius;
    let weight =
        4.0 * std::f64::consts::PI / (4.0 / 3.0 * std::f64::consts::PI * ctx.radius.powi(3));
    let m = ctx.air[band];
    let scale = ctx.absorption_scale[band];
    for ray in first..first + count {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return None;
        }
        let mut rng = Rng::new(ctx.seed, band as u64, ray);
        let mut dir = uniform_sphere(&mut rng);
        let mut o = ctx.source;
        let mut skip = None;
        let mut length = 0.0;
        let mut energy = 1.0;
        let mut scattered = false;
        let mut specular_order: u32 = 0;
        loop {
            let Some((t, s)) = ctx.prism.first_hit(o, dir, skip) else {
                lost += 1;
                break;
            };
            if scattered || specular_order > ctx.image_order as u32 {
                for (rc, h) in ctx.receivers.iter().zip(hist.iter_mut()) {
                    let v = sub(*rc, o);
                    let tc = dot(v, dir);
                    let d2 = dot(v, v) - tc * tc;
                    if d2 >= r2 {
                        continue;
                    }
                    let half = (r2 - d2).sqrt();
                    let t0 = (tc - half).max(0.0);
                    let t1 = (tc + half).min(t);
                    if t1 <= t0 {
                        continue;
                    }
                    let at = length + 0.5 * (t0 + t1);
                    let bin = (at / ctx.c / ctx.bin_s) as usize;
                    if bin < ctx.bins {
                        h[bin] += energy * (-m * at).exp() * weight * (t1 - t0);
                    }
                }
            }
            length += t;
            if length >= max_len {
                break;
            }
            o = add_scaled(o, t, dir);
            let co = ctx.prism.coefficients_at(s);
            energy *= 1.0 - (co.absorption[band] * scale).min(1.0);
            if energy * (-m * length).exp() < ENERGY_FLOOR {
                break;
            }
            let n = ctx.prism.plane(s).n;
            if rng.uniform() < co.scattering[band] {
                dir = lambert(&mut rng, n);
                scattered = true;
            } else {
                dir = add_scaled(dir, -2.0 * dot(dir, n), n);
                specular_order += 1;
            }
            skip = Some(s);
        }
    }
    Some(ChunkOut { hist, lost })
}

/// A direction uniform on the unit sphere: an omnidirectional source.
fn uniform_sphere(rng: &mut Rng) -> V3 {
    let z = 1.0 - 2.0 * rng.uniform();
    let phi = std::f64::consts::TAU * rng.uniform();
    let r = (1.0 - z * z).max(0.0).sqrt();
    [r * phi.cos(), r * phi.sin(), z]
}

/// A direction about the unit normal `n` with density ∝ cos θ (Lambert): cos θ = √u.
fn lambert(rng: &mut Rng, n: V3) -> V3 {
    let cos_t = rng.uniform().sqrt();
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let phi = std::f64::consts::TAU * rng.uniform();
    // An orthonormal basis (u, v, n): u from the axis least aligned with n.
    let a = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalise(cross(a, n));
    let v = cross(n, u);
    let (x, y) = (sin_t * phi.cos(), sin_t * phi.sin());
    [
        x * u[0] + y * v[0] + cos_t * n[0],
        x * u[1] + y * v[1] + cos_t * n[1],
        x * u[2] + y * v[2] + cos_t * n[2],
    ]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalise(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}
