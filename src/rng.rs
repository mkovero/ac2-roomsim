//! Seeded random numbers. Every ray gets its own generator, seeded from the run's seed, its
//! band and its index, so a ray's path does not depend on which thread traced it or in what
//! order: the result is the same for any thread count.

use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64Mcg;

/// SplitMix64 finaliser: spreads nearby inputs (ray 7, ray 8) to unrelated seeds.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub(crate) struct Rng(Pcg64Mcg);

impl Rng {
    pub fn new(seed: u64, stream: u64, index: u64) -> Rng {
        let hi = mix(seed ^ mix(stream));
        let lo = mix(hi ^ mix(index.wrapping_add(0x5851_f42d_4c95_7f2d)));
        Rng(Pcg64Mcg::seed_from_u64(lo ^ hi.rotate_left(17)))
    }

    /// Uniform in [0, 1) with 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        (self.0.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
