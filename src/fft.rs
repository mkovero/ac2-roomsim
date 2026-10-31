//! A radix-2 complex FFT, enough for impulse-response synthesis: band masks are applied in
//! the frequency domain because no low-order recursive filter is steep enough (see
//! [`crate::echogram::synthesize_band_irs`]).

use std::f64::consts::TAU;

/// A complex number as `[re, im]`.
pub(crate) type Complex = [f64; 2];

/// Plan for one power-of-two length.
pub(crate) struct Fft {
    pub n: usize,
    /// e^(−2πik/n) for k < n/2, each computed directly so rounding does not accumulate.
    twiddles: Vec<Complex>,
}

impl Fft {
    /// `n` must be a power of two.
    pub fn new(n: usize) -> Fft {
        assert!(n.is_power_of_two());
        let twiddles = (0..n / 2)
            .map(|k| {
                let w = -TAU * k as f64 / n as f64;
                [w.cos(), w.sin()]
            })
            .collect();
        Fft { n, twiddles }
    }

    /// In-place forward transform, X_k = Σ x_j e^(−2πijk/n).
    pub fn forward(&self, x: &mut [Complex]) {
        self.transform(x, false);
    }

    /// In-place inverse transform, scaled by 1/n so `inverse(forward(x)) = x`.
    pub fn inverse(&self, x: &mut [Complex]) {
        self.transform(x, true);
        let s = 1.0 / self.n as f64;
        x.iter_mut().for_each(|v| {
            v[0] *= s;
            v[1] *= s;
        });
    }

    fn transform(&self, x: &mut [Complex], inverse: bool) {
        let n = self.n;
        assert_eq!(x.len(), n);
        if n < 2 {
            return;
        }
        let bits = n.trailing_zeros();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - bits);
            if j > i {
                x.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let [wr, wi] = self.twiddles[k * step];
                    let wi = if inverse { -wi } else { wi };
                    let [br, bi] = x[start + k + half];
                    let t = [br * wr - bi * wi, br * wi + bi * wr];
                    let u = x[start + k];
                    x[start + k] = [u[0] + t[0], u[1] + t[1]];
                    x[start + k + half] = [u[0] - t[0], u[1] - t[1]];
                }
            }
            len *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_direct_dft_and_inverts() {
        let n = 16;
        let x: Vec<Complex> = (0..n)
            .map(|j| [(j as f64 * 0.7).sin(), (j as f64 * 1.3).cos() - 0.2])
            .collect();
        let mut y = x.clone();
        let fft = Fft::new(n);
        fft.forward(&mut y);
        for (k, yk) in y.iter().enumerate() {
            let mut want = [0.0, 0.0];
            for (j, xj) in x.iter().enumerate() {
                let w = -TAU * (j * k) as f64 / n as f64;
                want[0] += xj[0] * w.cos() - xj[1] * w.sin();
                want[1] += xj[0] * w.sin() + xj[1] * w.cos();
            }
            assert!((yk[0] - want[0]).abs() < 1e-12 && (yk[1] - want[1]).abs() < 1e-12);
        }
        fft.inverse(&mut y);
        for (a, b) in x.iter().zip(&y) {
            assert!((a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12);
        }
    }
}
