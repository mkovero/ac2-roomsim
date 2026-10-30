//! Echograms (energy per time bin per band), their Schroeder decay, and synthesis of an
//! impulse response from them.

use crate::band::{BAND_COUNT, Band};
use crate::image::Arrival;
use crate::rng::Rng;

/// Energy per time bin per octave band, on the scale of [`Arrival::energy`] (the direct
/// sound at 1 m in free field is 1).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Echogram {
    /// Bin width, s. Bin `k` covers `[k·bin_s, (k+1)·bin_s)` after the source emits.
    pub bin_s: f64,
    /// Per band (indexed by [`Band::index`]), the energy arriving in each bin.
    pub bands: [Vec<f64>; BAND_COUNT],
}

impl Echogram {
    pub(crate) fn zeros(bin_s: f64, bins: usize) -> Echogram {
        Echogram {
            bin_s,
            bands: std::array::from_fn(|_| vec![0.0; bins]),
        }
    }

    /// Number of bins.
    pub fn len(&self) -> usize {
        self.bands[0].len()
    }

    /// Whether there are no bins.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Length, s.
    pub fn duration_s(&self) -> f64 {
        self.len() as f64 * self.bin_s
    }

    /// One band's bins.
    pub fn band(&self, b: Band) -> &[f64] {
        &self.bands[b.index()]
    }

    /// Add specular arrivals, each into the bin of its arrival time.
    pub fn add_arrivals(&mut self, arrivals: &[Arrival]) {
        for a in arrivals {
            let k = (a.time_s / self.bin_s) as usize;
            if k >= self.len() {
                continue;
            }
            for (band, e) in self.bands.iter_mut().zip(a.energy) {
                band[k] += e;
            }
        }
    }

    /// Schroeder's backward-integrated decay curve, dB re the total energy (0 dB at t = 0).
    /// Backward integration of the energy response equals the ensemble average of
    /// interrupted-noise decays (Schroeder 1965), so a decay rate fitted to it is the
    /// reverberation time ISO 3382-1 means. Bins after the last energy read −∞.
    pub fn decay_db(&self, b: Band) -> Vec<f64> {
        let e = self.band(b);
        let mut tail = vec![0.0; e.len()];
        let mut acc = 0.0;
        for k in (0..e.len()).rev() {
            acc += e[k];
            tail[k] = acc;
        }
        let total = acc;
        tail.iter()
            .map(|&v| {
                if total > 0.0 {
                    10.0 * (v / total).log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Reverberation time (T60 extrapolated), s, from a least-squares line through the decay
    /// curve between `from_db` and `to_db` below the start (−5 and −25 for T20, −5 and −35
    /// for T30). `None` if the curve does not reach `to_db` or the fitted slope is not a
    /// decay.
    pub fn decay_time(&self, b: Band, from_db: f64, to_db: f64) -> Option<f64> {
        let curve = self.decay_db(b);
        let (mut n, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut reached = false;
        for (k, &y) in curve.iter().enumerate() {
            if y > from_db {
                continue;
            }
            if y < to_db {
                reached = true;
                break;
            }
            // The backward integral at bin k holds the energy from the bin's start onward.
            let x = k as f64 * self.bin_s;
            n += 1.0;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        if !reached || n < 3.0 {
            return None;
        }
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        (slope < 0.0).then(|| -60.0 / slope)
    }

    /// T20 (fit from −5 to −25 dB), s.
    pub fn t20(&self, b: Band) -> Option<f64> {
        self.decay_time(b, -5.0, -25.0)
    }

    /// T30 (fit from −5 to −35 dB), s.
    pub fn t30(&self, b: Band) -> Option<f64> {
        self.decay_time(b, -5.0, -35.0)
    }
}

/// Why an impulse response cannot be synthesised.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IrError {
    /// The sample rate is below twice the top edge of the 8 kHz band (about 22.6 kHz), so
    /// that band cannot be represented.
    SampleRateTooLow,
}

impl std::fmt::Display for IrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IrError::SampleRateTooLow => write!(f, "sample rate below the 8 kHz band's top edge"),
        }
    }
}

impl std::error::Error for IrError {}

/// Synthesise a pressure impulse response from specular arrivals and a late echogram.
///
/// A dense reflection sequence has random phase, so its pressure is band-limited noise with
/// the echogram as envelope. Per band, white Gaussian noise is band-filtered, shaped by the
/// late echogram (variance per sample = bin energy / samples in the bin), and then rescaled
/// window by window so each window holds exactly the echogram's energy. Without that last
/// step the energy of a band-noise realisation scatters by about 1/√(B·T) (bandwidth times
/// duration), tens of percent in the 125 Hz band of a short tail. Each window spans at
/// least four periods of the band centre, so the gain steps stay slow against the band's
/// own oscillation and add little energy outside it.
///
/// Each specular arrival adds a band-filtered impulse of energy E at its exact sample (the
/// filter normalised so Σh² = 1), and the bands are summed. Every band of the result carries
/// the simulated energy on the echogram's scale: Σ p² of the late part equals the echogram
/// total up to the small correlation between independent noises in overlapping bands.
///
/// The band filters are causal (two biquads per band), so energy is delayed by their group
/// delay, a few milliseconds at 125 Hz; decay rates are unaffected.
pub fn synthesize_ir(
    arrivals: &[Arrival],
    late: &Echogram,
    sample_rate: f64,
    seed: u64,
) -> Result<Vec<f64>, IrError> {
    if !(sample_rate.is_finite() && sample_rate > 2.0 * Band::Hz8k.upper_hz()) {
        return Err(IrError::SampleRateTooLow);
    }
    let n = (late.duration_s() * sample_rate).ceil() as usize;
    let mut out = vec![0.0; n];
    let mut env = vec![0.0; n];
    let mut x = vec![0.0; n];
    for b in Band::ALL {
        let e = late.band(b);
        env.iter_mut().for_each(|v| *v = 0.0);
        for (k, &energy) in e.iter().enumerate() {
            let s0 = ((k as f64 * late.bin_s) * sample_rate).round() as usize;
            let s1 = ((((k + 1) as f64) * late.bin_s) * sample_rate).round() as usize;
            let s1 = s1.min(n);
            if s1 <= s0 || energy <= 0.0 {
                continue;
            }
            let var = energy / (s1 - s0) as f64;
            env[s0..s1].iter_mut().for_each(|v| *v = var);
        }

        let mut filter = OctaveFilter::new(b.exact_hz(), sample_rate);
        let mut rng = Rng::new(seed, 0x1f_0000 + b.index() as u64, 0);
        for (xi, &v) in x.iter_mut().zip(&env) {
            *xi = filter.process(rng.normal()) * v.sqrt();
        }
        let window = ((4.0 * sample_rate / b.exact_hz()).ceil() as usize).max(1);
        for (xs, es) in x.chunks_mut(window).zip(env.chunks(window)) {
            let want: f64 = es.iter().sum();
            let got: f64 = xs.iter().map(|v| v * v).sum();
            if got > 0.0 {
                let g = (want / got).sqrt();
                xs.iter_mut().for_each(|v| *v *= g);
            }
        }

        // Impulses: filter an excitation of √E per arrival.
        x.iter_mut()
            .zip(out.iter_mut())
            .for_each(|(xi, o)| *o += *xi);
        x.iter_mut().for_each(|v| *v = 0.0);
        let mut any = false;
        for a in arrivals {
            let s = (a.time_s * sample_rate).round() as usize;
            if s < n {
                x[s] += a.energy[b.index()].sqrt();
                any = true;
            }
        }
        if any {
            let gain = 1.0 / filter.energy_gain().sqrt();
            filter.reset();
            for (o, &xi) in out.iter_mut().zip(&x) {
                *o += gain * filter.process(xi);
            }
        }
    }
    Ok(out)
}

/// An octave band-pass: two identical second-order band-passes in cascade. A second-order
/// band-pass has |H|² = 1/(1 + Q²(f/f₀ − f₀/f)²); for the cascade (|H|⁴) to be 3 dB down at
/// the octave edges f₀·2^(±1/2), where f/f₀ − f₀/f = 2^(−1/2), each stage needs
/// Q·2^(−1/2) = √(√2 − 1), so Q ≈ 0.910.
struct OctaveFilter {
    b: [f64; 3],
    a: [f64; 2],
    s: [[f64; 2]; 2],
}

impl OctaveFilter {
    fn new(f0: f64, fs: f64) -> OctaveFilter {
        let q = (std::f64::consts::SQRT_2 - 1.0).sqrt() * std::f64::consts::SQRT_2;
        // Bilinear transform with the centre pre-warped (RBJ cookbook band-pass, 0 dB peak).
        let w0 = std::f64::consts::TAU * f0 / fs;
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        OctaveFilter {
            b: [alpha / a0, 0.0, -alpha / a0],
            a: [-2.0 * w0.cos() / a0, (1.0 - alpha) / a0],
            s: [[0.0; 2]; 2],
        }
    }

    fn reset(&mut self) {
        self.s = [[0.0; 2]; 2];
    }

    /// Transposed direct form II, two stages.
    fn process(&mut self, mut x: f64) -> f64 {
        for st in &mut self.s {
            let y = self.b[0] * x + st[0];
            st[0] = self.b[1] * x - self.a[0] * y + st[1];
            st[1] = self.b[2] * x - self.a[1] * y;
            x = y;
        }
        x
    }

    /// Σh² of the impulse response: the factor by which the filter scales white noise's
    /// power and an impulse's energy alike.
    fn energy_gain(&mut self) -> f64 {
        self.reset();
        let mut sum = 0.0;
        let mut x = 1.0;
        let mut quiet = 0;
        // The ringing passes through zero every half period, so the response has ended only
        // once a run of samples longer than any half period has stayed negligible. A
        // million samples bounds the loop for any rate.
        for _ in 0..1_000_000 {
            let y = self.process(x);
            x = 0.0;
            sum += y * y;
            quiet = if y.abs() < 1e-15 { quiet + 1 } else { 0 };
            if quiet > 4096 {
                break;
            }
        }
        sum
    }
}
