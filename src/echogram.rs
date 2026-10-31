//! Echograms (energy per time bin per band), their Schroeder decay, and synthesis of an
//! impulse response from them.

use rayon::prelude::*;

use crate::band::{BAND_COUNT, Band};
use crate::fft::{Complex, Fft};
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

/// Width of the crossover between neighbouring band masks, octaves. Narrow, so a consumer's
/// octave filter at the nominal edges sees little of the neighbouring band; wide enough that
/// a band-limited impulse rings for only a few hundred milliseconds at the 125 Hz band's
/// lower edge (the ringing lasts about 1/Δf, Δf ≈ 89 Hz · ln2 · width).
const CROSSOVER_OCT: f64 = 1.0 / 12.0;

/// Synthesise a pressure impulse response from specular arrivals and a late echogram: the
/// sum of [`synthesize_band_irs`] with the same seed.
pub fn synthesize_ir(
    arrivals: &[Arrival],
    late: &Echogram,
    sample_rate: f64,
    seed: u64,
) -> Result<Vec<f64>, IrError> {
    let bands = synthesize_band_irs(arrivals, late, sample_rate, seed)?;
    let mut out = vec![0.0; bands[0].len()];
    for band in &bands {
        out.iter_mut().zip(band).for_each(|(o, v)| *o += v);
    }
    Ok(out)
}

/// Synthesise one pressure impulse response per octave band (indexed by [`Band::index`]),
/// each carrying only that band's energy, all of the length of the echogram.
///
/// A dense reflection sequence has random phase, so its pressure is band-limited noise with
/// the echogram as envelope. Per band, white Gaussian noise is band-limited, shaped by the
/// late echogram (variance per sample = bin energy / samples in the bin), and then rescaled
/// window by window so each window holds the echogram's energy. Without that step the energy
/// of a band-noise realisation scatters by about 1/√(B·T) (bandwidth times duration), tens
/// of percent in the 125 Hz band of a short tail. Each window spans four periods of the band
/// centre, so the gain steps stay slow against the band's own oscillation. The shaped noise
/// is band-limited once more, to remove what the envelope and gain steps spread outside the
/// band, and scaled so its total equals the late echogram's. Each specular arrival adds a
/// band-limited impulse of energy E at its exact sample.
///
/// Band limiting is a mask on the spectrum, flat across the octave and falling to zero
/// within a twelfth of an octave around each edge, power-complementary with the neighbouring
/// band's so the bands sum to a flat response. A recursive band-pass would not do: a
/// second-order pair is only about 36 dB down three octaves away, and the decay curves of
/// different bands drift apart by tens of dB over a tail, so a slow 125 Hz decay leaking
/// through that skirt dominates a fast 1 kHz decay well before −25 dB and a T20 read through
/// an external octave filter comes out several times too long. With the mask, a band holds
/// nothing beyond its crossovers, and an octave filter at the nominal edges sees the
/// neighbouring band only within half a crossover.
///
/// The mask has zero phase, so a band-limited impulse rings symmetrically about its arrival:
/// the response starts a few milliseconds before the direct sound (at the 125 Hz band
/// longest), and what would precede t = 0 is dropped. Energy arriving within a ringing time
/// of the end is likewise cut, so the energies match the echogram's to well under 1 %.
pub fn synthesize_band_irs(
    arrivals: &[Arrival],
    late: &Echogram,
    sample_rate: f64,
    seed: u64,
) -> Result<[Vec<f64>; BAND_COUNT], IrError> {
    if !(sample_rate.is_finite() && sample_rate > 2.0 * Band::Hz8k.upper_hz()) {
        return Err(IrError::SampleRateTooLow);
    }
    let n = (late.duration_s() * sample_rate).ceil() as usize;
    if n == 0 {
        return Ok(std::array::from_fn(|_| Vec::new()));
    }
    // The masks act circularly; a guard of several ringing times after the end keeps what
    // rings past the end, or before t = 0, from wrapping onto the kept samples.
    let df = Band::Hz125.lower_hz() * std::f64::consts::LN_2 * CROSSOVER_OCT;
    let guard = (6.0 / df * sample_rate).ceil() as usize;
    let len = (n + guard).next_power_of_two();
    let fft = Fft::new(len);
    let bands: Vec<Vec<f64>> = Band::ALL
        .par_iter()
        .map(|&b| band_ir(b, arrivals, late, sample_rate, seed, n, &fft))
        .collect();
    let mut bands = bands.into_iter();
    Ok(std::array::from_fn(|_| {
        bands.next().expect("one IR per band")
    }))
}

fn band_ir(
    b: Band,
    arrivals: &[Arrival],
    late: &Echogram,
    sample_rate: f64,
    seed: u64,
    n: usize,
    fft: &Fft,
) -> Vec<f64> {
    let len = fft.n;
    let mask: Vec<f64> = (0..len)
        .map(|k| band_mask(b, k.min(len - k) as f64 * sample_rate / len as f64))
        .collect();
    let apply = |x: &mut [Complex]| {
        fft.forward(x);
        x.iter_mut().zip(&mask).for_each(|(v, &m)| {
            v[0] *= m;
            v[1] *= m;
        });
        fft.inverse(x);
    };

    let mut env = vec![0.0; n];
    for (k, &energy) in late.band(b).iter().enumerate() {
        let s0 = ((k as f64 * late.bin_s) * sample_rate).round() as usize;
        let s1 = ((((k + 1) as f64) * late.bin_s) * sample_rate).round() as usize;
        let s1 = s1.min(n);
        if s1 <= s0 || energy <= 0.0 {
            continue;
        }
        let var = energy / (s1 - s0) as f64;
        env[s0..s1].iter_mut().for_each(|v| *v = var);
    }
    let want_late: f64 = env.iter().sum();

    // Stationary band noise: the real part of masked complex white noise is real band noise
    // (the mask is even in frequency).
    let mut rng = Rng::new(seed, 0x1f_0000 + b.index() as u64, 0);
    let mut buf: Vec<Complex> = (0..len).map(|_| [rng.normal(), rng.normal()]).collect();
    apply(&mut buf);
    let mut x: Vec<f64> = buf[..n]
        .iter()
        .zip(&env)
        .map(|(v, &e)| v[0] * e.sqrt())
        .collect();
    let window = ((4.0 * sample_rate / b.exact_hz()).ceil() as usize).max(1);
    for (xs, es) in x.chunks_mut(window).zip(env.chunks(window)) {
        let want: f64 = es.iter().sum();
        let got: f64 = xs.iter().map(|v| v * v).sum();
        if got > 0.0 {
            let g = (want / got).sqrt();
            xs.iter_mut().for_each(|v| *v *= g);
        }
    }

    // Second pass: the shaped noise in the real part, the arrivals' impulses (√E each) in
    // the imaginary part. Mask and signals are real, so the two stay apart.
    buf.iter_mut().for_each(|v| *v = [0.0, 0.0]);
    buf[..n].iter_mut().zip(&x).for_each(|(v, &xi)| v[0] = xi);
    for a in arrivals {
        let s = (a.time_s * sample_rate).round() as usize;
        if s < n {
            buf[s][1] += a.energy[b.index()].sqrt();
        }
    }
    apply(&mut buf);
    let got_late: f64 = buf[..n].iter().map(|v| v[0] * v[0]).sum();
    let g_late = if got_late > 0.0 {
        (want_late / got_late).sqrt()
    } else {
        0.0
    };
    // A masked unit impulse has energy Σ|M_k|²/len (Parseval).
    let mask_energy: f64 = mask.iter().map(|m| m * m).sum::<f64>() / len as f64;
    let g_imp = if mask_energy > 0.0 {
        1.0 / mask_energy.sqrt()
    } else {
        0.0
    };
    buf[..n]
        .iter()
        .map(|v| g_late * v[0] + g_imp * v[1])
        .collect()
}

/// The band's amplitude mask at `f` Hz: 1 inside the octave, 0 outside, crossing over in
/// log frequency around each edge as cos/sin of a raised-cosine ramp, so the squared masks of
/// two neighbouring bands sum to 1 and the ramp has no kink to make the impulse ring long.
fn band_mask(b: Band, f: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    // Base-10 bands are spaced 10^0.3, not 2, so one band's upper edge and the next one's
    // lower edge differ by 0.2 %; neighbours cross over at the geometric mean of their
    // centres so the masks stay complementary.
    let i = b.index();
    let lower = match i {
        0 => b.lower_hz(),
        _ => (Band::ALL[i - 1].exact_hz() * b.exact_hz()).sqrt(),
    };
    let upper = match Band::ALL.get(i + 1) {
        None => b.upper_hz(),
        Some(next) => (b.exact_hz() * next.exact_hz()).sqrt(),
    };
    let x = f.log2();
    rise(x - lower.log2()) * rise(upper.log2() - x)
}

/// 0 below −CROSSOVER_OCT/2 octaves from an edge, 1 above +CROSSOVER_OCT/2.
fn rise(d: f64) -> f64 {
    let h = CROSSOVER_OCT / 2.0;
    if d <= -h {
        0.0
    } else if d >= h {
        1.0
    } else {
        let s = 0.5 * (1.0 + (std::f64::consts::PI * d / CROSSOVER_OCT).sin());
        (std::f64::consts::FRAC_PI_2 * s).sin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbouring_masks_are_power_complementary() {
        for i in 0..2000 {
            let f = 60.0 * 2f64.powf(i as f64 * 8.0 / 2000.0);
            let sum: f64 = Band::ALL.iter().map(|&b| band_mask(b, f).powi(2)).sum();
            let inside = f > Band::Hz125.lower_hz() * 2f64.powf(CROSSOVER_OCT / 2.0)
                && f < Band::Hz8k.upper_hz() / 2f64.powf(CROSSOVER_OCT / 2.0);
            if inside {
                assert!((sum - 1.0).abs() < 1e-12, "{f} Hz: Σ|M|² = {sum}");
            }
            assert!(sum <= 1.0 + 1e-12);
        }
    }
}
