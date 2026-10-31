//! Impulse-response synthesis keeps the echogram's energy and is deterministic.

use ac2_roomsim::*;

use crate::common::{run, shoebox, uniform};

fn sim() -> Simulation {
    let prism = shoebox(10.0, 8.0, 4.0, uniform(0.2, 0.5));
    let cfg = SimConfig {
        rays_per_band: 4_000,
        seed: 3,
        ..SimConfig::default()
    };
    run(&prism, [2.5, 3.0, 1.6], &[[7.0, 5.0, 1.2]], &cfg)
}

#[test]
fn late_noise_carries_the_echogram_energy() {
    let s = sim();
    let late = &s.receivers[0].late;
    for b in Band::ALL {
        let mut one = late.clone();
        for (i, band) in one.bands.iter_mut().enumerate() {
            if i != b.index() {
                band.iter_mut().for_each(|v| *v = 0.0);
            }
        }
        let ir = synthesize_ir(&[], &one, 48_000.0, 1).unwrap();
        let want: f64 = one.band(b).iter().sum();
        let got: f64 = ir.iter().map(|p| p * p).sum();
        assert!(
            (got / want - 1.0).abs() < 1e-9,
            "{b:?}: Σp² {got} vs echogram {want}"
        );
    }
    let ir = synthesize_ir(&[], late, 48_000.0, 1).unwrap();
    let want: f64 = late.bands.iter().flatten().sum();
    let got: f64 = ir.iter().map(|p| p * p).sum();
    // Each band exact; what is left is chance correlation between neighbouring bands' noise
    // where their masks cross over.
    assert!(
        (got / want - 1.0).abs() < 0.02,
        "Σp² {got} vs echogram {want}"
    );
}

#[test]
fn ir_is_deterministic_and_starts_with_the_direct_sound() {
    let s = sim();
    let r = &s.receivers[0];
    let a = r.impulse_response(48_000.0, 9).unwrap();
    assert_eq!(a, r.impulse_response(48_000.0, 9).unwrap());
    assert_ne!(a, r.impulse_response(48_000.0, 10).unwrap());
    // The band masks have zero phase, so impulses ring on both sides of their arrival, an
    // octave band's ringing lasting about 1/bandwidth (11 ms at 125 Hz); the peak is the
    // direct sound and under 1 % of the energy comes before it.
    let direct = (r.arrivals[0].time_s * 48_000.0).round() as usize;
    let peak = (0..a.len())
        .max_by(|&i, &j| a[i].abs().total_cmp(&a[j].abs()))
        .unwrap();
    assert_eq!(peak, direct);
    let before: f64 = a[..direct - 48].iter().map(|p| p * p).sum();
    let total: f64 = a.iter().map(|p| p * p).sum();
    assert!(
        before < 0.01 * total,
        "{before} of {total} before the direct sound"
    );
    assert_eq!(
        r.impulse_response(16_000.0, 9).unwrap_err(),
        IrError::SampleRateTooLow
    );
}

#[test]
fn whole_echogram_adds_arrivals_to_the_late_part() {
    let s = sim();
    let r = &s.receivers[0];
    let total: f64 = r.echogram().band(Band::Hz1k).iter().sum();
    let late: f64 = r.late.band(Band::Hz1k).iter().sum();
    let spec: f64 = r
        .arrivals
        .iter()
        .map(|a| a.energy[Band::Hz1k.index()])
        .sum();
    assert!((total - late - spec).abs() < 1e-12 * total);
}

const FS: f64 = 48_000.0;

/// Strongly frequency-dependent absorption, fully diffuse: T falls from about 5 s at 125 Hz
/// to under 0.5 s at 8 kHz, so the decay curves of the bands part by tens of dB within the
/// first second. A band filter whose skirt passes the 125 Hz tail into the 1 kHz band
/// lengthens the 1 kHz T20 several times over.
fn sloped() -> Simulation {
    let material = Material::Custom(Coefficients {
        absorption: [0.03, 0.06, 0.12, 0.2, 0.3, 0.45, 0.55],
        scattering: [1.0; BAND_COUNT],
    });
    let prism = shoebox(10.0, 8.0, 4.0, material);
    let cfg = SimConfig {
        rays_per_band: 3_000,
        seed: 5,
        ..SimConfig::default()
    };
    run(&prism, [2.5, 3.0, 1.6], &[[7.0, 5.0, 1.2]], &cfg)
}

/// T20 of a pressure signal: its squared samples binned like `like` into band `b` of an
/// otherwise empty echogram, so the fit is the library's own.
fn t20_of(p: &[f64], like: &Echogram, b: Band) -> f64 {
    let per_bin = like.bin_s * FS;
    let mut eg = like.clone();
    eg.bands
        .iter_mut()
        .for_each(|v| v.iter_mut().for_each(|e| *e = 0.0));
    for (i, v) in p.iter().enumerate() {
        let k = ((i as f64 + 0.5) / per_bin) as usize;
        if k < eg.len() {
            eg.bands[b.index()][k] += v * v;
        }
    }
    eg.t20(b).expect("a T20 fit")
}

#[test]
fn band_irs_carry_each_bands_energy_and_sum_to_the_ir() {
    let s = sloped();
    let r = &s.receivers[0];
    let late = synthesize_band_irs(&[], &r.late, FS, 4).unwrap();
    for b in Band::ALL {
        let want: f64 = r.late.band(b).iter().sum();
        let got: f64 = late[b.index()].iter().map(|p| p * p).sum();
        assert!(
            (got / want - 1.0).abs() < 1e-9,
            "{b:?}: Σp² {got} vs {want}"
        );
    }
    let bands = r.band_impulse_responses(FS, 4).unwrap();
    let eg = r.echogram();
    for b in Band::ALL {
        let want: f64 = eg.band(b).iter().sum();
        let got: f64 = bands[b.index()].iter().map(|p| p * p).sum();
        eprintln!("{b:?}: Σp² / echogram − 1 = {:.2e}", got / want - 1.0);
        // The late noise is exact. What is left: chance correlation between the noise and
        // the impulses it overlaps, and what an impulse's ringing puts before t = 0 (half a
        // percent of the 125 Hz band here, its direct sound 14 ms in).
        assert!(
            (got / want - 1.0).abs() < 0.02,
            "{b:?}: Σp² {got} vs {want}"
        );
    }
    let ir = r.impulse_response(FS, 4).unwrap();
    let mut sum = vec![0.0; ir.len()];
    for band in &bands {
        sum.iter_mut().zip(band).for_each(|(o, v)| *o += v);
    }
    assert_eq!(sum, ir);
    assert_eq!(bands, r.band_impulse_responses(FS, 4).unwrap());
    assert_ne!(bands, r.band_impulse_responses(FS, 5).unwrap());
}

#[test]
fn band_ir_t20_matches_the_echogram() {
    let s = sloped();
    let r = &s.receivers[0];
    let eg = r.echogram();
    let bands = r.band_impulse_responses(FS, 4).unwrap();
    for b in Band::ALL {
        let want = eg.t20(b).unwrap();
        let got = t20_of(&bands[b.index()], &eg, b);
        eprintln!("{b:?}: echogram T20 {want:.3} s, band IR {got:.3} s");
        assert!((got / want - 1.0).abs() < 0.03, "{b:?}: {got} vs {want}");
    }
}

#[test]
fn broadband_ir_through_an_octave_filter_keeps_each_bands_t20() {
    let s = sloped();
    let r = &s.receivers[0];
    let eg = r.echogram();
    let ir = r.impulse_response(FS, 4).unwrap();
    for b in Band::ALL {
        let want = eg.t20(b).unwrap();
        let got = t20_of(&octave_filter(&ir, b), &eg, b);
        eprintln!("{b:?}: echogram T20 {want:.3} s, filtered IR {got:.3} s");
        assert!((got / want - 1.0).abs() < 0.05, "{b:?}: {got} vs {want}");
    }
}

/// A steep octave filter: the band between its nominal edges kept, the rest of the spectrum
/// zeroed (zero-padded to twice the length so nothing wraps onto the decay).
fn octave_filter(x: &[f64], b: Band) -> Vec<f64> {
    let len = (2 * x.len()).next_power_of_two();
    let mut re = vec![0.0; len];
    re[..x.len()].copy_from_slice(x);
    let mut im = vec![0.0; len];
    fft(&mut re, &mut im, false);
    for k in 0..len {
        let f = k.min(len - k) as f64 * FS / len as f64;
        if f < b.lower_hz() || f > b.upper_hz() {
            re[k] = 0.0;
            im[k] = 0.0;
        }
    }
    fft(&mut re, &mut im, true);
    re.truncate(x.len());
    re.iter().map(|v| v / len as f64).collect()
}

/// Unscaled in-place radix-2 FFT (the inverse without its 1/n).
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        for k in 0..half {
            let w = sign * std::f64::consts::TAU * k as f64 / len as f64;
            let (wr, wi) = (w.cos(), w.sin());
            for start in (0..n).step_by(len) {
                let (a, c) = (start + k, start + k + half);
                let (tr, ti) = (re[c] * wr - im[c] * wi, re[c] * wi + im[c] * wr);
                re[c] = re[a] - tr;
                im[c] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len *= 2;
    }
}
