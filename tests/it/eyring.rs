//! Reverberation time from ray tracing with fully diffuse reflections and uniform absorption
//! against Eyring's formula. With every reflection Lambertian the field becomes diffuse after
//! a few reflections, and each ray then meets a wall every mean free path 4V/S on average,
//! which is Eyring's assumption.
//!
//! Eyring takes every free path to be the mean one. Real free paths scatter about it, with
//! relative variance γ², and the decay is then slower: Kuttruff (*Room Acoustics*, 5th ed.,
//! §5.4) replaces −ln(1 − α) by −ln(1 − α)·(1 + (γ²/2)·ln(1 − α)), with γ² ≈ 0.4 for
//! rectangular rooms of ordinary proportions. The excess is about γ²/2·α, so plain Eyring
//! holds to 5 % only up to α ≈ 0.2; the corrected formula is checked over a wider range.

use ac2_roomsim::*;

use crate::common::{run, shoebox, uniform};

/// Kuttruff's path-variance factor on Eyring's T (see the module docs).
fn kuttruff_factor(alpha: f64) -> f64 {
    let gamma2 = 0.4;
    1.0 / (1.0 + gamma2 / 2.0 * (1.0 - alpha).ln())
}

fn decay_error(rays: u64, alpha: f64, air: bool) -> Vec<f64> {
    let prism = shoebox(10.0, 8.0, 4.0, uniform(alpha, 1.0));
    let cfg = SimConfig {
        rays_per_band: rays,
        air_absorption: air,
        image_order: 0,
        seed: 7,
        ..SimConfig::default()
    };
    let sim = run(&prism, [2.5, 3.0, 1.6], &[[7.0, 5.0, 1.2]], &cfg);
    let eg = sim.receivers[0].echogram();
    Band::ALL
        .iter()
        .map(|&b| {
            let t = eg.t30(b).expect("a T30 fit");
            t / sim.eyring_t_s[b.index()] - 1.0
        })
        .collect()
}

#[test]
fn t30_near_eyring_with_few_rays() {
    // 2 000 rays per band: a noisier estimate, held to 8 %.
    for e in decay_error(2_000, 0.2, false) {
        assert!(e.abs() < 0.08, "T30 off Eyring by {:.1} %", 100.0 * e);
    }
}

#[test]
fn slow_t30_within_5_percent_of_eyring() {
    for alpha in [0.05, 0.1, 0.15] {
        let errs = decay_error(40_000, alpha, false);
        eprintln!(
            "α = {alpha}: T30 / Eyring − 1 per band (%): {:?}",
            percent(&errs)
        );
        for e in errs {
            assert!(
                e.abs() < 0.05,
                "α = {alpha}: T30 off Eyring by {:.1} %",
                100.0 * e
            );
        }
    }
}

#[test]
fn slow_t30_near_eyring_with_path_variance() {
    // The correction is a second-order expansion in ln(1 − α), so it tightens the check
    // most where α is small: 2 % up to α = 0.2, 3 % at 0.4.
    for (alpha, tol) in [(0.1, 0.02), (0.2, 0.02), (0.4, 0.03)] {
        let k = kuttruff_factor(alpha);
        let errs: Vec<f64> = decay_error(40_000, alpha, false)
            .iter()
            .map(|e| (1.0 + e) / k - 1.0)
            .collect();
        eprintln!(
            "α = {alpha}: T30 / Kuttruff − 1 per band (%): {:?}",
            percent(&errs)
        );
        for e in errs {
            assert!(
                e.abs() < tol,
                "α = {alpha}: T30 off Kuttruff by {:.1} %",
                100.0 * e
            );
        }
    }
}

fn percent(errs: &[f64]) -> Vec<f64> {
    errs.iter().map(|e| (1000.0 * e).round() / 10.0).collect()
}

#[test]
fn slow_air_absorption_shortens_t_as_eyring_says() {
    // Air at 8 kHz takes 4mV of absorption area; the simulated T must follow it.
    let errs = decay_error(40_000, 0.1, true);
    eprintln!(
        "with air: T30 / Eyring − 1 per band (%): {:?}",
        percent(&errs)
    );
    for e in errs {
        assert!(
            e.abs() < 0.05,
            "T30 off Eyring (with air) by {:.1} %",
            100.0 * e
        );
    }
}

#[test]
fn eyring_formula_on_a_known_room() {
    // 10 × 8 × 4: V = 320 m³, S = 304 m², ᾱ = 0.2: T = 24 ln10 · V / (c · −S ln 0.8).
    let prism = shoebox(10.0, 8.0, 4.0, uniform(0.2, 1.0));
    let t = prism.eyring_t(&[1.0; BAND_COUNT], false);
    let c = prism.air().speed_of_sound();
    let want = 24.0 * std::f64::consts::LN_10 * 320.0 / (c * -304.0 * 0.8f64.ln());
    assert!((t[0] - want).abs() < 1e-12);
    // Half the absorption scale: Eyring with ᾱ = 0.1.
    let t = prism.eyring_t(&[0.5; BAND_COUNT], false);
    let want = 24.0 * std::f64::consts::LN_10 * 320.0 / (c * -304.0 * 0.9f64.ln());
    assert!((t[3] - want).abs() < 1e-12);
}

#[test]
fn schroeder_frequency_follows_t_and_v() {
    let prism = shoebox(10.0, 8.0, 4.0, uniform(0.2, 1.0));
    let cfg = SimConfig {
        rays_per_band: 0,
        air_absorption: false,
        ..SimConfig::default()
    };
    let sim = run(&prism, [2.0, 2.0, 1.5], &[[6.0, 5.0, 1.5]], &cfg);
    let t = sim.eyring_t_s[3];
    assert!((sim.schroeder_hz - 2000.0 * (t / 320.0).sqrt()).abs() < 1e-9);
    assert_eq!(sim.valid_above_hz, sim.schroeder_hz);
    // About 98 Hz here: the 125 Hz band's lower edge (89 Hz) is below it.
    assert!(!sim.band_valid(Band::Hz125));
    assert!(sim.band_valid(Band::Hz250));
}
