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
    // Each band exact; what is left is chance correlation between neighbouring bands' noise.
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
    let direct = (r.arrivals[0].time_s * 48_000.0).round() as usize;
    assert!(a[..direct].iter().all(|&p| p == 0.0));
    assert!(a[direct..direct + 48].iter().any(|&p| p != 0.0));
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
