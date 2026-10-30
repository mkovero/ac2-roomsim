//! Determinism, cancel, budget and progress.

use std::sync::atomic::AtomicBool;
use std::time::Duration;

use ac2_roomsim::*;

use crate::common::{l_shape, uniform};

fn setup() -> (Prism, Source, Vec<Receiver>, SimConfig) {
    let prism = l_shape(uniform(0.15, 0.3));
    let src = Source {
        at: [8.0, 2.0, 1.5],
    };
    let rcv = vec![
        Receiver {
            at: [2.0, 8.0, 1.2],
        },
        Receiver {
            at: [2.0, 2.0, 1.2],
        },
    ];
    let cfg = SimConfig {
        rays_per_band: 5_000,
        seed: 42,
        ..SimConfig::default()
    };
    (prism, src, rcv, cfg)
}

#[test]
fn same_seed_same_result_on_any_thread_count() {
    let (prism, src, rcv, cfg) = setup();
    let on = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| simulate(&prism, &src, &rcv, &cfg, &mut Control::default()).unwrap())
    };
    let one = on(1);
    assert_eq!(one, on(5));
    assert_eq!(one.rays_per_band, 5_000);
    let other = simulate(
        &prism,
        &src,
        &rcv,
        &SimConfig { seed: 43, ..cfg },
        &mut Control::default(),
    )
    .unwrap();
    assert_ne!(one.receivers[0].late, other.receivers[0].late);
}

#[test]
fn l_shape_keeps_rays_in() {
    let (prism, src, rcv, cfg) = setup();
    let sim = simulate(&prism, &src, &rcv, &cfg, &mut Control::default()).unwrap();
    // Rounding may lose a ray at an exact edge hit; a handful in 35 000 at most.
    assert!(sim.lost_rays <= 5, "{} rays lost", sim.lost_rays);
    // Round the corner there is late energy though the direct sound is hidden.
    let r = &sim.receivers[0];
    assert!(r.arrivals.iter().all(|a| !a.path.is_empty()));
    assert!(r.late.band(Band::Hz1k).iter().sum::<f64>() > 0.0);
}

#[test]
fn cancel_stops_the_run() {
    let (prism, src, rcv, cfg) = setup();
    let flag = AtomicBool::new(true);
    let mut control = Control {
        cancel: Some(&flag),
        ..Control::default()
    };
    assert_eq!(
        simulate(&prism, &src, &rcv, &cfg, &mut control).unwrap_err(),
        SimError::Cancelled
    );
}

#[test]
fn spent_budget_keeps_the_first_round() {
    let (prism, src, rcv, cfg) = setup();
    let cfg = SimConfig {
        rays_per_band: 100_000,
        ..cfg
    };
    let mut control = Control {
        budget: Some(Duration::ZERO),
        ..Control::default()
    };
    let sim = simulate(&prism, &src, &rcv, &cfg, &mut control).unwrap();
    assert!(sim.budget_exhausted);
    assert_eq!(sim.rays_per_band, 4096);
    // Normalised by the rays actually traced: the late energy is on the full run's scale.
    let full = simulate(
        &prism,
        &src,
        &rcv,
        &SimConfig {
            rays_per_band: 4096,
            ..cfg
        },
        &mut Control::default(),
    )
    .unwrap();
    assert_eq!(sim.receivers, full.receivers);
}

#[test]
fn progress_reaches_the_end_of_each_stage() {
    let (prism, src, rcv, cfg) = setup();
    let mut seen = Vec::new();
    let mut f = |p: Progress| seen.push(p);
    let mut control = Control {
        progress: Some(&mut f),
        ..Control::default()
    };
    simulate(&prism, &src, &rcv, &cfg, &mut control).unwrap();
    let last = |stage| {
        seen.iter()
            .rev()
            .find(|p| p.stage == stage)
            .copied()
            .unwrap()
    };
    let img = last(Stage::ImageSources);
    assert_eq!((img.done, img.total), (2, 2));
    let rays = last(Stage::RayTracing);
    assert_eq!(rays.done, rays.total);
    assert_eq!(rays.total, 2);
    assert!(
        seen.windows(2)
            .all(|w| w[0].stage != w[1].stage || w[0].done <= w[1].done)
    );
}

#[cfg(feature = "serde")]
#[test]
fn result_round_trips_through_json() {
    let (prism, src, rcv, cfg) = setup();
    let cfg = SimConfig {
        rays_per_band: 1000,
        ..cfg
    };
    let sim = simulate(&prism, &src, &rcv, &cfg, &mut Control::default()).unwrap();
    let back: Simulation = serde_json::from_str(&serde_json::to_string(&sim).unwrap()).unwrap();
    assert_eq!(back, sim);
}
