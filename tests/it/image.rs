//! Image sources in a shoebox against the analytic image lattice: along each axis the images
//! of a source at x_s in [0, L] lie at 2mL + x_s (|2m| reflections) and 2mL − x_s
//! (|2m − 1| reflections), and in a rectangular room every one of them is visible.

use ac2_roomsim::*;

use crate::common::{run, shoebox, uniform};

fn axis(l: f64, s: f64, max: i64) -> Vec<(f64, i64)> {
    let mut v = Vec::new();
    for m in -max..=max {
        v.push((2.0 * m as f64 * l + s, (2 * m).abs()));
        v.push((2.0 * m as f64 * l - s, (2 * m - 1).abs()));
    }
    v.retain(|&(_, o)| o <= max);
    v
}

#[test]
fn shoebox_matches_the_image_lattice() {
    let (lx, ly, lz) = (7.3, 5.1, 3.2);
    let alpha = 0.15;
    let src = [1.3, 2.2, 1.1];
    let rcv = [5.6, 3.9, 1.7];
    for order in [0u8, 1, 2, 3, 4] {
        let prism = shoebox(lx, ly, lz, uniform(alpha, 0.0));
        let cfg = SimConfig {
            image_order: order,
            rays_per_band: 0,
            air_absorption: false,
            duration_s: Some(0.2),
            ..SimConfig::default()
        };
        let sim = run(&prism, src, &[rcv], &cfg);
        let got = &sim.receivers[0].arrivals;

        let k = order as i64;
        let mut want = Vec::new();
        for (x, ox) in axis(lx, src[0], k) {
            for (y, oy) in axis(ly, src[1], k) {
                for (z, oz) in axis(lz, src[2], k) {
                    if ox + oy + oz <= k {
                        let d =
                            ((x - rcv[0]).powi(2) + (y - rcv[1]).powi(2) + (z - rcv[2]).powi(2))
                                .sqrt();
                        want.push((d, ox + oy + oz));
                    }
                }
            }
        }
        want.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(got.len(), want.len(), "order {order}: image count");
        for (a, &(d, o)) in got.iter().zip(&want) {
            assert!(
                (a.distance_m - d).abs() < 1e-9,
                "order {order}: {} vs {d}",
                a.distance_m
            );
            assert_eq!(
                a.path.len() as i64,
                o,
                "order {order}: reflections at {d} m"
            );
            let e = (1.0 - alpha).powi(o as i32) / (d * d);
            assert!((a.energy[3] / e - 1.0).abs() < 1e-12);
            assert!((a.time_s - d / sim.speed_of_sound).abs() < 1e-15);
        }
    }
}

#[test]
fn first_order_paths_name_their_wall() {
    let prism = shoebox(10.0, 6.0, 3.0, uniform(0.1, 0.0));
    let cfg = SimConfig {
        image_order: 1,
        rays_per_band: 0,
        ..SimConfig::default()
    };
    let sim = run(&prism, [2.0, 3.0, 1.5], &[[8.0, 3.0, 1.5]], &cfg);
    let a = &sim.receivers[0].arrivals;
    assert_eq!(a.len(), 7);
    assert!(a[0].path.is_empty());
    // Floor and ceiling are equidistant (z = 1.5 in a 3 m room), likewise the two side walls.
    let floor = a.iter().find(|x| x.path == [SurfaceId::Floor]).unwrap();
    assert!((floor.distance_m - (36.0f64 + 9.0).sqrt()).abs() < 1e-12);
    let back = a
        .iter()
        .find(|x| x.path == [SurfaceId::Wall { edge: 3 }])
        .unwrap();
    assert!((back.distance_m - 10.0).abs() < 1e-12);
    assert!((back.image[0] + 2.0).abs() < 1e-12);
}

#[test]
fn scattering_takes_energy_off_the_specular_path() {
    let prism = shoebox(10.0, 6.0, 3.0, uniform(0.2, 0.25));
    let cfg = SimConfig {
        image_order: 2,
        rays_per_band: 0,
        air_absorption: false,
        ..SimConfig::default()
    };
    let sim = run(&prism, [2.0, 3.0, 1.5], &[[8.0, 2.0, 1.2]], &cfg);
    for a in &sim.receivers[0].arrivals {
        let e = (0.8f64 * 0.75).powi(a.path.len() as i32) / a.distance_m.powi(2);
        assert!((a.energy[0] / e - 1.0).abs() < 1e-12);
    }
}
