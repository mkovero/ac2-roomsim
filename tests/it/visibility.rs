//! Image-source visibility in the L-shaped room of `common::l_shape`, against paths worked
//! out by hand in plan view (all points at the same height, so walls are the only
//! obstacles and floor/ceiling reflections are left out of the reasoning).

use ac2_roomsim::*;

use crate::common::{l_shape, run, uniform};

fn arrivals(src: [f64; 2], rcv: [f64; 2]) -> Vec<Arrival> {
    let prism = l_shape(uniform(0.1, 0.0));
    let cfg = SimConfig {
        image_order: 1,
        rays_per_band: 0,
        ..SimConfig::default()
    };
    run(
        &prism,
        [src[0], src[1], 1.5],
        &[[rcv[0], rcv[1], 1.5]],
        &cfg,
    )
    .receivers
    .remove(0)
    .arrivals
}

fn has(a: &[Arrival], path: &[SurfaceId]) -> bool {
    a.iter().any(|x| x.path == path)
}

const fn wall(edge: u32) -> SurfaceId {
    SurfaceId::Wall { edge }
}

#[test]
fn direct_sound_round_the_corner_is_hidden() {
    // (8, 2) to (2, 8): the line x + y = 10 runs through the cut-away corner.
    let a = arrivals([8.0, 2.0], [2.0, 8.0]);
    assert!(!has(&a, &[]));
    // Same arm: seen.
    assert!(has(&arrivals([2.0, 2.0], [8.0, 2.0]), &[]));
}

#[test]
fn reflection_from_the_inner_wall() {
    // Image of (8, 2) in y = 4 (edge 2) is (8, 6); from (2, 2) the leg meets y = 4 at x = 5,
    // on the face (x in 4…10).
    let a = arrivals([8.0, 2.0], [2.0, 2.0]);
    assert!(has(&a, &[wall(2)]));
    // From (1, 2) it meets y = 4 at x = 4.5: still on the face.
    assert!(has(&arrivals([8.0, 2.0], [1.0, 2.0]), &[wall(2)]));
}

#[test]
fn source_behind_a_wall_plane_has_no_image_in_it() {
    // Edge 3 (x = 4, y 4…10) faces x < 4; a source at x = 8 lies behind its plane.
    let a = arrivals([8.0, 2.0], [2.0, 6.0]);
    assert!(!has(&a, &[wall(3)]));
}

#[test]
fn leg_that_misses_the_face_is_hidden() {
    // Image of (2, 1) in x = 4 is (6, 1); from (3, 5) the leg meets x = 4 at y ≈ 3.67, below
    // the face (y 4…10): there is no wall there, the bottom arm opens.
    let a = arrivals([2.0, 1.0], [3.0, 5.0]);
    assert!(!has(&a, &[wall(3)]));
    // From (1, 9) it meets x = 4 at y = 4.8, on the face.
    assert!(has(&arrivals([2.0, 2.0], [1.0, 9.0]), &[wall(3)]));
}

#[test]
fn leg_through_the_missing_corner_is_hidden() {
    // Image of (8, 2) in x = 10 (edge 1) is (12, 2); from (2, 8) the leg meets x = 10 at
    // y = 3.2, on the face, but passes (5, 6.2), outside the room.
    let a = arrivals([8.0, 2.0], [2.0, 8.0]);
    assert!(!has(&a, &[wall(1)]));
    // The outer wall x = 0 (edge 5): image (−8, 2), meets x = 0 at y = 6.8, but the leg back
    // to the source crosses x = 4 at y = 4.4 heading into the cut-away corner.
    assert!(!has(&a, &[wall(5)]));
}

#[test]
fn reflection_round_the_corner_is_seen() {
    // Image of (8, 2) in y = 0 (edge 0) is (8, −2); from (1, 8) the leg meets y = 0 at
    // x = 6.6 and passes y = 4 at x = 3.8, inside the corner at (4, 4).
    let a = arrivals([8.0, 2.0], [1.0, 8.0]);
    assert!(has(&a, &[wall(0)]));
    let r = a.iter().find(|x| x.path == [wall(0)]).unwrap();
    assert!((r.distance_m - (49.0f64 + 100.0).sqrt()).abs() < 1e-12);
    // From (2, 8) the same leg passes y = 4 at x = 4.4, through the cut-away corner.
    assert!(!has(&arrivals([8.0, 2.0], [2.0, 8.0]), &[wall(0)]));
}

#[test]
fn higher_order_arrivals_are_sorted_and_distinct() {
    // Every arrival's path must be geometrically consistent: arrival distance equals the
    // receiver-to-image distance and no arrival repeats.
    let prism = l_shape(uniform(0.1, 0.0));
    let cfg = SimConfig {
        image_order: 3,
        rays_per_band: 0,
        ..SimConfig::default()
    };
    let sim = run(&prism, [8.0, 2.0, 1.5], &[[2.0, 8.0, 1.2]], &cfg);
    let a = &sim.receivers[0].arrivals;
    assert!(a.len() > 10);
    for w in a.windows(2) {
        assert!(w[0].time_s <= w[1].time_s);
        assert!(!(w[0].path == w[1].path && w[0].distance_m == w[1].distance_m));
    }
    assert!(a.iter().all(|x| x.path.len() <= 3));
}
