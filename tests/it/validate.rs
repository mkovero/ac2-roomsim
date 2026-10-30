use ac2_roomsim::*;

use crate::common::uniform;

fn room(outline: Vec<[f64; 2]>) -> Room {
    let n = outline.len();
    Room {
        outline,
        height: Some(3.0),
        surfaces: Surfaces::uniform(n, MaterialId::Concrete.into()),
        air: Air::default(),
    }
}

#[test]
fn shoebox_is_valid_with_its_volume_and_area() {
    let p = Prism::new(&room(vec![[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0]])).unwrap();
    assert!((p.volume() - 60.0).abs() < 1e-12);
    assert!((p.surface_area() - (2.0 * 20.0 + 2.0 * 15.0 + 2.0 * 12.0)).abs() < 1e-12);
    assert_eq!(p.surface_count(), 6);
    assert_eq!(p.surfaces().last(), Some(SurfaceId::Ceiling));
}

#[test]
fn clockwise_is_refused() {
    let r = room(vec![[0.0, 0.0], [0.0, 4.0], [5.0, 4.0], [5.0, 0.0]]);
    assert_eq!(Prism::new(&r).unwrap_err(), RoomError::Clockwise);
}

#[test]
fn self_intersection_names_the_edges() {
    // A bow tie: edges 0 and 2 cross.
    let r = room(vec![[0.0, 0.0], [4.0, 4.0], [4.0, 0.0], [0.0, 4.0]]);
    assert_eq!(
        Prism::new(&r).unwrap_err(),
        RoomError::EdgesCross { a: 0, b: 2 }
    );
}

#[test]
fn touching_edges_are_refused() {
    // Vertex 4 sits on edge 0.
    let r = room(vec![
        [0.0, 0.0],
        [6.0, 0.0],
        [6.0, 4.0],
        [4.0, 4.0],
        [3.0, 0.0],
        [0.0, 4.0],
    ]);
    assert!(matches!(
        Prism::new(&r).unwrap_err(),
        RoomError::EdgesCross { .. }
    ));
}

#[test]
fn degenerate_edge_and_fold_are_refused() {
    let r = room(vec![
        [0.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
        [5.0, 4.0],
        [0.0, 4.0],
    ]);
    assert_eq!(
        Prism::new(&r).unwrap_err(),
        RoomError::DegenerateEdge { edge: 1 }
    );
    // A spike: out to (8, 2) and straight back.
    let r = room(vec![
        [0.0, 0.0],
        [5.0, 0.0],
        [5.0, 2.0],
        [8.0, 2.0],
        [5.0, 2.0],
        [5.0, 4.0],
        [0.0, 4.0],
    ]);
    assert!(matches!(
        Prism::new(&r).unwrap_err(),
        RoomError::FoldedVertex { vertex: 3 }
            | RoomError::DegenerateEdge { .. }
            | RoomError::EdgesCross { .. }
    ));
}

#[test]
fn outdoor_and_bad_inputs_are_refused() {
    let mut r = room(vec![[0.0, 0.0], [5.0, 0.0], [5.0, 4.0]]);
    r.height = None;
    assert_eq!(Prism::new(&r).unwrap_err(), RoomError::Outdoor);
    r.height = Some(0.0);
    assert_eq!(Prism::new(&r).unwrap_err(), RoomError::BadHeight);
    r.height = Some(3.0);
    r.surfaces.walls.pop();
    assert_eq!(
        Prism::new(&r).unwrap_err(),
        RoomError::WallCount { walls: 2, edges: 3 }
    );
    let r = room(vec![[0.0, 0.0], [5.0, 0.0]]);
    assert_eq!(
        Prism::new(&r).unwrap_err(),
        RoomError::TooFewVertices { count: 2 }
    );
    let r = room(vec![[0.0, 0.0], [5.0, f64::NAN], [5.0, 4.0]]);
    assert_eq!(Prism::new(&r).unwrap_err(), RoomError::NonFinite);
    let mut r = room(vec![[0.0, 0.0], [5.0, 0.0], [5.0, 4.0]]);
    r.surfaces.floor = uniform(1.5, 0.1);
    assert_eq!(
        Prism::new(&r).unwrap_err(),
        RoomError::BadCoefficient {
            surface: SurfaceId::Floor
        }
    );
    let mut r = room(vec![[0.0, 0.0], [5.0, 0.0], [5.0, 4.0]]);
    r.air.humidity_pct = 120.0;
    assert_eq!(Prism::new(&r).unwrap_err(), RoomError::BadAir);
}

#[test]
fn positions_outside_and_bad_config_are_refused() {
    let p = Prism::new(&room(vec![[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0]])).unwrap();
    let cfg = SimConfig::default();
    let src = Source {
        at: [1.0, 1.0, 1.0],
    };
    let inside = Receiver {
        at: [2.0, 2.0, 1.0],
    };
    let run =
        |s: &Source, r: &[Receiver], c: &SimConfig| simulate(&p, s, r, c, &mut Control::default());
    assert_eq!(
        run(
            &Source {
                at: [6.0, 1.0, 1.0]
            },
            &[inside],
            &cfg
        )
        .unwrap_err(),
        SimError::SourceOutside
    );
    assert_eq!(
        run(
            &src,
            &[
                inside,
                Receiver {
                    at: [2.0, 2.0, 3.5]
                }
            ],
            &cfg
        )
        .unwrap_err(),
        SimError::ReceiverOutside { index: 1 }
    );
    let bad = SimConfig {
        image_order: MAX_IMAGE_ORDER + 1,
        ..cfg.clone()
    };
    assert_eq!(
        run(&src, &[inside], &bad).unwrap_err(),
        SimError::Config(ConfigError::ImageOrder)
    );
    let bad = SimConfig {
        absorption: Absorption::Fitted {
            scale: [-1.0; BAND_COUNT],
        },
        ..cfg
    };
    assert_eq!(
        run(&src, &[inside], &bad).unwrap_err(),
        SimError::Config(ConfigError::AbsorptionScale)
    );
}

#[test]
fn material_names_round_trip() {
    for m in MaterialId::ALL {
        assert_eq!(MaterialId::from_name(m.name()), Some(m));
        let c = m.coefficients();
        assert!(
            c.absorption
                .iter()
                .chain(&c.scattering)
                .all(|v| (0.0..=1.0).contains(v))
        );
    }
    assert_eq!(MaterialId::from_name("cheese"), None);
}
