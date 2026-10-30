use ac2_roomsim::*;

/// A uniform material: absorption `alpha` and scattering `s` in every band.
pub fn uniform(alpha: f64, s: f64) -> Material {
    Material::Custom(Coefficients {
        absorption: [alpha; BAND_COUNT],
        scattering: [s; BAND_COUNT],
    })
}

/// An `lx × ly × lz` shoebox with its corner at the origin, every surface `material`.
pub fn shoebox(lx: f64, ly: f64, lz: f64, material: Material) -> Prism {
    Prism::new(&Room {
        outline: vec![[0.0, 0.0], [lx, 0.0], [lx, ly], [0.0, ly]],
        height: Some(lz),
        surfaces: Surfaces::uniform(4, material),
        air: Air::default(),
    })
    .unwrap()
}

/// The L-shaped room of the visibility tests: a 10 × 10 m square with the 6 × 6 m corner
/// at x > 4, y > 4 cut away, 3 m high.
///
/// ```text
///  y
/// 10 +---+            edges: 0 (0,0)-(10,0)   1 (10,0)-(10,4)
///    |   |                   2 (10,4)-(4,4)   3 (4,4)-(4,10)
///  4 |   +-------+           4 (4,10)-(0,10)  5 (0,10)-(0,0)
///    |           |
///  0 +-----------+ x
///    0   4      10
/// ```
pub fn l_shape(material: Material) -> Prism {
    Prism::new(&Room {
        outline: vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 4.0],
            [4.0, 4.0],
            [4.0, 10.0],
            [0.0, 10.0],
        ],
        height: Some(3.0),
        surfaces: Surfaces::uniform(6, material),
        air: Air::default(),
    })
    .unwrap()
}

pub fn run(prism: &Prism, src: [f64; 3], rcv: &[[f64; 3]], config: &SimConfig) -> Simulation {
    let receivers: Vec<_> = rcv.iter().map(|&at| Receiver { at }).collect();
    simulate(
        prism,
        &Source { at: src },
        &receivers,
        config,
        &mut Control::default(),
    )
    .unwrap()
}
