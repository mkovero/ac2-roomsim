//! Air absorption against ISO 9613-2:1996 Table 2, whose values ISO 9613-1 computes at the
//! exact octave midband frequencies, tabulated to 0.1 dB/km (1 dB/km above 100).

use ac2_roomsim::*;

/// (°C, % RH, dB/km at 63, 125, 250, 500, 1k, 2k, 4k, 8k Hz).
const TABLE: [(f64, f64, [f64; 8]); 6] = [
    (10.0, 70.0, [0.1, 0.4, 1.0, 1.9, 3.7, 9.7, 32.8, 117.0]),
    (20.0, 70.0, [0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6]),
    (30.0, 70.0, [0.1, 0.3, 1.0, 3.1, 7.4, 12.7, 23.1, 59.3]),
    (15.0, 20.0, [0.3, 0.6, 1.2, 2.7, 8.2, 28.2, 88.8, 202.0]),
    (15.0, 50.0, [0.1, 0.5, 1.2, 2.2, 4.2, 10.8, 36.2, 129.0]),
    (15.0, 80.0, [0.1, 0.3, 1.1, 2.4, 4.1, 8.3, 23.7, 82.8]),
];

#[test]
fn matches_iso_9613_table() {
    for (t, h, row) in TABLE {
        let air = Air {
            temperature_c: t,
            humidity_pct: h,
            pressure_kpa: 101.325,
        };
        for (k, &want) in row.iter().enumerate() {
            let f = 1000.0 * 10f64.powf(0.3 * (k as f64 - 4.0));
            let got = air.attenuation_db_per_m(f) * 1000.0;
            // Agreement to the table's rounding (half its last digit), plus 0.01 dB/km for
            // values that sit on a rounding edge (15 °C, 80 %, 1 kHz computes to 4.15).
            let step = if want >= 100.0 { 1.0 } else { 0.1 };
            assert!(
                (got - want).abs() <= 0.5 * step + 0.01,
                "{t} °C {h} %: {f:.0} Hz {got:.3} dB/km, table {want}"
            );
        }
    }
}

#[test]
fn energy_attenuation_is_db_over_4_34() {
    let air = Air::default();
    let m = air.energy_attenuation();
    for b in Band::ALL {
        let db = air.attenuation_db_per_m(b.exact_hz());
        // Over 1 km the energy factor e^(−m·1000) must equal 10^(−α·1000/10).
        let lhs = (-m[b.index()] * 1000.0).exp();
        let rhs = 10f64.powf(-db * 1000.0 / 10.0);
        assert!((lhs / rhs - 1.0).abs() < 1e-9);
    }
}

#[test]
fn speed_of_sound_at_20_c() {
    assert!((Air::default().speed_of_sound() - 343.2).abs() < 0.1);
}
