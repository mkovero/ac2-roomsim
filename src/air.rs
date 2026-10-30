//! Air: the speed of sound and atmospheric absorption after ISO 9613-1:1993.
//!
//! Air absorbs sound through classical viscosity and heat conduction and, much more strongly,
//! through the vibrational relaxation of oxygen and nitrogen molecules. Water vapour shortens
//! the relaxation times, so the absorption depends on humidity in a non-monotonic way. Over
//! a reverberation tail the path lengths are hundreds of metres, which is why air absorption
//! sets T at 4–8 kHz in a large hall regardless of the surfaces.

use crate::band::{BAND_COUNT, Band, PerBand};

/// The air in the room.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Air {
    /// Temperature, °C.
    pub temperature_c: f64,
    /// Relative humidity, percent (0–100).
    pub humidity_pct: f64,
    /// Ambient pressure, kPa.
    pub pressure_kpa: f64,
}

impl Default for Air {
    /// 20 °C, 50 % RH, 101.325 kPa: a typical heated hall.
    fn default() -> Self {
        Air {
            temperature_c: 20.0,
            humidity_pct: 50.0,
            pressure_kpa: 101.325,
        }
    }
}

/// Reference ambient pressure, kPa (ISO 9613-1).
const P_REF_KPA: f64 = 101.325;
/// Reference temperature, K (ISO 9613-1).
const T_REF_K: f64 = 293.15;
/// Triple-point isotherm temperature, K (ISO 9613-1 annex B).
const T_01_K: f64 = 273.16;

impl Air {
    /// Speed of sound, m/s: c = 331.3·√(1 + θ/273.15), the ideal-gas value for dry air at
    /// temperature θ (°C). Humidity raises it by under 0.5 %, below what the geometry's
    /// uncertainty allows us to resolve.
    pub fn speed_of_sound(&self) -> f64 {
        331.3 * (1.0 + self.temperature_c / 273.15).sqrt()
    }

    /// Pure-tone attenuation coefficient at frequency `f_hz`, dB per metre (ISO 9613-1
    /// equation 5).
    pub fn attenuation_db_per_m(&self, f_hz: f64) -> f64 {
        let t = self.temperature_c + 273.15;
        let pa = self.pressure_kpa / P_REF_KPA;
        // Molar concentration of water vapour, percent (annex B): saturation pressure from
        // the triple-point isotherm.
        let c = -6.8346 * (T_01_K / t).powf(1.261) + 4.6151;
        let psat = 10f64.powf(c);
        let h = self.humidity_pct * psat / pa;
        let tr = t / T_REF_K;
        // Relaxation frequencies of oxygen and nitrogen (equations 3 and 4).
        let fr_o = pa * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
        let fr_n =
            pa * tr.powf(-0.5) * (9.0 + 280.0 * h * (-4.170 * (tr.powf(-1.0 / 3.0) - 1.0)).exp());
        let f2 = f_hz * f_hz;
        8.686
            * f2
            * (1.84e-11 / pa * tr.sqrt()
                + tr.powf(-2.5)
                    * (0.01275 * (-2239.1 / t).exp() / (fr_o + f2 / fr_o)
                        + 0.1068 * (-3352.0 / t).exp() / (fr_n + f2 / fr_n)))
    }

    /// Energy attenuation coefficient per band, 1/m: intensity falls as e^(−m·d) over a path
    /// of `d` metres. Evaluated at each band's exact midband frequency, the usual
    /// approximation for an octave of broadband sound.
    pub fn energy_attenuation(&self) -> PerBand {
        let mut m = [0.0; BAND_COUNT];
        for b in Band::ALL {
            // dB to nepers of intensity: 10·log10(e) dB per neper.
            m[b.index()] =
                self.attenuation_db_per_m(b.exact_hz()) / (10.0 * std::f64::consts::LOG10_E);
        }
        m
    }
}
