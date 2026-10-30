//! Octave bands. Geometric acoustics is run per band because absorption, scattering and air
//! absorption all depend on frequency; seven octaves (125 Hz–8 kHz) cover what room-acoustic
//! parameters are quoted in (ISO 3382-1 uses 125 Hz–4 kHz, 8 kHz is where air dominates).

/// Number of octave bands.
pub const BAND_COUNT: usize = 7;

/// A value per octave band, indexed by [`Band::index`].
pub type PerBand = [f64; BAND_COUNT];

/// One octave band, named by its nominal centre frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Band {
    /// 125 Hz.
    Hz125,
    /// 250 Hz.
    Hz250,
    /// 500 Hz.
    Hz500,
    /// 1 kHz.
    Hz1k,
    /// 2 kHz.
    Hz2k,
    /// 4 kHz.
    Hz4k,
    /// 8 kHz.
    Hz8k,
}

impl Band {
    /// All bands, low to high.
    pub const ALL: [Band; BAND_COUNT] = [
        Band::Hz125,
        Band::Hz250,
        Band::Hz500,
        Band::Hz1k,
        Band::Hz2k,
        Band::Hz4k,
        Band::Hz8k,
    ];

    /// Position in a [`PerBand`] array.
    pub fn index(self) -> usize {
        self as usize
    }

    /// Nominal centre frequency, Hz (the label: 125, 250, …, 8000).
    pub fn nominal_hz(self) -> f64 {
        [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0][self.index()]
    }

    /// Exact base-10 midband frequency, Hz: 1000·10^(3k/10) (IEC 61260-1), which the nominal
    /// value rounds. Air absorption grows with f², so 8 kHz versus 7.94 kHz is a 1.5 % change.
    pub fn exact_hz(self) -> f64 {
        let k = self.index() as f64 - 3.0;
        1000.0 * 10f64.powf(3.0 * k / 10.0)
    }

    /// Lower band edge, Hz: centre / 2^(1/2).
    pub fn lower_hz(self) -> f64 {
        self.exact_hz() / std::f64::consts::SQRT_2
    }

    /// Upper band edge, Hz: centre · 2^(1/2).
    pub fn upper_hz(self) -> f64 {
        self.exact_hz() * std::f64::consts::SQRT_2
    }
}
