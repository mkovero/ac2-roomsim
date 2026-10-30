//! Surface materials: random-incidence absorption and scattering coefficients per octave band.
//!
//! The absorption values are typical published random-incidence coefficients for common
//! constructions. The 125 Hz–4 kHz values follow the widely reproduced tables in
//! M. Long, *Architectural Acoustics* (Elsevier 2006, ch. 7), F. A. Everest & K. C. Pohlmann,
//! *Master Handbook of Acoustics* (McGraw-Hill, absorption-coefficient appendix) and, for
//! audience, L. Beranek, *Concert Halls and Opera Houses* (Springer 2004, occupied
//! upholstered seating). Those tables stop at 4 kHz; the 8 kHz value repeats 4 kHz, since
//! porous and rigid materials both change little in that octave and air absorption dominates
//! the decay there in any room large enough to need simulating.
//!
//! Scattering coefficients are not tabulated for generic surfaces; the values follow the
//! guidance in M. Vorländer, *Auralization* (Springer 2008, ch. 11): about 0.05–0.1 for flat
//! walls, rising with frequency as the wavelength approaches the surface's relief, and large
//! (0.4–0.7) for seated audience. They are assumptions, which is why only an absorption scale
//! is ever fitted to measurement.
//!
//! On site the real absorption is unknown, so a simulation from this table is labelled
//! *absorption assumed* until a per-band scale has been fitted (see [`crate::Absorption`]).

use crate::band::PerBand;

/// A built-in material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum MaterialId {
    /// Smooth poured concrete, unpainted, or a concrete floor.
    Concrete,
    /// Concrete block, painted.
    ConcreteBlockPainted,
    /// Unglazed brick.
    Brick,
    /// Gypsum plaster on masonry.
    PlasterOnMasonry,
    /// 12.5 mm gypsum board on studs with an air gap (a resonant panel: absorbs bass).
    GypsumBoard,
    /// Large panes of heavy glass.
    GlassHeavy,
    /// Ordinary window glass.
    GlassWindow,
    /// Wooden floor on joists.
    WoodFloor,
    /// Thin plywood or wood panelling with an air space behind.
    WoodPanel,
    /// Heavy carpet laid directly on concrete.
    CarpetOnConcrete,
    /// Heavy carpet on underlay.
    CarpetOnUnderlay,
    /// Heavy velour curtain, draped to half its flat area.
    CurtainHeavy,
    /// Suspended mineral-fibre ceiling tiles.
    AcousticTile,
    /// 50 mm mineral wool on a rigid backing.
    MineralWool50,
    /// Audience seated in upholstered seats (per floor area they occupy).
    AudienceSeated,
}

impl MaterialId {
    /// Every built-in material.
    pub const ALL: [MaterialId; 15] = [
        MaterialId::Concrete,
        MaterialId::ConcreteBlockPainted,
        MaterialId::Brick,
        MaterialId::PlasterOnMasonry,
        MaterialId::GypsumBoard,
        MaterialId::GlassHeavy,
        MaterialId::GlassWindow,
        MaterialId::WoodFloor,
        MaterialId::WoodPanel,
        MaterialId::CarpetOnConcrete,
        MaterialId::CarpetOnUnderlay,
        MaterialId::CurtainHeavy,
        MaterialId::AcousticTile,
        MaterialId::MineralWool50,
        MaterialId::AudienceSeated,
    ];

    /// The stable name (snake_case), as a plan stores it.
    pub fn name(self) -> &'static str {
        match self {
            MaterialId::Concrete => "concrete",
            MaterialId::ConcreteBlockPainted => "concrete_block_painted",
            MaterialId::Brick => "brick",
            MaterialId::PlasterOnMasonry => "plaster_on_masonry",
            MaterialId::GypsumBoard => "gypsum_board",
            MaterialId::GlassHeavy => "glass_heavy",
            MaterialId::GlassWindow => "glass_window",
            MaterialId::WoodFloor => "wood_floor",
            MaterialId::WoodPanel => "wood_panel",
            MaterialId::CarpetOnConcrete => "carpet_on_concrete",
            MaterialId::CarpetOnUnderlay => "carpet_on_underlay",
            MaterialId::CurtainHeavy => "curtain_heavy",
            MaterialId::AcousticTile => "acoustic_tile",
            MaterialId::MineralWool50 => "mineral_wool_50",
            MaterialId::AudienceSeated => "audience_seated",
        }
    }

    /// The material with this [`name`](Self::name), if any.
    pub fn from_name(name: &str) -> Option<MaterialId> {
        MaterialId::ALL.into_iter().find(|m| m.name() == name)
    }

    /// The coefficients of this material.
    pub fn coefficients(self) -> Coefficients {
        let (absorption, scattering): (PerBand, PerBand) = match self {
            MaterialId::Concrete => (
                [0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03],
                [0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.15],
            ),
            MaterialId::ConcreteBlockPainted => (
                [0.10, 0.05, 0.06, 0.07, 0.09, 0.08, 0.08],
                [0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.25],
            ),
            MaterialId::Brick => (
                [0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.07],
                [0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.25],
            ),
            MaterialId::PlasterOnMasonry => (
                [0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05],
                [0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.15],
            ),
            MaterialId::GypsumBoard => (
                [0.29, 0.10, 0.05, 0.04, 0.07, 0.09, 0.09],
                [0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.15],
            ),
            MaterialId::GlassHeavy => (
                [0.18, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02],
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            ),
            MaterialId::GlassWindow => (
                [0.35, 0.25, 0.18, 0.12, 0.07, 0.04, 0.04],
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            ),
            MaterialId::WoodFloor => (
                [0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07],
                [0.05, 0.05, 0.05, 0.10, 0.10, 0.15, 0.20],
            ),
            MaterialId::WoodPanel => (
                [0.28, 0.22, 0.17, 0.09, 0.10, 0.11, 0.11],
                [0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.25],
            ),
            MaterialId::CarpetOnConcrete => (
                [0.02, 0.06, 0.14, 0.37, 0.60, 0.65, 0.65],
                [0.05, 0.05, 0.05, 0.10, 0.10, 0.15, 0.20],
            ),
            MaterialId::CarpetOnUnderlay => (
                [0.08, 0.24, 0.57, 0.69, 0.71, 0.73, 0.73],
                [0.05, 0.05, 0.05, 0.10, 0.10, 0.15, 0.20],
            ),
            MaterialId::CurtainHeavy => (
                [0.14, 0.35, 0.55, 0.72, 0.70, 0.65, 0.65],
                [0.10, 0.20, 0.30, 0.40, 0.50, 0.50, 0.50],
            ),
            MaterialId::AcousticTile => (
                [0.50, 0.65, 0.75, 0.80, 0.80, 0.75, 0.75],
                [0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.25],
            ),
            MaterialId::MineralWool50 => (
                [0.20, 0.65, 0.90, 0.95, 0.95, 0.95, 0.95],
                [0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.25],
            ),
            MaterialId::AudienceSeated => (
                [0.60, 0.74, 0.88, 0.96, 0.93, 0.85, 0.85],
                [0.30, 0.40, 0.50, 0.60, 0.70, 0.70, 0.70],
            ),
        };
        Coefficients {
            absorption,
            scattering,
        }
    }
}

/// Random-incidence coefficients of a surface, per octave band, each in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Coefficients {
    /// Share of incident energy absorbed (not reflected).
    pub absorption: PerBand,
    /// Share of reflected energy scattered diffusely (Lambert) instead of specularly
    /// (ISO 17497-1 scattering coefficient).
    pub scattering: PerBand,
}

/// The material of one surface: a built-in one or explicit coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Material {
    /// A material from the built-in table.
    Builtin(MaterialId),
    /// Explicit coefficients.
    Custom(Coefficients),
}

impl Material {
    /// The coefficients this material stands for.
    pub fn coefficients(&self) -> Coefficients {
        match self {
            Material::Builtin(id) => id.coefficients(),
            Material::Custom(c) => *c,
        }
    }
}

impl From<MaterialId> for Material {
    fn from(id: MaterialId) -> Self {
        Material::Builtin(id)
    }
}
