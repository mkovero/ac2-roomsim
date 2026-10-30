//! A simulation run: image sources for the early specular part, ray tracing for the rest,
//! under a time budget with progress and a cancel flag.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::band::{BAND_COUNT, Band, PerBand};
use crate::echogram::{Echogram, IrError, synthesize_ir};
use crate::image::{Arrival, arrivals, images};
use crate::ray::{RayCtx, trace_chunk};
use crate::room::Prism;

/// An omnidirectional point source.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Source {
    /// Position, m: plan x, plan y, height above the floor.
    pub at: [f64; 3],
}

/// A receiver (a microphone or listener position).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Receiver {
    /// Position, m: plan x, plan y, height above the floor.
    pub at: [f64; 3],
}

/// Where the absorption coefficients come from.
///
/// Absorption is not known on site: a built-in table is a guess. One scale per band can be
/// identified from measurement (the simulated T20 matched to the measured one); per-surface
/// absorption cannot, so after a fit the geometry of a reflection holds but its level is
/// still the table's split between surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum Absorption {
    /// The materials' coefficients as given: absorption assumed.
    #[default]
    Table,
    /// Coefficients scaled per band, not (yet) fitted to a measurement, such as a fit's trial
    /// step: absorption still assumed.
    Scaled {
        /// Multiplier per band on every surface's absorption (the product clipped to 1).
        scale: PerBand,
    },
    /// Coefficients scaled per band by a fit to measured decay times.
    Fitted {
        /// Multiplier per band on every surface's absorption (the product clipped to 1).
        scale: PerBand,
    },
}

impl Absorption {
    /// The per-band multiplier.
    pub fn scale(&self) -> PerBand {
        match self {
            Absorption::Table => [1.0; BAND_COUNT],
            Absorption::Scaled { scale } | Absorption::Fitted { scale } => *scale,
        }
    }

    /// Whether the absorption was fitted to a measurement.
    pub fn is_fitted(&self) -> bool {
        matches!(self, Absorption::Fitted { .. })
    }
}

/// How to run a simulation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimConfig {
    /// Highest image-source order (reflections per specular path), at most
    /// [`MAX_IMAGE_ORDER`]. The image count grows as (surfaces − 1)^order.
    pub image_order: u8,
    /// Rays per band; 0 skips ray tracing (image sources only).
    pub rays_per_band: u64,
    /// Seed: the same seed, room and config give the same result on any thread count.
    pub seed: u64,
    /// Receiver sphere radius, m. Larger spheres catch more rays (less noise) but blur
    /// arrival times by up to 2r/c and must stay clear of walls to see the field at the
    /// point.
    pub receiver_radius_m: f64,
    /// Echogram bin width, s.
    pub bin_s: f64,
    /// Echogram length, s. `None`: 1.2 times the longest Eyring T over the bands (clamped
    /// to 0.1–10 s), long enough for a T30 fit.
    pub duration_s: Option<f64>,
    /// Include air absorption (ISO 9613-1). Off only to compare with surface-only theory.
    pub air_absorption: bool,
    /// Absorption scale.
    pub absorption: Absorption,
}

/// The highest image-source order accepted: at order 8 a 12-surface room already has about
/// 2·10⁸ candidate images.
pub const MAX_IMAGE_ORDER: u8 = 8;

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            image_order: 3,
            rays_per_band: 20_000,
            seed: 0,
            receiver_radius_m: 0.5,
            bin_s: 0.001,
            duration_s: None,
            air_absorption: true,
            absorption: Absorption::Table,
        }
    }
}

/// Which stage a run is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Building and checking image sources.
    ImageSources,
    /// Tracing rays.
    RayTracing,
}

/// Progress of a run: `done` of `total` steps of `stage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// The current stage.
    pub stage: Stage,
    /// Steps done.
    pub done: u64,
    /// Steps in the stage.
    pub total: u64,
}

/// How a caller steers a run from another thread.
#[derive(Default)]
pub struct Control<'a> {
    /// Raised by the caller to stop the run; the run returns [`SimError::Cancelled`].
    pub cancel: Option<&'a AtomicBool>,
    /// Wall-clock budget for ray tracing. When it runs out the run stops after the current
    /// round of rays and returns what it has, with [`Simulation::budget_exhausted`] set; the
    /// first round always completes, so a result always has rays in it.
    pub budget: Option<Duration>,
    /// Called on the calling thread between steps.
    pub progress: Option<&'a mut dyn FnMut(Progress)>,
}

impl std::fmt::Debug for Control<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Control")
            .field("cancel", &self.cancel)
            .field("budget", &self.budget)
            .field("progress", &self.progress.is_some())
            .finish()
    }
}

/// A config value out of range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    /// `image_order` above [`MAX_IMAGE_ORDER`].
    ImageOrder,
    /// `receiver_radius_m` not positive and finite.
    ReceiverRadius,
    /// `bin_s` not positive and finite.
    BinWidth,
    /// `duration_s` not positive and finite, or longer than 60 s.
    Duration,
    /// An absorption scale negative or not finite.
    AbsorptionScale,
}

/// Why a run did not produce a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimError {
    /// A config value is out of range.
    Config(ConfigError),
    /// The source is not strictly inside the room.
    SourceOutside,
    /// A receiver is not strictly inside the room.
    ReceiverOutside {
        /// Index into the receivers given.
        index: usize,
    },
    /// The cancel flag was raised.
    Cancelled,
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimError::Config(c) => write!(f, "config out of range: {c:?}"),
            SimError::SourceOutside => write!(f, "the source is not inside the room"),
            SimError::ReceiverOutside { index } => {
                write!(f, "receiver {index} is not inside the room")
            }
            SimError::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for SimError {}

/// What one receiver gets.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReceiverResult {
    /// The receiver's position, m.
    pub at: [f64; 3],
    /// Direct sound and specular reflections up to the image order, sorted by time. A
    /// reflection whose energy is all scattered still appears (with zero energy): which
    /// surface makes an arrival at that time is geometry, whatever its level.
    pub arrivals: Vec<Arrival>,
    /// The rest of the energy, from ray tracing.
    pub late: Echogram,
}

impl ReceiverResult {
    /// The whole echogram: the arrivals binned onto the late part.
    pub fn echogram(&self) -> Echogram {
        let mut e = self.late.clone();
        e.add_arrivals(&self.arrivals);
        e
    }

    /// A synthesised pressure impulse response at `sample_rate`, on the echogram's scale
    /// (see [`synthesize_ir`]). The same seed gives the same samples.
    pub fn impulse_response(&self, sample_rate: f64, seed: u64) -> Result<Vec<f64>, IrError> {
        synthesize_ir(&self.arrivals, &self.late, sample_rate, seed)
    }
}

/// A finished run. Everything in it is simulated, valid above [`Self::valid_above_hz`] only.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Simulation {
    /// One result per receiver, in the order given.
    pub receivers: Vec<ReceiverResult>,
    /// Rays traced per band (fewer than asked when the budget ran out).
    pub rays_per_band: u64,
    /// The time budget ran out before every ray was traced.
    pub budget_exhausted: bool,
    /// Rays that slipped out of the room through rounding at an edge (over all bands);
    /// a few per million is expected, more means a degenerate outline.
    pub lost_rays: u64,
    /// Eyring's T per band for the room as simulated (materials, scale, air), s.
    pub eyring_t_s: PerBand,
    /// Schroeder frequency f_S = 2000·√(T/V) Hz, with T the mean Eyring T of the 500 Hz and
    /// 1 kHz bands. Below it the room's modes are sparse enough that the sound field is a
    /// sum of resonances, not rays, and geometric acoustics does not apply.
    pub schroeder_hz: f64,
    /// The frequency below which no part of this result may be shown as valid. Equal to
    /// [`Self::schroeder_hz`].
    pub valid_above_hz: f64,
    /// The absorption used; anything but [`Absorption::Fitted`] means *absorption assumed*.
    pub absorption: Absorption,
    /// Speed of sound used, m/s.
    pub speed_of_sound: f64,
}

impl Simulation {
    /// Whether a whole band lies above [`Self::valid_above_hz`].
    pub fn band_valid(&self, b: Band) -> bool {
        b.lower_hz() >= self.valid_above_hz
    }
}

/// Rays per work item. Fixed (never derived from the thread count), so the partition of the
/// work, and with it the order of every floating-point sum, is the same on any machine.
const CHUNK: u64 = 1024;
/// Chunks per band per round; a round is the unit of the budget and of progress.
const CHUNKS_PER_ROUND: u64 = 4;

/// Simulate `source` heard at `receivers` in `prism`.
///
/// Ray tracing runs on the current rayon pool (wrap the call in `pool.install` to choose
/// one); everything else, including the progress callback, runs on the calling thread.
pub fn simulate(
    prism: &Prism,
    source: &Source,
    receivers: &[Receiver],
    config: &SimConfig,
    control: &mut Control<'_>,
) -> Result<Simulation, SimError> {
    let started = Instant::now();
    check_config(config)?;
    if !prism.contains(source.at) {
        return Err(SimError::SourceOutside);
    }
    if let Some(index) = receivers.iter().position(|r| !prism.contains(r.at)) {
        return Err(SimError::ReceiverOutside { index });
    }
    let cancel = control.cancel;
    let cancelled = || control_cancelled(cancel);
    let scale = config.absorption.scale();
    let air = prism.air();
    let c = air.speed_of_sound();
    let m = if config.air_absorption {
        air.energy_attenuation()
    } else {
        [0.0; BAND_COUNT]
    };
    let eyring = prism.eyring_t(&scale, config.air_absorption);
    let t_mid = 0.5 * (eyring[Band::Hz500.index()] + eyring[Band::Hz1k.index()]);
    let schroeder_hz = 2000.0 * (t_mid / prism.volume()).sqrt();
    let duration = config.duration_s.unwrap_or_else(|| {
        let longest = eyring.iter().copied().fold(0.0, f64::max);
        (1.2 * longest).clamp(0.1, 10.0)
    });
    let bins = (duration / config.bin_s).ceil() as usize;

    let total_receivers = receivers.len() as u64;
    report(control, Stage::ImageSources, 0, total_receivers);
    let imgs = images(prism, source.at, config.image_order, control.cancel)
        .map_err(|_| SimError::Cancelled)?;
    let mut results = Vec::with_capacity(receivers.len());
    for (i, r) in receivers.iter().enumerate() {
        if cancelled() {
            return Err(SimError::Cancelled);
        }
        results.push(ReceiverResult {
            at: r.at,
            arrivals: arrivals(prism, &imgs, r.at, &scale, &m, c),
            late: Echogram::zeros(config.bin_s, bins),
        });
        report(control, Stage::ImageSources, i as u64 + 1, total_receivers);
    }

    let at: Vec<_> = receivers.iter().map(|r| r.at).collect();
    let ctx = RayCtx {
        prism,
        source: source.at,
        receivers: &at,
        radius: config.receiver_radius_m,
        image_order: config.image_order,
        absorption_scale: scale,
        air: m,
        c,
        bin_s: config.bin_s,
        bins,
        seed: config.seed,
    };
    let chunks = config.rays_per_band.div_ceil(CHUNK);
    let rounds = chunks.div_ceil(CHUNKS_PER_ROUND);
    let mut traced = 0;
    let mut lost = 0;
    let mut budget_exhausted = false;
    report(control, Stage::RayTracing, 0, rounds);
    for round in 0..rounds {
        if round > 0 && control.budget.is_some_and(|b| started.elapsed() >= b) {
            budget_exhausted = true;
            break;
        }
        let first_chunk = round * CHUNKS_PER_ROUND;
        let last_chunk = (first_chunk + CHUNKS_PER_ROUND).min(chunks);
        // Items in a fixed order (band-major within the round); `collect` keeps that order
        // whatever thread ran each.
        let items: Vec<(usize, u64, u64)> = (0..BAND_COUNT)
            .flat_map(|b| {
                (first_chunk..last_chunk).map(move |k| {
                    let first = k * CHUNK;
                    (b, first, CHUNK.min(config.rays_per_band - first))
                })
            })
            .collect();
        let outs: Vec<_> = items
            .par_iter()
            .map(|&(b, first, count)| trace_chunk(&ctx, b, first, count, control.cancel))
            .collect();
        for (&(b, _, _), out) in items.iter().zip(outs) {
            let Some(out) = out else {
                return Err(SimError::Cancelled);
            };
            lost += out.lost;
            for (res, h) in results.iter_mut().zip(out.hist) {
                for (acc, v) in res.late.bands[b].iter_mut().zip(h) {
                    *acc += v;
                }
            }
        }
        traced = (last_chunk * CHUNK).min(config.rays_per_band);
        report(control, Stage::RayTracing, round + 1, rounds);
    }
    if traced > 0 {
        let w = 1.0 / traced as f64;
        for res in &mut results {
            for band in &mut res.late.bands {
                band.iter_mut().for_each(|v| *v *= w);
            }
        }
    }

    Ok(Simulation {
        receivers: results,
        rays_per_band: traced,
        budget_exhausted,
        lost_rays: lost,
        eyring_t_s: eyring,
        schroeder_hz,
        valid_above_hz: schroeder_hz,
        absorption: config.absorption,
        speed_of_sound: c,
    })
}

fn control_cancelled(cancel: Option<&AtomicBool>) -> bool {
    cancel.is_some_and(|c| c.load(Ordering::Relaxed))
}

fn report(control: &mut Control<'_>, stage: Stage, done: u64, total: u64) {
    if let Some(p) = control.progress.as_mut() {
        p(Progress { stage, done, total });
    }
}

fn check_config(c: &SimConfig) -> Result<(), SimError> {
    let err = |e| Err(SimError::Config(e));
    if c.image_order > MAX_IMAGE_ORDER {
        return err(ConfigError::ImageOrder);
    }
    if !(c.receiver_radius_m.is_finite() && c.receiver_radius_m > 0.0) {
        return err(ConfigError::ReceiverRadius);
    }
    if !(c.bin_s.is_finite() && c.bin_s > 0.0) {
        return err(ConfigError::BinWidth);
    }
    if let Some(d) = c.duration_s
        && !(d.is_finite() && d > 0.0 && d <= 60.0)
    {
        return err(ConfigError::Duration);
    }
    if c.absorption
        .scale()
        .iter()
        .any(|s| !(s.is_finite() && *s >= 0.0))
    {
        return err(ConfigError::AbsorptionScale);
    }
    Ok(())
}
