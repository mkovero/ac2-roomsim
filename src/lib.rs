//! Geometric room acoustics for prism-shaped rooms: a simple floor polygon extruded to a
//! flat ceiling, the shape of halls, fan-shaped auditoria, L-shaped bars and wedges.
//!
//! Per octave band (125 Hz–8 kHz):
//! - **image sources** up to a chosen order, with Borish's validity and visibility tests so
//!   concave outlines hide what they should ([`Arrival`]: time, energy, which surfaces);
//! - **stochastic ray tracing** with Lambert scattering for the late tail, detected in
//!   receiver spheres ([`Echogram`]), deterministic per seed on any thread count;
//! - **air absorption** after ISO 9613-1 ([`Air`]);
//! - Schroeder decay fits and an **impulse-response synthesis** from the echogram.
//!
//! Geometric acoustics treats sound as rays, which holds only where wavelengths are small
//! against the room and its modes overlap: above the Schroeder frequency
//! ([`Simulation::schroeder_hz`]). Absorption is taken from a table and is an assumption
//! until scaled to fit a measurement ([`Absorption`]).
//!
//! ```
//! use ac2_roomsim::*;
//!
//! let room = Room {
//!     outline: vec![[0.0, 0.0], [12.0, 0.0], [12.0, 8.0], [0.0, 8.0]],
//!     height: Some(4.0),
//!     surfaces: Surfaces::uniform(4, MaterialId::PlasterOnMasonry.into()),
//!     air: Air::default(),
//! };
//! let prism = Prism::new(&room)?;
//! let config = SimConfig { rays_per_band: 2_000, ..SimConfig::default() };
//! let sim = simulate(
//!     &prism,
//!     &Source { at: [2.0, 4.0, 1.5] },
//!     &[Receiver { at: [9.0, 3.0, 1.2] }],
//!     &config,
//!     &mut Control::default(),
//! )?;
//! let first = &sim.receivers[0].arrivals[1];
//! println!("first reflection after {:.1} ms from {:?}", first.time_s * 1e3, first.path);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The library does no I/O and holds no global state; ray tracing runs on the caller's rayon
//! pool.

mod air;
mod band;
mod echogram;
mod image;
mod material;
mod ray;
mod rng;
mod room;
mod sim;
mod vec3;

pub use air::Air;
pub use band::{BAND_COUNT, Band, PerBand};
pub use echogram::{Echogram, IrError, synthesize_ir};
pub use image::Arrival;
pub use material::{Coefficients, Material, MaterialId};
pub use room::{Prism, Room, RoomError, SurfaceId, Surfaces};
pub use sim::{
    Absorption, ConfigError, Control, MAX_IMAGE_ORDER, Progress, Receiver, ReceiverResult,
    SimConfig, SimError, Simulation, Source, Stage, simulate,
};
