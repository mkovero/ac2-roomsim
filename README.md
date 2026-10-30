# ac2-roomsim

Geometric room acoustics for prism-shaped rooms: a simple floor polygon extruded to a flat
ceiling, which covers halls, fan-shaped auditoria, L-shaped bars and wedges. Per octave band
(125 Hz–8 kHz) it computes

- **image sources** up to a chosen order, with the validity and visibility tests of Borish
  (1984), so a concave outline hides the reflections it should. Each arrival carries its
  time, its energy per band and the surfaces it reflected from;
- **stochastic ray tracing** with Lambert scattering for the late tail, detected in receiver
  spheres as a per-band echogram, deterministic per seed whatever the thread count (rayon);
- **air absorption** after ISO 9613-1;
- Schroeder decay curves and T20/T30 fits, and an **impulse response** synthesised from
  the echogram for analysis by tools that work on measured IRs.

```rust
use ac2_roomsim::*;

let room = Room {
    outline: vec![[0.0, 0.0], [12.0, 0.0], [12.0, 8.0], [0.0, 8.0]],
    height: Some(4.0),
    surfaces: Surfaces::uniform(4, MaterialId::PlasterOnMasonry.into()),
    air: Air::default(),
};
let prism = Prism::new(&room)?; // simple, counter-clockwise, has a ceiling
let sim = simulate(
    &prism,
    &Source { at: [2.0, 4.0, 1.5] },
    &[Receiver { at: [9.0, 3.0, 1.2] }],
    &SimConfig::default(),
    &mut Control::default(), // cancel flag, time budget, progress callback
)?;
for a in &sim.receivers[0].arrivals[..4] {
    println!("{:6.1} ms  {:?}", a.time_s * 1e3, a.path);
}
let t30 = sim.receivers[0].echogram().t30(Band::Hz1k);
let ir = sim.receivers[0].impulse_response(48_000.0, 1)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## What it can claim

- **Above the Schroeder frequency only.** `Simulation::schroeder_hz` is
  f_S = 2000·√(T/V) Hz (T from Eyring at 500 Hz–1 kHz, V the volume) and `valid_above_hz`
  is the limit below which nothing may be shown as valid. Below it a room is a set of
  sparse modes and rays are the wrong model.
- **Absorption is assumed** unless fitted. The built-in material table holds typical
  published coefficients (sources in `src/material.rs`); a real room's surfaces differ.
  `Absorption::Fitted { scale }` records one scale per band chosen so the simulated decay
  matches a measured one. One scale per band is identifiable from measurement; per-surface
  absorption is not, so after a fit the geometry of a reflection (which wall, when) holds
  but its level is still an assumption.
- Omnidirectional sources, flat floor and ceiling, no objects inside the room, no diffraction.

## Accuracy, as tested

- Image sources in a shoebox reproduce the analytic image lattice to order 4 exactly
  (count, distances, energies).
- Visibility in an L-shaped room matches hand-worked cases (hidden direct sound, a leg
  missing the finite wall, a leg through the cut-away corner, a reflection round it).
- With fully diffuse reflections and uniform absorption α, the ray-traced T30 lies within
  1.5 / 3 / 4 % of Eyring for α = 0.05 / 0.1 / 0.15, the excess that free-path variance
  predicts; with Kuttruff's path-variance correction (γ² = 0.4) it is within 1 % up to
  α = 0.2 and 2.5 % at α = 0.4.
- Air absorption agrees with the ISO 9613-2 table (computed from ISO 9613-1) to its
  rounding.

## Using it from another program

The library does no I/O and holds no global state. `simulate` runs ray tracing on the
current rayon pool (wrap the call in `ThreadPool::install` to choose one), checks a cancel
flag, stops after the current round when its time budget runs out (returning what it has,
marked `budget_exhausted`), and reports progress on the calling thread. The `serde`
feature derives `Serialize`/`Deserialize` on the input and output types.

[ac2](https://github.com/mkovero/ac2), a live dual-channel analyzer for PA tuning, runs it
on a daemon worker thread for a room plan: predicted reflection arrivals over a measured
energy-time curve (which wall made that 7 ms reflection), a direct-plus-early energy map
over audience zones per speaker, and simulated decay fitted to the T20 measured on a walk
through the room. ac2 converts its own plan types into `Room`; this crate depends on no
ac2 crate.

## Tests

```
cargo nextest run              # quick tier: skips tests named slow_…
cargo nextest run --profile full
cargo test --doc
```

## Requirements

Rust 1.95 or later (edition 2024). No `unsafe` (`forbid(unsafe_code)`).

## License

MIT, see [LICENSE](LICENSE).
