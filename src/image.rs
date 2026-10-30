//! Image sources (Allen & Berkley 1979) with the validity and visibility tests of Borish
//! (1984), which make the method correct for any polyhedral room, not just a shoebox.
//!
//! A specular reflection in a plane looks, from the receiver, like sound from the source's
//! mirror image behind that plane. Reflecting images in further planes gives every specular
//! path up to a given order. In a convex room every image whose parent lies in front of the
//! mirroring plane is seen by every receiver; in a concave room (an L-shaped bar, a hall with
//! a stage house) the path to an image may miss the finite face it reflects from, or pass
//! through a wall, so each image is checked per receiver by walking its path back from the
//! receiver to the source.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::band::{BAND_COUNT, PerBand};
use crate::room::{Prism, SurfaceId};
use crate::vec3::{V3, add_scaled, norm, sub};

/// One specular arrival at a receiver: the direct sound (empty path) or a reflection.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Arrival {
    /// Arrival time after the source emits, s.
    pub time_s: f64,
    /// Path length, m (the distance from the receiver to the image).
    pub distance_m: f64,
    /// Energy per band, relative to the direct sound at 1 m in free field without air
    /// absorption (so the direct sound at distance d carries 1/d², less the air).
    pub energy: PerBand,
    /// The surfaces the path reflects from, in order from the source.
    pub path: Vec<SurfaceId>,
    /// Where the image source lies, m.
    pub image: [f64; 3],
}

/// An image source; `parent` is an index into the same list, `surface` the plane it was
/// mirrored in. Entry 0 is the source itself.
#[derive(Debug, Clone)]
pub(crate) struct Image {
    pos: V3,
    parent: usize,
    surface: usize,
    order: u8,
}

/// The cancel flag was raised.
#[derive(Debug)]
pub(crate) struct Cancelled;

/// Every valid image of `source` up to `max_order` reflections.
///
/// An image is valid (Borish) only if its parent lies strictly in front of the mirroring
/// plane: sound from a parent behind a plane can never reach that plane's front face, so the
/// image and its whole subtree are dropped. Mirroring twice in the same plane in a row returns
/// to the parent and is skipped.
pub(crate) fn images(
    prism: &Prism,
    source: V3,
    max_order: u8,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<Image>, Cancelled> {
    let mut out = vec![Image {
        pos: source,
        parent: usize::MAX,
        surface: usize::MAX,
        order: 0,
    }];
    let mut start = 0;
    for order in 1..=max_order {
        let end = out.len();
        for parent in start..end {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err(Cancelled);
            }
            let p = out[parent].pos;
            for s in 0..prism.surface_count() {
                if s == out[parent].surface {
                    continue;
                }
                let pl = prism.plane(s);
                if pl.distance(p) <= 1e-9 {
                    continue;
                }
                out.push(Image {
                    pos: pl.mirror(p),
                    parent,
                    surface: s,
                    order,
                });
            }
        }
        start = end;
    }
    Ok(out)
}

/// The images `receiver` sees, as arrivals sorted by time.
///
/// `absorption_scale` multiplies every absorption coefficient per band; `air` is the energy
/// attenuation per metre per band; `c` the speed of sound. A reflection passes on the share
/// (1 − α)(1 − s) of the energy: the absorbed share α is lost and the scattered share s leaves
/// the specular path (the ray tracer carries it).
pub(crate) fn arrivals(
    prism: &Prism,
    images: &[Image],
    receiver: V3,
    absorption_scale: &PerBand,
    air: &PerBand,
    c: f64,
) -> Vec<Arrival> {
    let source = images[0].pos;
    let mut out = Vec::new();
    let mut chain = Vec::new();
    'image: for (idx, img) in images.iter().enumerate() {
        chain.clear();
        let mut cur = receiver;
        let mut cur_surface: Option<usize> = None;
        let mut node = idx;
        while node != 0 {
            let im = &images[node];
            let pl = prism.plane(im.surface);
            let da = pl.distance(cur);
            let di = pl.distance(im.pos);
            // The leg toward the image must cross the plane from the front.
            if !(da > 0.0 && di < 0.0) {
                continue 'image;
            }
            let q = add_scaled(cur, da / (da - di), sub(im.pos, cur));
            if !prism.on_face(im.surface, q) {
                continue 'image;
            }
            if prism.blocked(cur, q, [cur_surface, Some(im.surface)]) {
                continue 'image;
            }
            chain.push(im.surface);
            cur = q;
            cur_surface = Some(im.surface);
            node = im.parent;
        }
        if prism.blocked(cur, source, [cur_surface, None]) {
            continue;
        }
        let d = norm(sub(receiver, img.pos));
        if d <= 0.0 {
            continue;
        }
        let mut energy = [0.0; BAND_COUNT];
        for (b, e) in energy.iter_mut().enumerate() {
            let mut v = (-air[b] * d).exp() / (d * d);
            for &s in &chain {
                let co = prism.coefficients_at(s);
                v *= (1.0 - (co.absorption[b] * absorption_scale[b]).min(1.0))
                    * (1.0 - co.scattering[b]);
            }
            *e = v;
        }
        debug_assert_eq!(chain.len(), img.order as usize);
        out.push(Arrival {
            time_s: d / c,
            distance_m: d,
            energy,
            path: chain.iter().rev().map(|&s| prism.id(s)).collect(),
            image: img.pos,
        });
    }
    out.sort_by(|a, b| a.time_s.total_cmp(&b.time_s));
    out
}
