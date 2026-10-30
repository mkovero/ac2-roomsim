//! The room: a simple floor polygon extruded to a flat ceiling (a prism), its surfaces'
//! materials and its air. [`Room`] is the input; [`Prism`] is a validated room with the
//! planes and areas the simulation works on.

use crate::air::Air;
use crate::band::{BAND_COUNT, PerBand};
use crate::material::{Coefficients, Material};
use crate::vec3::{V3, dot, sub};

/// A room as given: floor outline, ceiling height, materials and air. Metres throughout; plan
/// x/y in the floor plane, z up from the floor.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Room {
    /// Floor outline: a simple polygon, counter-clockwise, at least 3 vertices. Edge `i` runs
    /// from vertex `i` to vertex `i + 1` (the last back to the first); wall `i` stands on it.
    pub outline: Vec<[f64; 2]>,
    /// Ceiling height above the floor. `None`: outdoors, which this crate refuses (with no
    /// ceiling there is no reverberant field for geometric acoustics to describe).
    pub height: Option<f64>,
    /// The material of every surface.
    pub surfaces: Surfaces,
    /// The air in the room.
    pub air: Air,
}

/// One material per surface of the prism.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Surfaces {
    /// One per outline edge, in edge order.
    pub walls: Vec<Material>,
    /// The floor.
    pub floor: Material,
    /// The ceiling.
    pub ceiling: Material,
}

impl Surfaces {
    /// Every surface of an `edges`-sided room in the same material.
    pub fn uniform(edges: usize, material: Material) -> Surfaces {
        Surfaces {
            walls: vec![material; edges],
            floor: material,
            ceiling: material,
        }
    }
}

/// One surface of the prism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum SurfaceId {
    /// The wall on outline edge `edge` (vertex `edge` to `edge + 1`).
    Wall {
        /// Outline edge index.
        edge: u32,
    },
    /// The floor (z = 0).
    Floor,
    /// The ceiling (z = height).
    Ceiling,
}

/// Why a room cannot be simulated.
#[derive(Debug, Clone, PartialEq)]
pub enum RoomError {
    /// The outline has fewer than 3 vertices.
    TooFewVertices {
        /// Vertices given.
        count: usize,
    },
    /// A coordinate, the height or an air value is NaN or infinite.
    NonFinite,
    /// An edge has (near) zero length.
    DegenerateEdge {
        /// Edge index.
        edge: usize,
    },
    /// The outline doubles back on itself at this vertex (the two edges meeting there are
    /// collinear and opposite).
    FoldedVertex {
        /// Vertex index.
        vertex: usize,
    },
    /// Two edges cross or touch, so the outline is not a simple polygon.
    EdgesCross {
        /// The lower edge index.
        a: usize,
        /// The higher edge index.
        b: usize,
    },
    /// The outline runs clockwise; it must run counter-clockwise (interior on the left of
    /// every edge, which fixes each wall's inward normal).
    Clockwise,
    /// No ceiling: an outdoor plan.
    Outdoor,
    /// The ceiling height is not positive.
    BadHeight,
    /// The wall material list does not have one entry per outline edge.
    WallCount {
        /// Wall materials given.
        walls: usize,
        /// Outline edges.
        edges: usize,
    },
    /// A material coefficient lies outside `[0, 1]`.
    BadCoefficient {
        /// The surface.
        surface: SurfaceId,
    },
    /// The air is outside the range ISO 9613-1 covers (−20…50 °C, 0…100 % RH,
    /// 50…120 kPa).
    BadAir,
}

impl std::fmt::Display for RoomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoomError::TooFewVertices { count } => {
                write!(f, "the outline has {count} vertices, at least 3 needed")
            }
            RoomError::NonFinite => write!(f, "a coordinate or value is not a finite number"),
            RoomError::DegenerateEdge { edge } => write!(f, "edge {edge} has zero length"),
            RoomError::FoldedVertex { vertex } => {
                write!(f, "the outline folds back on itself at vertex {vertex}")
            }
            RoomError::EdgesCross { a, b } => write!(f, "edge {a} crosses edge {b}"),
            RoomError::Clockwise => write!(f, "the outline runs clockwise"),
            RoomError::Outdoor => write!(f, "no ceiling: outdoor plans are not simulated"),
            RoomError::BadHeight => write!(f, "the ceiling height must be positive"),
            RoomError::WallCount { walls, edges } => {
                write!(f, "{walls} wall materials for {edges} outline edges")
            }
            RoomError::BadCoefficient { surface } => {
                write!(f, "a coefficient of {surface:?} is outside 0…1")
            }
            RoomError::BadAir => write!(f, "air outside −20…50 °C, 0…100 % RH, 50…120 kPa"),
        }
    }
}

impl std::error::Error for RoomError {}

/// A plane `n·p = d` with unit normal `n` pointing into the room.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Plane {
    pub n: V3,
    pub d: f64,
}

impl Plane {
    /// Signed distance of `p` from the plane, positive on the room side.
    pub fn distance(&self, p: V3) -> f64 {
        dot(self.n, p) - self.d
    }

    /// The mirror image of `p` in the plane.
    pub fn mirror(&self, p: V3) -> V3 {
        let s = 2.0 * self.distance(p);
        [
            p[0] - s * self.n[0],
            p[1] - s * self.n[1],
            p[2] - s * self.n[2],
        ]
    }
}

/// Edges shorter than this are degenerate. A millimetre is far below any wall that matters
/// acoustically and far above floating-point noise on room-sized coordinates.
const MIN_EDGE_M: f64 = 1e-3;
/// Slack when deciding whether a point on a surface's plane lies on the face itself. Rays that
/// hit within a micrometre of a corner would otherwise slip between the two faces meeting
/// there through rounding.
const FACE_TOL: f64 = 1e-9;

/// A validated room: the prism, each surface's plane, area and coefficients.
#[derive(Debug, Clone)]
pub struct Prism {
    outline: Vec<[f64; 2]>,
    height: f64,
    /// Walls `0..n`, then floor, then ceiling.
    planes: Vec<Plane>,
    coefficients: Vec<Coefficients>,
    areas: Vec<f64>,
    floor_area: f64,
    air: Air,
}

impl Prism {
    /// Validate `room`: a simple counter-clockwise outline without degenerate edges, a
    /// positive ceiling height, one material per wall, coefficients in `[0, 1]` and air in the
    /// range of ISO 9613-1.
    pub fn new(room: &Room) -> Result<Prism, RoomError> {
        let pts = &room.outline;
        let n = pts.len();
        if n < 3 {
            return Err(RoomError::TooFewVertices { count: n });
        }
        if pts.iter().flatten().any(|v| !v.is_finite()) {
            return Err(RoomError::NonFinite);
        }
        let height = room.height.ok_or(RoomError::Outdoor)?;
        if !height.is_finite() {
            return Err(RoomError::NonFinite);
        }
        if height <= 0.0 {
            return Err(RoomError::BadHeight);
        }
        let edge = |i: usize| (pts[i], pts[(i + 1) % n]);
        for i in 0..n {
            let (a, b) = edge(i);
            if ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() < MIN_EDGE_M {
                return Err(RoomError::DegenerateEdge { edge: i });
            }
        }
        for v in 0..n {
            let p = pts[(v + n - 1) % n];
            let q = pts[v];
            let r = pts[(v + 1) % n];
            let (e1, e2) = ([q[0] - p[0], q[1] - p[1]], [r[0] - q[0], r[1] - q[1]]);
            let cross = e1[0] * e2[1] - e1[1] * e2[0];
            let dotp = e1[0] * e2[0] + e1[1] * e2[1];
            let scale = (e1[0].hypot(e1[1])) * (e2[0].hypot(e2[1]));
            if cross.abs() <= 1e-12 * scale && dotp < 0.0 {
                return Err(RoomError::FoldedVertex { vertex: v });
            }
        }
        for a in 0..n {
            for b in a + 1..n {
                // Adjacent edges share a vertex by construction; a fold between them is caught
                // above.
                if b == a + 1 || (a == 0 && b == n - 1) {
                    continue;
                }
                let (p1, p2) = edge(a);
                let (q1, q2) = edge(b);
                if segments_touch(p1, p2, q1, q2) {
                    return Err(RoomError::EdgesCross { a, b });
                }
            }
        }
        let signed = signed_area(pts);
        if signed < 0.0 {
            return Err(RoomError::Clockwise);
        }
        if room.surfaces.walls.len() != n {
            return Err(RoomError::WallCount {
                walls: room.surfaces.walls.len(),
                edges: n,
            });
        }
        let air = room.air;
        if ![air.temperature_c, air.humidity_pct, air.pressure_kpa]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(RoomError::NonFinite);
        }
        if !(-20.0..=50.0).contains(&air.temperature_c)
            || !(0.0..=100.0).contains(&air.humidity_pct)
            || !(50.0..=120.0).contains(&air.pressure_kpa)
        {
            return Err(RoomError::BadAir);
        }

        let mut planes = Vec::with_capacity(n + 2);
        let mut areas = Vec::with_capacity(n + 2);
        for i in 0..n {
            let (a, b) = edge(i);
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let len = ex.hypot(ey);
            // Counter-clockwise: the interior lies to the left of each edge.
            let nrm = [-ey / len, ex / len, 0.0];
            planes.push(Plane {
                n: nrm,
                d: nrm[0] * a[0] + nrm[1] * a[1],
            });
            areas.push(len * height);
        }
        planes.push(Plane {
            n: [0.0, 0.0, 1.0],
            d: 0.0,
        });
        planes.push(Plane {
            n: [0.0, 0.0, -1.0],
            d: -height,
        });
        areas.push(signed);
        areas.push(signed);

        let materials = room
            .surfaces
            .walls
            .iter()
            .chain([&room.surfaces.floor, &room.surfaces.ceiling]);
        let mut coefficients = Vec::with_capacity(n + 2);
        for (i, m) in materials.enumerate() {
            let c = m.coefficients();
            let ok = c
                .absorption
                .iter()
                .chain(c.scattering.iter())
                .all(|v| (0.0..=1.0).contains(v));
            if !ok {
                return Err(RoomError::BadCoefficient {
                    surface: surface_id(i, n),
                });
            }
            coefficients.push(c);
        }

        Ok(Prism {
            outline: pts.clone(),
            height,
            planes,
            coefficients,
            areas,
            floor_area: signed,
            air,
        })
    }

    /// The floor outline.
    pub fn outline(&self) -> &[[f64; 2]] {
        &self.outline
    }

    /// Ceiling height, m.
    pub fn height(&self) -> f64 {
        self.height
    }

    /// The air.
    pub fn air(&self) -> &Air {
        &self.air
    }

    /// Volume, m³.
    pub fn volume(&self) -> f64 {
        self.floor_area * self.height
    }

    /// Total surface area, m².
    pub fn surface_area(&self) -> f64 {
        self.areas.iter().sum()
    }

    /// Number of surfaces: one wall per edge, the floor and the ceiling.
    pub fn surface_count(&self) -> usize {
        self.planes.len()
    }

    /// Every surface, walls in edge order, then floor, then ceiling.
    pub fn surfaces(&self) -> impl Iterator<Item = SurfaceId> + '_ {
        (0..self.planes.len()).map(|i| surface_id(i, self.outline.len()))
    }

    /// Area of one surface, m².
    pub fn area(&self, s: SurfaceId) -> f64 {
        self.areas[self.index(s)]
    }

    /// Coefficients of one surface.
    pub fn coefficients(&self, s: SurfaceId) -> Coefficients {
        self.coefficients[self.index(s)]
    }

    /// Whether `p` lies strictly inside the room.
    pub fn contains(&self, p: [f64; 3]) -> bool {
        p.iter().all(|v| v.is_finite())
            && p[2] > 0.0
            && p[2] < self.height
            && point_in_polygon(&self.outline, [p[0], p[1]])
    }

    /// Area-weighted mean absorption per band, after `scale` (each band's coefficients
    /// multiplied by its scale and clipped to 1).
    pub fn mean_absorption(&self, scale: &PerBand) -> PerBand {
        let mut out = [0.0; BAND_COUNT];
        let total = self.surface_area();
        for (c, a) in self.coefficients.iter().zip(&self.areas) {
            for b in 0..BAND_COUNT {
                out[b] += a * (c.absorption[b] * scale[b]).min(1.0);
            }
        }
        out.map(|v| v / total)
    }

    /// Eyring's reverberation time per band, s:
    /// T = 24·ln(10)·V / (c·(−S·ln(1 − ᾱ) + 4·m·V)),
    /// the decay of a diffuse field whose rays meet a wall every mean free path 4V/S and lose
    /// the share ᾱ there, plus the air's energy attenuation `m` along the way. With
    /// `air_absorption` false, m = 0.
    pub fn eyring_t(&self, scale: &PerBand, air_absorption: bool) -> PerBand {
        let alpha = self.mean_absorption(scale);
        let m = self.air.energy_attenuation();
        let c = self.air.speed_of_sound();
        let (v, s) = (self.volume(), self.surface_area());
        let mut t = [0.0; BAND_COUNT];
        for b in 0..BAND_COUNT {
            let air = if air_absorption { 4.0 * m[b] * v } else { 0.0 };
            let surf = -s * (1.0 - alpha[b]).max(f64::MIN_POSITIVE).ln();
            t[b] = 24.0 * std::f64::consts::LN_10 * v / (c * (surf + air));
        }
        t
    }

    pub(crate) fn plane(&self, i: usize) -> &Plane {
        &self.planes[i]
    }

    pub(crate) fn coefficients_at(&self, i: usize) -> &Coefficients {
        &self.coefficients[i]
    }

    pub(crate) fn id(&self, i: usize) -> SurfaceId {
        surface_id(i, self.outline.len())
    }

    fn index(&self, s: SurfaceId) -> usize {
        let n = self.outline.len();
        match s {
            SurfaceId::Wall { edge } => {
                assert!((edge as usize) < n, "wall {edge} of a {n}-edge room");
                edge as usize
            }
            SurfaceId::Floor => n,
            SurfaceId::Ceiling => n + 1,
        }
    }

    /// Whether `p`, a point on surface `i`'s plane, lies on its face.
    pub(crate) fn on_face(&self, i: usize, p: V3) -> bool {
        let n = self.outline.len();
        if i < n {
            let a = self.outline[i];
            let b = self.outline[(i + 1) % n];
            let e = [b[0] - a[0], b[1] - a[1]];
            let u = ((p[0] - a[0]) * e[0] + (p[1] - a[1]) * e[1]) / (e[0] * e[0] + e[1] * e[1]);
            (-FACE_TOL..=1.0 + FACE_TOL).contains(&u)
                && (-FACE_TOL..=self.height + FACE_TOL).contains(&p[2])
        } else {
            point_in_polygon(&self.outline, [p[0], p[1]])
        }
    }

    /// The first surface a ray from `o` along unit `dir` meets, and the distance to it.
    /// `skip` is the surface the ray leaves from: a plane cannot be met again by a ray leaving
    /// it, and skipping it keeps rounding from reporting a hit at distance zero.
    pub(crate) fn first_hit(&self, o: V3, dir: V3, skip: Option<usize>) -> Option<(f64, usize)> {
        let mut best: Option<(f64, usize)> = None;
        for (i, pl) in self.planes.iter().enumerate() {
            if Some(i) == skip {
                continue;
            }
            let dn = dot(pl.n, dir);
            // Only a surface the ray approaches from the room side can stop it.
            if dn >= 0.0 {
                continue;
            }
            let t = -pl.distance(o) / dn;
            if t <= 0.0 || best.is_some_and(|(bt, _)| t >= bt) {
                continue;
            }
            let p = [o[0] + t * dir[0], o[1] + t * dir[1], o[2] + t * dir[2]];
            if self.on_face(i, p) {
                best = Some((t, i));
            }
        }
        best
    }

    /// Whether the segment `a`–`b` (both in or on the room) leaves the room, i.e. crosses a
    /// wall face strictly between its ends. Only walls can block: floor and ceiling are
    /// horizontal planes the segment's ends already lie between. The surfaces `a` and `b`
    /// lie on are left out so a path's own reflection points never block it.
    pub(crate) fn blocked(&self, a: V3, b: V3, ends: [Option<usize>; 2]) -> bool {
        let d = sub(b, a);
        for i in 0..self.outline.len() {
            if ends.contains(&Some(i)) {
                continue;
            }
            let pl = &self.planes[i];
            let da = pl.distance(a);
            let db = pl.distance(b);
            // Same side (or touching): no crossing of this plane.
            if (da > 0.0) == (db > 0.0) || da == db {
                continue;
            }
            let t = da / (da - db);
            if t <= 1e-9 || t >= 1.0 - 1e-9 {
                continue;
            }
            let p = [a[0] + t * d[0], a[1] + t * d[1], a[2] + t * d[2]];
            if self.on_face(i, p) {
                return true;
            }
        }
        false
    }
}

pub(crate) fn surface_id(i: usize, edges: usize) -> SurfaceId {
    if i < edges {
        SurfaceId::Wall { edge: i as u32 }
    } else if i == edges {
        SurfaceId::Floor
    } else {
        SurfaceId::Ceiling
    }
}

fn signed_area(pts: &[[f64; 2]]) -> f64 {
    let n = pts.len();
    (0..n)
        .map(|i| {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
        / 2.0
}

/// Crossing-number test; points exactly on an edge may fall either way, which is harmless
/// here: callers only ask about points inside the room or on its floor and ceiling faces.
fn point_in_polygon(pts: &[[f64; 2]], p: [f64; 2]) -> bool {
    let n = pts.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (pts[i], pts[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if p[0] < x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn on_segment(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> bool {
    p[0] >= a[0].min(b[0])
        && p[0] <= a[0].max(b[0])
        && p[1] >= a[1].min(b[1])
        && p[1] <= a[1].max(b[1])
}

/// Whether two closed segments share a point (cross, touch or overlap).
fn segments_touch(p1: [f64; 2], p2: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> bool {
    let d1 = orient(q1, q2, p1);
    let d2 = orient(q1, q2, p2);
    let d3 = orient(p1, p2, q1);
    let d4 = orient(p1, p2, q2);
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }
    (d1 == 0.0 && on_segment(q1, q2, p1))
        || (d2 == 0.0 && on_segment(q1, q2, p2))
        || (d3 == 0.0 && on_segment(p1, p2, q1))
        || (d4 == 0.0 && on_segment(p1, p2, q2))
}
