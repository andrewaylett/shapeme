//! Gene-level traits and types for shapeme's genetic representation.
//!
//! A `Gene` is a single mutable unit of genetic information. `ShapeGene` is an enum
//! over three concrete types — `TriangleGene`, `CircleGene`, `PolygonGene` — each
//! owning its fields directly and implementing `Gene` independently. This eliminates
//! the shape-type match within mutation code: each struct's `Gene::mutate` knows
//! exactly which fields it has. `BlurGene` encodes the optional global Gaussian blur
//! radius.
//!
//! # Why explicit z-ordering?
//!
//! Previously the draw order was determined by position in a `Vec`. Recombination via
//! single-point crossover would silently reorder shapes if the two parents had different
//! lengths. By tagging each gene with a `z_order` key we can sort consistently after
//! crossover, and swap/nudge z-values as a distinct mutation step.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::shapes::{
    MINALPHA, MAXALPHA, ShapeKind, clamp_coord, nudge_oklab, rand_between, random_oklab_color,
    select_shape_type,
};

/// Approximate bincode byte cost of a `ShapeGene` wrapping a Triangle.
///
/// Used as the base budget unit: `--max-shapes N` → `max_cost = N × TRIANGLE_COST`.
/// Colours are stored as [f32; 3] (12 bytes) in `OKlab`.  Bincode v2 uses varint encoding
/// for integer fields, so actual sizes vary with coordinate magnitude; these constants
/// approximate the typical encoding for coordinates in a 256×256 image.
pub const TRIANGLE_COST: usize = 21;
/// Approximate bincode byte cost of a `ShapeGene` wrapping a Circle.
pub const CIRCLE_COST: usize = 18;
/// Approximate bincode byte cost of a `ShapeGene` wrapping a Polygon, excluding per-vertex cost.
pub const POLYGON_BASE_COST: usize = 16;
/// Additional cost per polygon vertex (one (i16, i16) pair ≈ 4 bytes for typical coordinates).
pub const POLYGON_VERTEX_COST: usize = 4;

/// Parameters needed by gene-level mutation operations.
///
/// Passed by the genome to each `Gene::mutate` call.  `BlurGene` ignores spatial fields;
/// shape kind flags are ignored by `BlurGene` and used by `ShapeGene`.
#[derive(Clone, Debug)]
pub struct MutationConfig {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Blur-derived margin: coords in `[-margin, w+margin)` are valid.
    pub margin: i16,
    /// Mutation probability per shape per generation (0–1000).
    pub mutation_rate: u32,
    /// Whether triangle shapes are in use for this run.
    pub use_triangles: bool,
    /// Whether circle shapes are in use for this run.
    pub use_circles: bool,
    /// Whether polygon shapes are in use for this run.
    pub use_polygons: bool,
    /// Maximum number of vertices a single Polygon gene may grow to via `split_edge`.
    pub max_polygon_vertices: usize,
}

/// A single mutable, recombineable unit of genetic information.
///
/// Not object-safe by design — `recombine` takes `&Self` so implementations can use
/// type-specific blending without boxing.
pub trait Gene: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> {
    /// Return a mutated copy of `self`.  Never mutates in place; the caller decides acceptance.
    #[must_use]
    fn mutate(&self, rng: &mut impl Rng, config: &MutationConfig) -> Self;
    /// Produce an offspring by combining genetic material from `self` and `other`.
    #[must_use]
    fn recombine(&self, other: &Self, rng: &mut impl Rng) -> Self;
}

/// Filled triangle with vertices sorted by y-coordinate for the rasteriser.
///
/// The rasteriser requires y1 ≤ y2 ≤ y3; `normalize` maintains this invariant
/// via a bubble-sort of (x, y) vertex pairs, identical to the original C algorithm.
/// Alpha is stored as an integer percentage (1–100) divided by 100.0 on render.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriangleGene {
    /// First vertex x coordinate.
    pub x1: i16,
    /// First vertex y coordinate (smallest y after normalisation).
    pub y1: i16,
    /// Second vertex x coordinate.
    pub x2: i16,
    /// Second vertex y coordinate.
    pub y2: i16,
    /// Third vertex x coordinate.
    pub x3: i16,
    /// Third vertex y coordinate (largest y after normalisation).
    pub y3: i16,
    /// Colour in `OKlab` [L, a, b].
    pub oklab: [f32; 3],
    /// Alpha as an integer percentage (1–100).
    pub alpha: u8,
    /// Draw-order key; lower values are drawn first (behind higher-valued genes).
    pub z_order: u16,
}

impl TriangleGene {
    fn normalize(&mut self, width: u32, height: u32, margin: i16) {
        let w = width as i16;
        let h = height as i16;
        loop {
            let mut swapped = false;
            if self.y1 > self.y2 {
                std::mem::swap(&mut self.y1, &mut self.y2);
                std::mem::swap(&mut self.x1, &mut self.x2);
                swapped = true;
            }
            if self.y2 > self.y3 {
                std::mem::swap(&mut self.y2, &mut self.y3);
                std::mem::swap(&mut self.x2, &mut self.x3);
                swapped = true;
            }
            if !swapped {
                break;
            }
        }
        self.x1 = clamp_coord(self.x1, -margin, w - 1 + margin);
        self.y1 = clamp_coord(self.y1, -margin, h - 1 + margin);
        self.x2 = clamp_coord(self.x2, -margin, w - 1 + margin);
        self.y2 = clamp_coord(self.y2, -margin, h - 1 + margin);
        self.x3 = clamp_coord(self.x3, -margin, w - 1 + margin);
        self.y3 = clamp_coord(self.y3, -margin, h - 1 + margin);
    }
}

impl Gene for TriangleGene {
    fn mutate(&self, rng: &mut impl Rng, config: &MutationConfig) -> Self {
        let mut g = self.clone();
        match rng.random_range(0..6u32) {
            0 => {
                g.x1 = (rng.random::<u32>() % config.width) as i16;
                g.y1 = (rng.random::<u32>() % config.height) as i16;
                g.x2 = (rng.random::<u32>() % config.width) as i16;
                g.y2 = (rng.random::<u32>() % config.height) as i16;
                g.x3 = (rng.random::<u32>() % config.width) as i16;
                g.y3 = (rng.random::<u32>() % config.height) as i16;
                g.normalize(config.width, config.height, config.margin);
            }
            1 => {
                g.x1 = g.x1.saturating_add(rand_between(rng, -20, 20) as i16);
                g.y1 = g.y1.saturating_add(rand_between(rng, -20, 20) as i16);
                g.x2 = g.x2.saturating_add(rand_between(rng, -20, 20) as i16);
                g.y2 = g.y2.saturating_add(rand_between(rng, -20, 20) as i16);
                g.x3 = g.x3.saturating_add(rand_between(rng, -20, 20) as i16);
                g.y3 = g.y3.saturating_add(rand_between(rng, -20, 20) as i16);
                g.normalize(config.width, config.height, config.margin);
            }
            2 => {
                g.x1 = g.x1.saturating_add(rand_between(rng, -5, 5) as i16);
                g.y1 = g.y1.saturating_add(rand_between(rng, -5, 5) as i16);
                g.x2 = g.x2.saturating_add(rand_between(rng, -5, 5) as i16);
                g.y2 = g.y2.saturating_add(rand_between(rng, -5, 5) as i16);
                g.x3 = g.x3.saturating_add(rand_between(rng, -5, 5) as i16);
                g.y3 = g.y3.saturating_add(rand_between(rng, -5, 5) as i16);
                g.normalize(config.width, config.height, config.margin);
            }
            3 => {
                let (new_oklab, _) = random_oklab_color(rng);
                g.oklab = new_oklab;
            }
            4 => {
                g.oklab = nudge_oklab(rng, g.oklab);
            }
            _ => {
                g.alpha = rand_between(rng, MINALPHA as i32, MAXALPHA as i32) as u8;
            }
        }
        g
    }

    fn recombine(&self, other: &Self, rng: &mut impl Rng) -> Self {
        if rng.random_bool(0.5) { self.clone() } else { other.clone() }
    }
}

/// Filled circle; centre is clamped to margin-extended bounds, radius is unclamped.
///
/// The rasteriser clips scanlines at the framebuffer edge, so a large radius extending
/// beyond the image is rendered correctly — only the visible arc is drawn.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircleGene {
    /// Centre x coordinate.
    pub cx: i16,
    /// Centre y coordinate.
    pub cy: i16,
    /// Radius in pixels (unclamped; rasteriser clips at image edge).
    pub radius: i16,
    /// Colour in `OKlab` [L, a, b].
    pub oklab: [f32; 3],
    /// Alpha as an integer percentage (1–100).
    pub alpha: u8,
    /// Draw-order key; lower values are drawn first (behind higher-valued genes).
    pub z_order: u16,
}

impl CircleGene {
    fn normalize(&mut self, width: u32, height: u32, margin: i16) {
        let w = width as i16;
        let h = height as i16;
        self.cx = clamp_coord(self.cx, -margin, w - 1 + margin);
        self.cy = clamp_coord(self.cy, -margin, h - 1 + margin);
    }
}

impl Gene for CircleGene {
    fn mutate(&self, rng: &mut impl Rng, config: &MutationConfig) -> Self {
        let mut g = self.clone();
        match rng.random_range(0..6u32) {
            0 => {
                g.cx = (rng.random::<u32>() % config.width) as i16;
                g.cy = (rng.random::<u32>() % config.height) as i16;
                g.radius = (rng.random::<u32>() % config.width) as i16;
                g.normalize(config.width, config.height, config.margin);
            }
            1 => {
                g.cx = g.cx.saturating_add(rand_between(rng, -20, 20) as i16);
                g.cy = g.cy.saturating_add(rand_between(rng, -20, 20) as i16);
                g.radius = g.radius.saturating_add(rand_between(rng, -20, 20) as i16);
                g.normalize(config.width, config.height, config.margin);
            }
            2 => {
                g.cx = g.cx.saturating_add(rand_between(rng, -5, 5) as i16);
                g.cy = g.cy.saturating_add(rand_between(rng, -5, 5) as i16);
                g.radius = g.radius.saturating_add(rand_between(rng, -5, 5) as i16);
                g.normalize(config.width, config.height, config.margin);
            }
            3 => {
                let (new_oklab, _) = random_oklab_color(rng);
                g.oklab = new_oklab;
            }
            4 => {
                g.oklab = nudge_oklab(rng, g.oklab);
            }
            _ => {
                g.alpha = rand_between(rng, MINALPHA as i32, MAXALPHA as i32) as u8;
            }
        }
        g
    }

    fn recombine(&self, other: &Self, rng: &mut impl Rng) -> Self {
        if rng.random_bool(0.5) { self.clone() } else { other.clone() }
    }
}

/// Filled n-gon (≥ 3 vertices) using scanline rasterisation.
///
/// Vertices are kept sorted by angle from the centroid on every `normalize` call,
/// eliminating self-intersecting edges. Alpha is stored as an integer percentage (1–100).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolygonGene {
    /// Vertices in angle-sorted order; always ≥ 3 entries.
    pub vertices: Vec<(i16, i16)>,
    /// Colour in `OKlab` [L, a, b].
    pub oklab: [f32; 3],
    /// Alpha as an integer percentage (1–100).
    pub alpha: u8,
    /// Draw-order key; lower values are drawn first (behind higher-valued genes).
    pub z_order: u16,
}

impl PolygonGene {
    fn normalize(&mut self, width: u32, height: u32, margin: i16) {
        let w = width as i16;
        let h = height as i16;
        for (vx, vy) in &mut self.vertices {
            *vx = clamp_coord(*vx, -margin, w - 1 + margin);
            *vy = clamp_coord(*vy, -margin, h - 1 + margin);
        }
        let n = self.vertices.len() as f32;
        let cx = self.vertices.iter().map(|(x, _)| *x as f32).sum::<f32>() / n;
        let cy = self.vertices.iter().map(|(_, y)| *y as f32).sum::<f32>() / n;
        self.vertices.sort_unstable_by(|(ax, ay), (bx, by)| {
            let a_angle = (*ay as f32 - cy).atan2(*ax as f32 - cx);
            let b_angle = (*by as f32 - cy).atan2(*bx as f32 - cx);
            a_angle.partial_cmp(&b_angle).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Split into two colour-diverged polygons by dividing the angle-sorted vertex list.
    ///
    /// Divides at a random start index: half A gets the first `n/2` vertices (wrapping),
    /// half B gets the rest. Colours are nudged in opposite directions on the a/b channels
    /// while L is kept identical. Returns `None` when fewer than 6 vertices.
    pub fn split(
        &self,
        rng: &mut impl Rng,
        width: u32,
        height: u32,
        margin: i16,
    ) -> Option<(Self, Self)> {
        let n = self.vertices.len();
        if n < 6 {
            return None;
        }
        let s = rng.random_range(0..n);
        let half = n / 2;

        let verts_a: Vec<(i16, i16)> = (0..half).map(|i| self.vertices[(s + i) % n]).collect();
        let verts_b: Vec<(i16, i16)> = (half..n).map(|i| self.vertices[(s + i) % n]).collect();

        let [l, a_ch, b_ch] = self.oklab;
        let mut gene_a = Self {
            vertices: verts_a,
            oklab: [l, a_ch + 0.05, b_ch - 0.05],
            alpha: self.alpha,
            z_order: self.z_order,
        };
        let mut gene_b = Self {
            vertices: verts_b,
            oklab: [l, a_ch - 0.05, b_ch + 0.05],
            alpha: self.alpha,
            z_order: self.z_order.saturating_add(1),
        };
        gene_a.normalize(width, height, margin);
        gene_b.normalize(width, height, margin);
        Some((gene_a, gene_b))
    }

    /// Angle-based crossover: keep half of `self`'s vertices and half of `other`'s.
    ///
    /// Selects a random dividing angle; keeps vertices from `self` whose angle from
    /// `self`'s centroid falls in `[α, α+π)` and vertices from `other` whose angle
    /// from `other`'s centroid falls in `[α+π, α+2π)`. Returns `None` if fewer than
    /// 3 vertices result. Colour is the per-channel `OKlab` midpoint; alpha is the average.
    pub fn angle_crossover(
        &self,
        other: &Self,
        rng: &mut impl Rng,
        width: u32,
        height: u32,
        margin: i16,
    ) -> Option<Self> {
        let dividing: f32 = rng.random::<f32>() * std::f32::consts::TAU;

        let na = self.vertices.len() as f32;
        let center_a = [
            self.vertices.iter().map(|(x, _)| *x as f32).sum::<f32>() / na,
            self.vertices.iter().map(|(_, y)| *y as f32).sum::<f32>() / na,
        ];

        let nb = other.vertices.len() as f32;
        let center_b = [
            other.vertices.iter().map(|(x, _)| *x as f32).sum::<f32>() / nb,
            other.vertices.iter().map(|(_, y)| *y as f32).sum::<f32>() / nb,
        ];

        let kept_a: Vec<(i16, i16)> = self
            .vertices
            .iter()
            .filter(|(vx, vy)| {
                let angle = (*vy as f32 - center_a[1]).atan2(*vx as f32 - center_a[0]);
                let relative = (angle - dividing).rem_euclid(std::f32::consts::TAU);
                relative < std::f32::consts::PI
            })
            .copied()
            .collect();

        let kept_b: Vec<(i16, i16)> = other
            .vertices
            .iter()
            .filter(|(vx, vy)| {
                let angle = (*vy as f32 - center_b[1]).atan2(*vx as f32 - center_b[0]);
                let relative = (angle - dividing).rem_euclid(std::f32::consts::TAU);
                relative >= std::f32::consts::PI
            })
            .copied()
            .collect();

        let mut combined = kept_a;
        combined.extend(kept_b);
        if combined.len() < 3 {
            return None;
        }

        let oklab = std::array::from_fn(|i| f32::midpoint(self.oklab[i], other.oklab[i]));
        let alpha =
            u16::midpoint(u16::from(self.alpha), u16::from(other.alpha)).max(u16::from(MINALPHA))
                as u8;

        let mut result = Self {
            vertices: combined,
            oklab,
            alpha,
            z_order: self.z_order,
        };
        result.normalize(width, height, margin);
        Some(result)
    }
}

impl Gene for PolygonGene {
    fn mutate(&self, rng: &mut impl Rng, config: &MutationConfig) -> Self {
        let mut g = self.clone();
        match rng.random_range(0..32u32) {
            0..4 => {
                let n = g.vertices.len();
                g.vertices = (0..n)
                    .map(|_| {
                        (
                            (rng.random::<u32>() % config.width) as i16,
                            (rng.random::<u32>() % config.height) as i16,
                        )
                    })
                    .collect();
                g.normalize(config.width, config.height, config.margin);
            }
            4..8 => {
                for (vx, vy) in &mut g.vertices {
                    *vx = vx.saturating_add(rand_between(rng, -20, 20) as i16);
                    *vy = vy.saturating_add(rand_between(rng, -20, 20) as i16);
                }
                g.normalize(config.width, config.height, config.margin);
            }
            8..12 => {
                for (vx, vy) in &mut g.vertices {
                    *vx = vx.saturating_add(rand_between(rng, -5, 5) as i16);
                    *vy = vy.saturating_add(rand_between(rng, -5, 5) as i16);
                }
                g.normalize(config.width, config.height, config.margin);
            }
            12..16 => {
                let (new_oklab, _) = random_oklab_color(rng);
                g.oklab = new_oklab;
            }
            16..20 => {
                g.oklab = nudge_oklab(rng, g.oklab);
            }
            20..24 => {
                g.alpha = rand_between(rng, MINALPHA as i32, MAXALPHA as i32) as u8;
            }
            // Less likely
            24 => {
                // Split a random edge by inserting its midpoint (nudged ±20 px).
                // Falls back to a small nudge when the vertex cap has been reached.
                if g.vertices.len() < config.max_polygon_vertices {
                    let n = g.vertices.len();
                    let edge = rng.random_range(0..n);
                    let (x1, y1) = g.vertices[edge];
                    let (x2, y2) = g.vertices[(edge + 1) % n];
                    let mx = ((x1 as i32 + x2 as i32) / 2 + rand_between(rng, -20, 20))
                        .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    let my = ((y1 as i32 + y2 as i32) / 2 + rand_between(rng, -20, 20))
                        .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    g.vertices.insert(edge + 1, (mx, my));
                } else {
                    for (vx, vy) in &mut g.vertices {
                        *vx = vx.saturating_add(rand_between(rng, -5, 5) as i16);
                        *vy = vy.saturating_add(rand_between(rng, -5, 5) as i16);
                    }
                }
                g.normalize(config.width, config.height, config.margin);
            }
            _ => {
                // Remove a random vertex; fall through to a small nudge if already at minimum.
                if g.vertices.len() > 3 {
                    let idx = rng.random_range(0..g.vertices.len());
                    g.vertices.remove(idx);
                } else {
                    for (vx, vy) in &mut g.vertices {
                        *vx = vx.saturating_add(rand_between(rng, -5, 5) as i16);
                        *vy = vy.saturating_add(rand_between(rng, -5, 5) as i16);
                    }
                }
                g.normalize(config.width, config.height, config.margin);
            }
        }
        g
    }

    fn recombine(&self, other: &Self, rng: &mut impl Rng) -> Self {
        if rng.random_bool(0.5) { self.clone() } else { other.clone() }
    }
}

/// The magnitude of a single z-order nudge mutation.
const Z_ORDER_NUDGE: u16 = 1000;

/// A geometric gene: Triangle, Circle, or Polygon.
///
/// Implements `Gene` by delegating to the inner typed struct. Each variant carries its
/// own `z_order` field so the enum can be freely moved without losing layering information.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ShapeGene {
    /// Filled triangle.
    Triangle(TriangleGene),
    /// Filled circle.
    Circle(CircleGene),
    /// Filled n-gon.
    Polygon(PolygonGene),
}

impl ShapeGene {
    /// Draw-order key; lower values are drawn first (appear behind higher-valued genes).
    #[must_use]
    pub fn z_order(&self) -> u16 {
        match self {
            Self::Triangle(t) => t.z_order,
            Self::Circle(c) => c.z_order,
            Self::Polygon(p) => p.z_order,
        }
    }

    /// Set the draw-order key without changing any other field.
    pub fn set_z_order(&mut self, z: u16) {
        match self {
            Self::Triangle(t) => t.z_order = z,
            Self::Circle(c) => c.z_order = z,
            Self::Polygon(p) => p.z_order = z,
        }
    }

    /// Approximate serialised byte cost of this gene, used for budget accounting.
    #[must_use]
    pub fn cost(&self) -> usize {
        match self {
            Self::Triangle(_) => TRIANGLE_COST,
            Self::Circle(_) => CIRCLE_COST,
            Self::Polygon(p) => POLYGON_BASE_COST + POLYGON_VERTEX_COST * p.vertices.len(),
        }
    }

    /// Create a `ShapeGene` with a random shape and a fully random z-order.
    pub(crate) fn random(rng: &mut impl Rng, config: &MutationConfig) -> Self {
        let z_order = rng.random::<u16>();
        match rng.random_range(0..5u32) {
            0 => Self::random_full(rng, config, z_order),
            1 => Self::random_small(rng, config, 5, z_order),
            2 => Self::random_small(rng, config, 10, z_order),
            3 => Self::random_small(rng, config, 25, z_order),
            _ => Self::random_small(rng, config, 2, z_order),
        }
    }

    /// Create a random gene anywhere in the image bounds.
    pub fn random_full(rng: &mut impl Rng, config: &MutationConfig, z_order: u16) -> Self {
        let (oklab, alpha) = random_oklab_color(rng);
        let width = config.width;
        let height = config.height;
        let margin = config.margin;
        match select_shape_type(rng, config.use_triangles, config.use_circles, config.use_polygons)
        {
            ShapeKind::Triangle => {
                let mut g = TriangleGene {
                    x1: (rng.random::<u32>() % width) as i16,
                    y1: (rng.random::<u32>() % height) as i16,
                    x2: (rng.random::<u32>() % width) as i16,
                    y2: (rng.random::<u32>() % height) as i16,
                    x3: (rng.random::<u32>() % width) as i16,
                    y3: (rng.random::<u32>() % height) as i16,
                    oklab,
                    alpha,
                    z_order,
                };
                g.normalize(width, height, margin);
                Self::Triangle(g)
            }
            ShapeKind::Circle => {
                let mut g = CircleGene {
                    cx: (rng.random::<u32>() % width) as i16,
                    cy: (rng.random::<u32>() % height) as i16,
                    radius: (rng.random::<u32>() % width) as i16,
                    oklab,
                    alpha,
                    z_order,
                };
                g.normalize(width, height, margin);
                Self::Circle(g)
            }
            ShapeKind::Polygon => {
                let n = rng.random_range(3..=6usize);
                let vertices = (0..n)
                    .map(|_| {
                        (
                            (rng.random::<u32>() % width) as i16,
                            (rng.random::<u32>() % height) as i16,
                        )
                    })
                    .collect();
                let mut g = PolygonGene { vertices, oklab, alpha, z_order };
                g.normalize(width, height, margin);
                Self::Polygon(g)
            }
        }
    }

    fn random_small(rng: &mut impl Rng, config: &MutationConfig, delta: i32, z_order: u16) -> Self {
        let (oklab, alpha) = random_oklab_color(rng);
        let width = config.width;
        let height = config.height;
        let margin = config.margin;
        let x = (rng.random::<u32>() % width) as i32;
        let y = (rng.random::<u32>() % height) as i32;
        match select_shape_type(rng, config.use_triangles, config.use_circles, config.use_polygons)
        {
            ShapeKind::Triangle => {
                let mut g = TriangleGene {
                    x1: (x + rand_between(rng, -delta, delta)) as i16,
                    y1: (y + rand_between(rng, -delta, delta)) as i16,
                    x2: (x + rand_between(rng, -delta, delta)) as i16,
                    y2: (y + rand_between(rng, -delta, delta)) as i16,
                    x3: (x + rand_between(rng, -delta, delta)) as i16,
                    y3: (y + rand_between(rng, -delta, delta)) as i16,
                    oklab,
                    alpha,
                    z_order,
                };
                g.normalize(width, height, margin);
                Self::Triangle(g)
            }
            ShapeKind::Circle => {
                let mut g = CircleGene {
                    cx: x as i16,
                    cy: y as i16,
                    radius: rand_between(rng, 1, delta) as i16,
                    oklab,
                    alpha,
                    z_order,
                };
                g.normalize(width, height, margin);
                Self::Circle(g)
            }
            ShapeKind::Polygon => {
                let n = rng.random_range(3..=6usize);
                let vertices = (0..n)
                    .map(|_| {
                        (
                            (x + rand_between(rng, -delta, delta)) as i16,
                            (y + rand_between(rng, -delta, delta)) as i16,
                        )
                    })
                    .collect();
                let mut g = PolygonGene { vertices, oklab, alpha, z_order };
                g.normalize(width, height, margin);
                Self::Polygon(g)
            }
        }
    }
}

impl Gene for ShapeGene {
    fn mutate(&self, rng: &mut impl Rng, config: &MutationConfig) -> Self {
        // ~10% chance: nudge z_order rather than mutating the shape geometry/colour
        if rng.random_range(0..10u32) == 0 {
            let delta = rng.random_range(0..=Z_ORDER_NUDGE);
            let new_z = if rng.random_bool(0.5) {
                self.z_order().saturating_add(delta)
            } else {
                self.z_order().saturating_sub(delta)
            };
            let mut result = self.clone();
            result.set_z_order(new_z);
            return result;
        }
        match self {
            Self::Triangle(t) => Self::Triangle(t.mutate(rng, config)),
            Self::Circle(c) => Self::Circle(c.mutate(rng, config)),
            Self::Polygon(p) => Self::Polygon(p.mutate(rng, config)),
        }
    }

    fn recombine(&self, other: &Self, rng: &mut impl Rng) -> Self {
        // Blending across shape variants is ill-defined; pick one parent's gene wholesale.
        // z_order midpoint preserves approximate layering from both parents.
        let z_mid =
            u32::midpoint(u32::from(self.z_order()), u32::from(other.z_order())) as u16;
        let mut child = if rng.random_bool(0.5) { self.clone() } else { other.clone() };
        child.set_z_order(z_mid);
        child
    }
}

/// An evolving background colour (always fully opaque).
///
/// No alpha — the background fills the canvas before any shapes are composited on top.
/// Colour is stored in `OKlab` [L, a, b] for perceptually uniform mutation and correct
/// midpoint interpolation during recombination.
/// `Default` gives [0,0,0] — pure black in `OKlab` — preserving existing behaviour for
/// old checkpoints without this field.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackgroundGene {
    /// Colour in `OKlab` [L, a, b].
    pub oklab: [f32; 3],
}

impl Gene for BackgroundGene {
    fn mutate(&self, rng: &mut impl Rng, _config: &MutationConfig) -> Self {
        if rng.random_bool(0.5) {
            let (oklab, _) = random_oklab_color(rng);
            Self { oklab }
        } else {
            Self { oklab: nudge_oklab(rng, self.oklab) }
        }
    }

    fn recombine(&self, other: &Self, _rng: &mut impl Rng) -> Self {
        // Arithmetic mean in OKlab is the perceptually correct midpoint.
        Self {
            oklab: std::array::from_fn(|i| f32::midpoint(self.oklab[i], other.oklab[i])),
        }
    }
}

/// An evolving Gaussian blur radius (always positive when present).
///
/// `None` blur is represented by `Option<BlurGene>` at the genome level rather than a
/// sentinel value, so the gene itself is always valid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlurGene {
    /// Gaussian sigma in pixels.  Always > 0.
    pub radius: f32,
}

impl Gene for BlurGene {
    fn mutate(&self, rng: &mut impl Rng, _config: &MutationConfig) -> Self {
        let delta = rng.random::<f32>() * 2.0;
        let new_r = if rng.random_bool(0.5) {
            self.radius + delta
        } else {
            self.radius - delta
        };
        // Caller converts to None if radius drops below threshold — no clamping here
        // so the genome can organically discover and remove blur.
        Self { radius: new_r }
    }

    fn recombine(&self, other: &Self, _rng: &mut impl Rng) -> Self {
        Self {
            radius: f32::midpoint(self.radius, other.radius),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    fn seeded() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    fn black_oklab() -> [f32; 3] {
        [0.0, 0.0, 0.0]
    }

    fn sample_triangle_gene() -> TriangleGene {
        TriangleGene {
            x1: 0, y1: 0, x2: 50, y2: 0, x3: 25, y3: 50,
            oklab: [0.5, 0.0, 0.0],
            alpha: 50,
            z_order: 100,
        }
    }

    fn sample_config() -> MutationConfig {
        MutationConfig {
            width: 100,
            height: 100,
            margin: 0,
            mutation_rate: 500,
            use_triangles: true,
            use_circles: false,
            use_polygons: false,
            max_polygon_vertices: 64,
        }
    }

    #[test]
    fn triangle_normalise_sorts_by_y() {
        let mut g = TriangleGene {
            x1: 10, y1: 30,
            x2: 20, y2: 10,
            x3: 30, y3: 20,
            oklab: black_oklab(),
            alpha: 50,
            z_order: 0,
        };
        g.normalize(100, 100, 0);
        assert!(g.y1 <= g.y2 && g.y2 <= g.y3, "y values must be non-decreasing after normalise");
    }

    #[test]
    fn circle_normalise_clamps_centre_only() {
        let mut g = CircleGene {
            cx: -200, cy: 300, radius: 100,
            oklab: black_oklab(),
            alpha: 50,
            z_order: 0,
        };
        g.normalize(50, 50, 0);
        assert_eq!(g.cx, 0, "cx should be clamped to 0");
        assert_eq!(g.cy, 49, "cy should be clamped to h-1");
        assert_eq!(g.radius, 100, "radius must be unchanged");
    }

    #[test]
    fn polygon_normalise_clamps_vertices() {
        let mut g = PolygonGene {
            vertices: vec![(-200, 300), (100, 100), (50, -100)],
            oklab: black_oklab(),
            alpha: 50,
            z_order: 0,
        };
        g.normalize(50, 50, 0);
        for (vx, vy) in &g.vertices {
            assert!(*vx >= 0 && *vx <= 49, "vx {vx} out of bounds [0,49]");
            assert!(*vy >= 0 && *vy <= 49, "vy {vy} out of bounds [0,49]");
        }
    }

    #[test]
    fn normalise_with_margin_allows_coords_outside_image() {
        let margin: i16 = 10;
        let mut g = TriangleGene {
            x1: -5, y1: -5,
            x2: 104, y2: 5,
            x3: 5, y3: 104,
            oklab: black_oklab(),
            alpha: 50,
            z_order: 0,
        };
        g.normalize(100, 100, margin);
        assert_eq!(g.x1, -5, "x1 within margin should be unchanged");
        assert_eq!(g.y1, -5, "y1 within margin should be unchanged");
        assert_eq!(g.x2, 104, "x2 within margin should be unchanged");
        assert_eq!(g.x3, 5, "x3 in-bounds should be unchanged");
        assert_eq!(g.y3, 104, "y3 within margin should be unchanged");
        assert!(g.y1 <= g.y2 && g.y2 <= g.y3, "y values must be non-decreasing");
    }

    #[test]
    fn polygon_normalise_sorts_by_angle() {
        let mut g = PolygonGene {
            vertices: vec![(50, 0), (100, 50), (50, 100), (0, 50)],
            oklab: black_oklab(),
            alpha: 50,
            z_order: 0,
        };
        g.normalize(200, 200, 0);
        let cx = 50.0f32;
        let cy = 50.0f32;
        let angles: Vec<f32> = g
            .vertices
            .iter()
            .map(|(x, y)| (*y as f32 - cy).atan2(*x as f32 - cx))
            .collect();
        for w in angles.windows(2) {
            assert!(w[0] <= w[1], "angles not sorted: {:?} > {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn alpha_mutation_stays_in_bounds() {
        let mut rng = seeded();
        let config = sample_config();
        let mut g = ShapeGene::Triangle(sample_triangle_gene());
        for _ in 0..1000 {
            g = g.mutate(&mut rng, &config);
            if let ShapeGene::Triangle(t) = &g {
                assert!(
                    (MINALPHA..=MAXALPHA).contains(&t.alpha),
                    "alpha {} out of bounds",
                    t.alpha
                );
            }
        }
    }

    #[test]
    fn polygon_mutation_vertex_count_minimum() {
        let mut rng = seeded();
        let config = MutationConfig {
            use_triangles: false,
            use_circles: false,
            use_polygons: true,
            ..sample_config()
        };
        let mut g = PolygonGene {
            vertices: vec![(0, 0), (50, 0), (25, 50)],
            oklab: black_oklab(),
            alpha: 50,
            z_order: 0,
        };
        for _ in 0..1000 {
            g = g.mutate(&mut rng, &config);
            assert!(
                g.vertices.len() >= 3,
                "polygon must never have fewer than 3 vertices, got {}",
                g.vertices.len()
            );
        }
    }

    #[test]
    fn polygon_mutation_split_increases_count() {
        let mut rng = seeded();
        let config = MutationConfig {
            use_triangles: false,
            use_circles: false,
            use_polygons: true,
            ..sample_config()
        };
        let mut g = PolygonGene {
            vertices: vec![(0, 0), (50, 0), (25, 50)],
            oklab: black_oklab(),
            alpha: 50,
            z_order: 0,
        };
        let mut saw_split = false;
        for _ in 0..1000 {
            let before = g.vertices.len();
            g = g.mutate(&mut rng, &config);
            let after = g.vertices.len();
            assert!(after >= 3, "vertex count must never fall below 3");
            if after > before {
                assert_eq!(
                    after,
                    before + 1,
                    "split must add exactly one vertex (before={before}, after={after})"
                );
                saw_split = true;
            }
        }
        assert!(saw_split, "expected at least one split in 1000 mutations");
    }

    #[test]
    fn polygon_split_respects_vertex_cap() {
        let mut rng = seeded();
        let cap = 5usize;
        let config = MutationConfig {
            use_polygons: true,
            use_triangles: false,
            use_circles: false,
            max_polygon_vertices: cap,
            ..sample_config()
        };
        let mut g = PolygonGene {
            vertices: vec![(0, 0), (50, 0), (25, 50)],
            oklab: black_oklab(),
            alpha: 50,
            z_order: 0,
        };
        for _ in 0..5000 {
            g = g.mutate(&mut rng, &config);
            assert!(
                g.vertices.len() <= cap,
                "vertex count {} exceeds cap {cap}",
                g.vertices.len()
            );
        }
    }

    #[test]
    fn split_produces_two_valid_polygons() {
        let mut rng = seeded();
        let g = PolygonGene {
            vertices: vec![(0, 0), (50, 0), (100, 0), (100, 50), (50, 100), (0, 50)],
            oklab: [0.5, 0.0, 0.0],
            alpha: 50,
            z_order: 10,
        };
        let result = g.split(&mut rng, 200, 200, 0);
        assert!(result.is_some(), "split should succeed on a 6-vertex polygon");
        let (a, b) = result.unwrap();
        assert!(a.vertices.len() >= 3, "half A must have >= 3 vertices, got {}", a.vertices.len());
        assert!(b.vertices.len() >= 3, "half B must have >= 3 vertices, got {}", b.vertices.len());
        assert_eq!(a.vertices.len() + b.vertices.len(), 6, "total vertex count must equal original");
        assert_eq!(a.z_order, 10, "half A inherits parent z_order");
        assert_eq!(b.z_order, 11, "half B gets z_order + 1");
    }

    #[test]
    fn split_rejects_small_polygon() {
        let mut rng = seeded();
        let g = PolygonGene {
            vertices: vec![(0, 0), (50, 0), (25, 50)],
            oklab: [0.5, 0.0, 0.0],
            alpha: 50,
            z_order: 0,
        };
        assert!(
            g.split(&mut rng, 100, 100, 0).is_none(),
            "split should return None for < 6 vertices"
        );
    }

    #[test]
    fn angle_crossover_produces_valid_shape() {
        let mut rng = seeded();
        let a = PolygonGene {
            vertices: vec![(0, 0), (50, 0), (100, 50), (50, 100), (0, 50)],
            oklab: [0.5, 0.1, 0.0],
            alpha: 50,
            z_order: 0,
        };
        let b = PolygonGene {
            vertices: vec![(25, 25), (75, 25), (75, 75), (25, 75)],
            oklab: [0.7, -0.1, 0.1],
            alpha: 70,
            z_order: 0,
        };
        let mut found = false;
        for _ in 0..200 {
            if let Some(result) = a.angle_crossover(&b, &mut rng, 200, 200, 0) {
                assert!(
                    result.vertices.len() >= 3,
                    "crossover result must have >= 3 vertices"
                );
                found = true;
                break;
            }
        }
        assert!(found, "expected at least one successful crossover in 200 tries");
    }

    #[test]
    fn gene_costs_are_in_plausible_range() {
        let triangle_gene = ShapeGene::Triangle(TriangleGene {
            x1: 0, y1: 0, x2: 50, y2: 0, x3: 25, y3: 50,
            oklab: [0.5, 0.0, 0.0],
            alpha: 50,
            z_order: 100,
        });
        let circle_gene = ShapeGene::Circle(CircleGene {
            cx: 25, cy: 25, radius: 10,
            oklab: [0.5, 0.0, 0.0],
            alpha: 50,
            z_order: 100,
        });
        let triangle_bytes =
            bincode::serde::encode_to_vec(&triangle_gene, bincode::config::standard())
                .expect("encode")
                .len();
        let circle_bytes =
            bincode::serde::encode_to_vec(&circle_gene, bincode::config::standard())
                .expect("encode")
                .len();
        // Allow 50% variance: constants are heuristics for a 256×256 image with typical coords.
        assert!(
            triangle_bytes <= TRIANGLE_COST * 3 / 2,
            "TRIANGLE_COST far off: actual={triangle_bytes}, const={TRIANGLE_COST}"
        );
        assert!(
            circle_bytes <= CIRCLE_COST * 3 / 2,
            "CIRCLE_COST far off: actual={circle_bytes}, const={CIRCLE_COST}"
        );
        assert!(TRIANGLE_COST > CIRCLE_COST, "triangle must cost more than circle");
    }
}
