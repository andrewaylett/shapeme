//! Gene-level traits and types for shapeme's genetic representation.
//!
//! A `Gene` is a single mutable unit of genetic information. `ShapeGene` wraps a `Shape`
//! together with an explicit z-ordering key, enabling stable layering independent of
//! `Vec` position. `BlurGene` encodes the optional global Gaussian blur radius.
//!
//! # Why explicit z-ordering?
//!
//! Previously the draw order was determined by position in a `Vec`. Recombination via
//! single-point crossover would silently reorder shapes if the two parents had different
//! lengths. By tagging each gene with a `z_order` key we can sort consistently after
//! crossover, and swap/nudge z-values as a distinct mutation step.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::oklab;
use crate::shapes::{Shape, mutate_shape, random_shape, random_small_shape};

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

/// A shape combined with an explicit z-ordering key for layering.
///
/// `z_order` is the primary sort key when rasterising; lower values are drawn first
/// (i.e., appear behind higher-valued shapes).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeGene {
    /// The geometric primitive to draw.
    pub shape: Shape,
    /// Draw order key.  Shapes are sorted ascending before rasterisation.
    pub z_order: u16,
}

/// The magnitude of a single z-order nudge mutation.
///
/// With `max_shapes ≤ 64`, a delta of 1000 provides ~16 distinct levels per slot,
/// producing meaningful reorderings without saturating immediately.
const Z_ORDER_NUDGE: u16 = 1000;

impl Gene for ShapeGene {
    fn mutate(&self, rng: &mut impl Rng, config: &MutationConfig) -> Self {
        let mut shape = self.shape.clone();
        let mut z_order = self.z_order;

        // ~10% chance: nudge z_order rather than mutating the shape geometry/colour
        if rng.random_range(0..10u32) == 0 {
            let delta = rng.random_range(0..=Z_ORDER_NUDGE);
            z_order = if rng.random_bool(0.5) {
                z_order.saturating_add(delta)
            } else {
                z_order.saturating_sub(delta)
            };
        } else {
            mutate_shape(rng, &mut shape, config.width, config.height, config.margin, config.max_polygon_vertices);
        }

        Self { shape, z_order }
    }

    fn recombine(&self, other: &Self, rng: &mut impl Rng) -> Self {
        // Blending across shape variants is ill-defined; pick one parent's shape wholesale.
        let shape = if rng.random_bool(0.5) {
            self.shape.clone()
        } else {
            other.shape.clone()
        };
        // z_order midpoint preserves approximate layering from both parents.
        let z_order = u32::midpoint(u32::from(self.z_order), u32::from(other.z_order)) as u16;
        Self { shape, z_order }
    }
}

impl ShapeGene {
    /// Approximate serialised byte cost of this gene, used for budget accounting.
    #[must_use]
    pub fn cost(&self) -> usize {
        match &self.shape {
            Shape::Triangle { .. } => TRIANGLE_COST,
            Shape::Circle { .. } => CIRCLE_COST,
            Shape::Polygon { vertices, .. } => POLYGON_BASE_COST + POLYGON_VERTEX_COST * vertices.len(),
        }
    }

    /// Create a `ShapeGene` with a random shape and a fully random z-order.
    pub(crate) fn random(rng: &mut impl Rng, config: &MutationConfig) -> Self {
        let shape = match rng.random_range(0..5u32) {
            0 => random_shape(
                rng,
                config.width,
                config.height,
                config.use_triangles,
                config.use_circles,
                config.use_polygons,
                config.margin,
            ),
            1 => random_small_shape(
                rng,
                config.width,
                config.height,
                5,
                config.use_triangles,
                config.use_circles,
                config.use_polygons,
                config.margin,
            ),
            2 => random_small_shape(
                rng,
                config.width,
                config.height,
                10,
                config.use_triangles,
                config.use_circles,
                config.use_polygons,
                config.margin,
            ),
            3 => random_small_shape(
                rng,
                config.width,
                config.height,
                25,
                config.use_triangles,
                config.use_circles,
                config.use_polygons,
                config.margin,
            ),
            _ => random_small_shape(
                rng,
                config.width,
                config.height,
                2,
                config.use_triangles,
                config.use_circles,
                config.use_polygons,
                config.margin,
            ),
        };
        Self {
            shape,
            z_order: rng.random::<u16>(),
        }
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
            // Full random replacement via a random sRGB colour
            let r: u8 = rng.random();
            let g: u8 = rng.random();
            let b: u8 = rng.random();
            Self { oklab: oklab::srgb_u8_to_oklab(r, g, b) }
        } else {
            // Small perceptually uniform nudge on each OKlab channel
            let mut delta = || (rng.random::<f32>() * 2.0 - 1.0) * 0.02;
            Self {
                oklab: [
                    (self.oklab[0] + delta()).clamp(0.0, 1.0),
                    self.oklab[1] + delta(),
                    self.oklab[2] + delta(),
                ],
            }
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
    use super::*;
    use crate::shapes::Shape;

    #[test]
    fn gene_costs_are_in_plausible_range() {
        // Actual encoded sizes vary with coordinate magnitude (bincode v2 uses varint).
        // Constants are approximate; this test verifies they are in the right ballpark.
        let triangle_gene = ShapeGene {
            shape: Shape::Triangle { x1: 0, y1: 0, x2: 50, y2: 0, x3: 25, y3: 50, oklab: [0.5, 0.0, 0.0], alpha: 50 },
            z_order: 100,
        };
        let circle_gene = ShapeGene {
            shape: Shape::Circle { cx: 25, cy: 25, radius: 10, oklab: [0.5, 0.0, 0.0], alpha: 50 },
            z_order: 100,
        };
        let triangle_bytes = bincode::serde::encode_to_vec(&triangle_gene, bincode::config::standard()).expect("encode").len();
        let circle_bytes = bincode::serde::encode_to_vec(&circle_gene, bincode::config::standard()).expect("encode").len();
        // Allow 50% variance: constants are heuristics for a 256×256 image with typical coords.
        assert!(triangle_bytes <= TRIANGLE_COST * 3 / 2, "TRIANGLE_COST far off: actual={triangle_bytes}, const={TRIANGLE_COST}");
        assert!(circle_bytes <= CIRCLE_COST * 3 / 2, "CIRCLE_COST far off: actual={circle_bytes}, const={CIRCLE_COST}");
        // Triangle should cost more than circle (extra coordinates)
        assert!(TRIANGLE_COST > CIRCLE_COST, "triangle must cost more than circle");
    }
}
