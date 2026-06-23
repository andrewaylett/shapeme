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

use crate::shapes::{Shape, mutate_shape, random_shape, random_small_shape};

/// Approximate bincode byte cost of a `ShapeGene` wrapping a Triangle.
///
/// Used as the base budget unit: `--max-shapes N` → `max_cost = N × TRIANGLE_COST`.
pub const TRIANGLE_COST: usize = 22;
/// Approximate bincode byte cost of a `ShapeGene` wrapping a Circle.
pub const CIRCLE_COST: usize = 16;
/// Approximate bincode byte cost of a `ShapeGene` wrapping a Polygon, excluding per-vertex cost.
pub const POLYGON_BASE_COST: usize = 18;
/// Additional cost per polygon vertex.
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
    /// Maximum number of vertices a single Polygon gene may grow to via split_edge.
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
/// `Default` gives black, preserving existing behaviour for old checkpoints without this field.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackgroundGene {
    /// Red channel (0–255).
    pub r: u8,
    /// Green channel (0–255).
    pub g: u8,
    /// Blue channel (0–255).
    pub b: u8,
}

impl Gene for BackgroundGene {
    fn mutate(&self, rng: &mut impl Rng, _config: &MutationConfig) -> Self {
        if rng.random_bool(0.5) {
            // Full random replacement
            Self {
                r: rng.random(),
                g: rng.random(),
                b: rng.random(),
            }
        } else {
            // Small nudge on each channel
            let mut nudge = |v: u8| -> u8 {
                let delta = rng.random_range(0u8..=5u8);
                if rng.random_bool(0.5) { v.saturating_add(delta) } else { v.saturating_sub(delta) }
            };
            Self { r: nudge(self.r), g: nudge(self.g), b: nudge(self.b) }
        }
    }

    fn recombine(&self, other: &Self, _rng: &mut impl Rng) -> Self {
        Self {
            r: u8::midpoint(self.r, other.r),
            g: u8::midpoint(self.g, other.g),
            b: u8::midpoint(self.b, other.b),
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
