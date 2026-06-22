use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::shapes::{Shape, mutate_shape, random_shape, random_small_shape};

/// Persistent annealing control variables, serialised into the checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnnealingState {
    /// Total shape budget for this run.
    pub max_shapes: usize,
    /// Current incremental shape cap, growing toward `max_shapes` over generations.
    pub max_shapes_incremental: usize,
    /// Simulated-annealing temperature; decays toward 0 as generations accumulate.
    pub temperature: f32,
    /// Lowest diff percentage seen across all generations; persisted in the checkpoint.
    pub absbestdiff: f32,
    /// Total generation count, advancing by one per candidate evaluation.
    pub generation: i64,
    /// Blur radius of the current absolute best, evolved during the run.
    pub blur_radius: Option<f32>,
}

impl AnnealingState {
    /// Create a fresh annealing state for a run with the given shape budget.
    #[must_use]
    pub fn new(max_shapes: usize, initial_shapes: usize) -> Self {
        Self {
            max_shapes,
            max_shapes_incremental: initial_shapes,
            temperature: 0.10,
            absbestdiff: 100.0,
            generation: 0,
            blur_radius: None,
        }
    }
}

/// A candidate solution: an ordered list of shapes with an associated blur radius.
pub struct ShapeSet {
    /// Shapes in draw order.
    pub shapes: Vec<Shape>,
    pub(crate) capacity: usize,
    /// Per-candidate blur radius, mutated independently each generation.
    pub blur_radius: Option<f32>,
}

impl ShapeSet {
    /// Create an empty shape set with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            shapes: Vec::with_capacity(capacity),
            capacity,
            blur_radius: None,
        }
    }

    /// The active shapes in draw order.
    #[must_use]
    pub fn active(&self) -> &[Shape] {
        &self.shapes
    }

    pub(crate) fn len(&self) -> usize {
        self.shapes.len()
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "annealing parameters and shape configuration flags are all necessary"
)]
/// Apply one generation of mutations to a shape set.
///
/// May add, remove, swap, or individually mutate shapes, and may nudge the blur radius.
pub fn mutate_shapes(
    rng: &mut impl Rng,
    set: &mut ShapeSet,
    annealing: &AnnealingState,
    width: u32,
    height: u32,
    mutation_rate: u32,
    use_triangles: bool,
    use_circles: bool,
) {
    // The C source hardcodes 10 mutation attempts per generation regardless
    // of the number of shapes, so we match that here.
    const MUTATION_ATTEMPTS: usize = 10;

    let margin = set.blur_radius.map_or(0, |r| r.ceil() as i16);

    // 10% chance: add a new shape
    if rng.random_range(0..10u32) == 0
        && set.len() < set.capacity
        && set.len() < annealing.max_shapes_incremental
    {
        tracing::trace!(mutation = "add-shape", "adding shape");
        let new_shape = match rng.random_range(0..5u32) {
            0 => random_shape(rng, width, height, use_triangles, use_circles, margin),
            1 => random_small_shape(rng, width, height, 5, use_triangles, use_circles, margin),
            2 => random_small_shape(rng, width, height, 10, use_triangles, use_circles, margin),
            3 => random_small_shape(rng, width, height, 25, use_triangles, use_circles, margin),
            _ => random_small_shape(rng, width, height, 2, use_triangles, use_circles, margin),
        };
        set.shapes.push(new_shape);
        return;
    }

    // 5% chance: remove a shape
    if rng.random_range(0..20u32) == 0 && set.len() > 1 {
        tracing::trace!(mutation = "remove-shape", "removing shape");
        let idx = rng.random_range(0..set.len());
        set.shapes.remove(idx);
        return;
    }

    // 5% chance: swap two shapes
    if rng.random_range(0..20u32) == 0 && set.len() >= 2 {
        tracing::trace!(mutation = "swap-shapes", "swapping shapes");
        let a = rng.random_range(0..set.len());
        let b = rng.random_range(0..set.len());
        if a != b {
            set.shapes.swap(a, b);
        }
    }

    // ~5% chance: nudge the blur radius so the algorithm can discover or remove blur organically
    if rng.random_range(0..20u32) == 0 {
        tracing::trace!(mutation = "nudge-blur", "nudging blur radius");
        set.blur_radius = set.blur_radius.map_or(Some(0.5), |r| {
            let delta = rng.random::<f32>() * 2.0;
            let new_r = if rng.random_bool(0.5) {
                r + delta
            } else {
                r - delta
            };
            if new_r < 0.1 { None } else { Some(new_r) }
        });
    }

    // Mutate random shapes
    tracing::trace!(mutation = "mutate-shapes", "mutating individual shapes");
    for _ in 0..MUTATION_ATTEMPTS {
        let idx = rng.random_range(0..set.len());
        if rng.random_range(0..1000u32) < mutation_rate {
            mutate_shape(rng, &mut set.shapes[idx], width, height, margin);
        }
    }
}
