use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::shapes::{Shape, mutate_shape, random_shape, random_small_shape};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AnnealingState {
    pub(crate) max_shapes: usize,
    pub(crate) max_shapes_incremental: usize,
    pub(crate) temperature: f32,
    pub(crate) absbestdiff: f32,
    pub(crate) generation: i64,
    /// Blur radius of the current absolute best, evolved during the run and persisted in the checkpoint.
    pub(crate) blur_radius: Option<f32>,
}

impl AnnealingState {
    pub(crate) fn new(max_shapes: usize, initial_shapes: usize) -> Self {
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

pub(crate) struct ShapeSet {
    pub(crate) shapes: Vec<Shape>,
    pub(crate) capacity: usize,
    /// Per-candidate blur radius, mutated independently from the accepted best each generation.
    pub(crate) blur_radius: Option<f32>,
}

impl ShapeSet {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            shapes: Vec::with_capacity(capacity),
            capacity,
            blur_radius: None,
        }
    }

    pub(crate) fn active(&self) -> &[Shape] {
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
pub(crate) fn mutate_shapes(
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

    let margin = set.blur_radius.map(|r| r.ceil() as i16).unwrap_or(0);

    // 10% chance: add a new shape
    if rng.random_range(0..10u32) == 0
        && set.len() < set.capacity
        && set.len() < annealing.max_shapes_incremental
    {
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
        let idx = rng.random_range(0..set.len());
        set.shapes.remove(idx);
        return;
    }

    // 5% chance: swap two shapes
    if rng.random_range(0..20u32) == 0 && set.len() >= 2 {
        let a = rng.random_range(0..set.len());
        let b = rng.random_range(0..set.len());
        if a != b {
            set.shapes.swap(a, b);
        }
    }

    // ~5% chance: nudge the blur radius so the algorithm can discover or remove blur organically
    if rng.random_range(0..20u32) == 0 {
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
    for _ in 0..MUTATION_ATTEMPTS {
        let idx = rng.random_range(0..set.len());
        if rng.random_range(0..1000u32) < mutation_rate {
            mutate_shape(rng, &mut set.shapes[idx], width, height, margin);
        }
    }
}
