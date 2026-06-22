use serde::{Deserialize, Serialize};

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
