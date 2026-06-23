use serde::{Deserialize, Serialize};

use crate::gene::TRIANGLE_COST;

/// Persistent annealing control variables, serialised into the checkpoint.
///
/// `max_cost` and `max_cost_incremental` are denominated in approximate bincode bytes
/// (see `TRIANGLE_COST` and friends in `gene`). Old checkpoints written before the cost
/// model was introduced have raw shape-count values in these fields and must be restarted
/// with `process --restart`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnnealingState {
    /// Total byte-cost budget for the genome in this run.
    pub max_cost: usize,
    /// Current incremental cost cap, growing toward `max_cost` over generations.
    pub max_cost_incremental: usize,
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
    /// Create a fresh annealing state with the given cost budget values.
    #[must_use]
    pub fn new(max_cost: usize, initial_cost: usize) -> Self {
        Self {
            max_cost,
            max_cost_incremental: initial_cost,
            temperature: 0.10,
            absbestdiff: 100.0,
            generation: 0,
            blur_radius: None,
        }
    }

    /// Convenience constructor for tests: multiply shape counts by `TRIANGLE_COST`.
    #[must_use]
    pub fn new_from_shape_count(max_shapes: usize, initial_shapes: usize) -> Self {
        Self::new(max_shapes * TRIANGLE_COST, initial_shapes * TRIANGLE_COST)
    }
}
