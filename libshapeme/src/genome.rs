//! Genome-level traits and types combining multiple genes into a complete candidate solution.
//!
//! A `Genome` can evaluate its own fitness, mutate itself, and recombine with another genome
//! of the same type.  `ShapeGenome` is the concrete implementation used by the main annealing
//! loop: it holds a non-empty `Vec<ShapeGene>` and an optional `BlurGene`.
//!
//! # Design rationale
//!
//! Moving mutation and fitness evaluation into the genome type makes the annealing loop
//! independent of representation details.  The `fitness` method accepts a caller-owned
//! scratch buffer so the hot path never allocates per-generation.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::annealing::AnnealingState;
use crate::gene::{BackgroundGene, BlurGene, Gene, MutationConfig, ShapeGene};
use crate::render::{apply_blur, compute_diff, draw_genes};

/// A complete candidate solution produced by combining genes.
pub trait Genome: Clone + Send + Sync {
    /// Compute the percentage diff between this genome's rendering and `target`.
    ///
    /// `target` and `scratch` are `OKlab` f32 buffers (3 floats per pixel).
    /// `scratch` is reused by the caller between calls — avoids a per-call allocation.
    fn fitness(&self, target: &[f32], width: u32, height: u32, scratch: &mut Vec<f32>) -> f32;

    /// Return a mutated copy of this genome.
    #[must_use]
    fn mutate(
        &self,
        rng: &mut impl Rng,
        state: &AnnealingState,
        config: &MutationConfig,
    ) -> Self;

    /// Produce an offspring genome from `self` and `other`.
    #[must_use]
    fn recombine(&self, other: &Self, rng: &mut impl Rng) -> Self;
}

/// A candidate solution: a non-empty list of `ShapeGene`s plus an optional blur and a background.
///
/// The `shapes` invariant (non-empty) is maintained by all mutation and recombination paths.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeGenome {
    /// Shapes in arbitrary order; sorted by `z_order` before rasterisation.
    pub shapes: Vec<ShapeGene>,
    /// Optional global Gaussian blur applied after rasterisation.
    pub blur: Option<BlurGene>,
    /// Evolving canvas background colour, composited before any shapes.
    #[serde(default)]
    pub background: BackgroundGene,
}

impl ShapeGenome {
    /// Extract genes sorted by z-order, for display or SVG output.
    #[must_use]
    pub fn sorted_genes(&self) -> Vec<&ShapeGene> {
        let mut genes: Vec<&ShapeGene> = self.shapes.iter().collect();
        genes.sort_unstable_by_key(|g| g.z_order());
        genes
    }


    /// The blur radius, if any.
    #[must_use]
    pub fn blur_radius(&self) -> Option<f32> {
        self.blur.as_ref().map(|b| b.radius)
    }

    /// Background colour in `OKlab` [L, a, b].
    #[must_use]
    pub fn background_oklab(&self) -> [f32; 3] {
        self.background.oklab
    }

    /// Total approximate byte cost of all shape genes in this genome.
    #[must_use]
    pub fn total_cost(&self) -> usize {
        self.shapes.iter().map(ShapeGene::cost).sum()
    }
}

impl ShapeGenome {
    /// Recombine via single-point crossover, with optional polygon angle crossover at the boundary.
    ///
    /// When the genes adjacent to the crossover point are both `Polygon`s, there is a 50%
    /// chance of replacing parent-A's boundary gene with an angle-blended hybrid of the two.
    /// Use this instead of `Genome::recombine` when image dimensions are available.
    #[must_use]
    pub fn recombine_configured(
        &self,
        other: &Self,
        rng: &mut impl Rng,
        config: &MutationConfig,
    ) -> Self {
        let mut a_sorted = self.shapes.clone();
        let mut b_sorted = other.shapes.clone();
        a_sorted.sort_unstable_by_key(ShapeGene::z_order);
        b_sorted.sort_unstable_by_key(ShapeGene::z_order);

        let min_len = a_sorted.len().min(b_sorted.len());
        let shapes = if min_len <= 1 {
            if a_sorted.len() >= b_sorted.len() {
                a_sorted
            } else {
                b_sorted
            }
        } else {
            let k = rng.random_range(1..min_len);
            let mut child: Vec<ShapeGene> = a_sorted[..k].to_vec();
            child.extend_from_slice(&b_sorted[k..]);

            // At the crossover boundary, try angle-based polygon crossover.
            // child[k-1] came from parent A; child[k] came from parent B.
            if rng.random_bool(0.5) {
                if let (ShapeGene::Polygon(pa), ShapeGene::Polygon(pb)) =
                    (&a_sorted[k - 1], &b_sorted[k])
                {
                    if let Some(mut hybrid) =
                        pa.angle_crossover(pb, rng, config.width, config.height, config.margin)
                    {
                        let za = a_sorted[k - 1].z_order();
                        let zb = b_sorted[k].z_order();
                        hybrid.z_order =
                            u32::midpoint(u32::from(za), u32::from(zb)) as u16;
                        child[k - 1] = ShapeGene::Polygon(hybrid);
                    }
                }
            }

            if child.is_empty() {
                a_sorted[..1].to_vec()
            } else {
                child
            }
        };

        let blur = match (&self.blur, &other.blur) {
            (Some(a), Some(b)) => Some(a.recombine(b, rng)),
            (Some(b), None) | (None, Some(b)) => {
                if rng.random_bool(0.5) { Some(b.clone()) } else { None }
            }
            (None, None) => None,
        };

        let background = self.background.recombine(&other.background, rng);

        Self { shapes, blur, background }
    }
}

impl Genome for ShapeGenome {
    fn fitness(&self, target: &[f32], width: u32, height: u32, scratch: &mut Vec<f32>) -> f32 {
        let bg = self.background_oklab();
        for pixel in scratch.chunks_exact_mut(3) {
            pixel.copy_from_slice(&bg);
        }
        draw_genes(scratch, width, height, &self.shapes);
        let blurred_opt = self.blur_radius().map(|r| apply_blur(scratch, width, height, r));
        let display: &[f32] = blurred_opt.as_deref().unwrap_or(scratch);
        let diff = compute_diff(target, display);
        // RMSE is in [0, ~1.0]; multiply by 100 to express as a percentage.
        diff as f32 * 100.0
    }

    #[allow(
        clippy::too_many_lines,
        reason = "mutation branches mirror the original mutate_shapes logic; factoring them out adds noise"
    )]
    fn mutate(
        &self,
        rng: &mut impl Rng,
        state: &AnnealingState,
        config: &MutationConfig,
    ) -> Self {
        const MUTATION_ATTEMPTS: usize = 10;

        let mut shapes = self.shapes.clone();
        let mut blur = self.blur.clone();
        let mut background = self.background.clone();

        // 10% chance: add a new gene (respects incremental cost cap)
        if rng.random_range(0..10u32) == 0 {
            let candidate = ShapeGene::random(rng, config);
            let current_cost: usize = shapes.iter().map(ShapeGene::cost).sum();
            if current_cost + candidate.cost() <= state.max_cost
                && current_cost + candidate.cost() <= state.max_cost_incremental
            {
                tracing::trace!(mutation = "add-gene", "adding shape gene");
                shapes.push(candidate);
                return Self { shapes, blur, background };
            }
            // Budget exceeded — fall through to normal mutation
        }

        // 5% chance: remove a gene (enforces non-empty invariant)
        if rng.random_range(0..20u32) == 0 && shapes.len() > 1 {
            tracing::trace!(mutation = "remove-gene", "removing shape gene");
            let idx = rng.random_range(0..shapes.len());
            shapes.remove(idx);
            return Self { shapes, blur, background };
        }

        // ~5% chance: split a large polygon into two colour-diverged halves
        if config.use_polygons && rng.random_range(0..20u32) == 0 {
            let poly_indices: Vec<usize> = shapes
                .iter()
                .enumerate()
                .filter(|(_, g)| {
                    matches!(g, ShapeGene::Polygon(p) if p.vertices.len() >= 6)
                })
                .map(|(i, _)| i)
                .collect();
            if !poly_indices.is_empty() {
                let idx = poly_indices[rng.random_range(0..poly_indices.len())];
                if let ShapeGene::Polygon(p) = &shapes[idx] {
                    if let Some((g_a, g_b)) =
                        p.split(rng, config.width, config.height, config.margin)
                    {
                        let rest_cost: usize = shapes
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != idx)
                            .map(|(_, g)| g.cost())
                            .sum();
                        let gene_a = ShapeGene::Polygon(g_a);
                        let gene_b = ShapeGene::Polygon(g_b);
                        let cost_ab = gene_a.cost() + gene_b.cost();
                        if rest_cost + cost_ab <= state.max_cost
                            && rest_cost + cost_ab <= state.max_cost_incremental
                        {
                            shapes[idx] = gene_a;
                            shapes.push(gene_b);
                            tracing::trace!(
                                mutation = "split-polygon",
                                "split polygon into two colour-diverged halves"
                            );
                            return Self { shapes, blur, background };
                        }
                    }
                }
            }
        }

        // 5% chance: swap z_orders of two genes (equivalent to reordering)
        if rng.random_range(0..20u32) == 0 && shapes.len() >= 2 {
            tracing::trace!(mutation = "swap-z-order", "swapping z_orders");
            let a = rng.random_range(0..shapes.len());
            let b = rng.random_range(0..shapes.len());
            if a != b {
                let za = shapes[a].z_order();
                let zb = shapes[b].z_order();
                shapes[a].set_z_order(zb);
                shapes[b].set_z_order(za);
            }
        }

        // ~5% chance: nudge or toggle blur
        if rng.random_range(0..20u32) == 0 {
            tracing::trace!(mutation = "nudge-blur", "nudging blur gene");
            blur = blur.map_or(Some(BlurGene { radius: 0.5 }), |b| {
                let mutated = b.mutate(rng, config);
                if mutated.radius < 0.1 { None } else { Some(mutated) }
            });
        }

        // ~5% chance: nudge background colour
        if rng.random_range(0..20u32) == 0 {
            tracing::trace!(mutation = "nudge-background", "nudging background gene");
            background = background.mutate(rng, config);
        }

        // Attempt individual shape gene mutations
        for _ in 0..MUTATION_ATTEMPTS {
            let idx = rng.random_range(0..shapes.len());
            if rng.random_range(0..1000u32) < config.mutation_rate {
                shapes[idx] = shapes[idx].mutate(rng, config);
            }
        }

        Self { shapes, blur, background }
    }

    fn recombine(&self, other: &Self, rng: &mut impl Rng) -> Self {
        // Single-point crossover in z-order space.
        let mut a_sorted = self.shapes.clone();
        let mut b_sorted = other.shapes.clone();
        a_sorted.sort_unstable_by_key(ShapeGene::z_order);
        b_sorted.sort_unstable_by_key(ShapeGene::z_order);

        let min_len = a_sorted.len().min(b_sorted.len());
        let shapes = if min_len <= 1 {
            // Can't split; pick the longer parent wholesale
            if a_sorted.len() >= b_sorted.len() {
                a_sorted
            } else {
                b_sorted
            }
        } else {
            let k = rng.random_range(1..min_len);
            let mut child: Vec<ShapeGene> = a_sorted[..k].to_vec();
            child.extend_from_slice(&b_sorted[k..]);
            if child.is_empty() {
                // Fallback: should never happen with k ≥ 1, but enforce invariant
                a_sorted[..1].to_vec()
            } else {
                child
            }
        };

        // Blur recombination
        let blur = match (&self.blur, &other.blur) {
            (Some(a), Some(b)) => Some(a.recombine(b, rng)),
            (Some(b), None) | (None, Some(b)) => {
                if rng.random_bool(0.5) { Some(b.clone()) } else { None }
            }
            (None, None) => None,
        };

        let background = self.background.recombine(&other.background, rng);

        Self { shapes, blur, background }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::gene::{CircleGene, PolygonGene, TRIANGLE_COST, TriangleGene};

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

    fn sample_genome() -> ShapeGenome {
        ShapeGenome {
            shapes: vec![
                ShapeGene::Triangle(TriangleGene {
                    x1: 0, y1: 0, x2: 50, y2: 0, x3: 25, y3: 50,
                    oklab: [0.6279, -0.2516, 0.0000],
                    alpha: 50,
                    z_order: 100,
                }),
                ShapeGene::Triangle(TriangleGene {
                    x1: 10, y1: 10, x2: 60, y2: 10, x3: 35, y3: 60,
                    oklab: [0.8664, -0.2334, 0.1795],
                    alpha: 50,
                    z_order: 200,
                }),
            ],
            blur: None,
            background: BackgroundGene::default(),
        }
    }

    #[test]
    fn mutate_never_empties_genome() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut genome = ShapeGenome {
            shapes: vec![ShapeGene::Triangle(TriangleGene {
                x1: 0, y1: 0, x2: 50, y2: 0, x3: 25, y3: 50,
                oklab: [0.5987, 0.0, 0.0],
                alpha: 50,
                z_order: 0,
            })],
            blur: None,
            background: BackgroundGene::default(),
        };
        let config = sample_config();
        let state = AnnealingState::new_from_shape_count(64, 1);
        for _ in 0..500 {
            genome = genome.mutate(&mut rng, &state, &config);
            assert!(!genome.shapes.is_empty(), "genome must never be empty");
        }
    }

    #[test]
    fn recombine_preserves_non_empty() {
        let mut rng = SmallRng::seed_from_u64(42);
        let a = sample_genome();
        let b = sample_genome();
        for _ in 0..100 {
            let child = a.recombine(&b, &mut rng);
            assert!(!child.shapes.is_empty(), "recombined genome must not be empty");
        }
    }

    #[test]
    fn split_mutation_increases_gene_count() {
        let mut rng = SmallRng::seed_from_u64(42);
        let polygon_config = MutationConfig {
            use_polygons: true,
            use_triangles: false,
            use_circles: false,
            ..sample_config()
        };
        // Use a large incremental budget so the split can succeed.
        let state = AnnealingState::new_from_shape_count(64, 10);
        let genome = ShapeGenome {
            shapes: vec![ShapeGene::Polygon(PolygonGene {
                vertices: vec![
                    (0, 0),
                    (50, 0),
                    (100, 0),
                    (100, 50),
                    (100, 100),
                    (50, 100),
                    (0, 100),
                    (0, 50),
                ],
                oklab: [0.5, 0.0, 0.0],
                alpha: 50,
                z_order: 0,
            })],
            blur: None,
            background: BackgroundGene::default(),
        };
        let mut found_split = false;
        for _ in 0..500 {
            let mutated = genome.mutate(&mut rng, &state, &polygon_config);
            if mutated.shapes.len() > 1 {
                found_split = true;
                break;
            }
        }
        assert!(found_split, "split mutation should fire within 500 attempts");
    }

    #[test]
    fn fitness_returns_non_negative() {
        let genome = sample_genome();
        let w = 10u32;
        let h = 10u32;
        let target = vec![0.0f32; (w * h * 3) as usize];
        let mut scratch = vec![0.0f32; (w * h * 3) as usize];
        let f = genome.fitness(&target, w, h, &mut scratch);
        assert!(f >= 0.0, "fitness must be non-negative: {f}");
    }

    #[test]
    fn sorted_genes_respects_z_order() {
        let genome = ShapeGenome {
            shapes: vec![
                ShapeGene::Circle(CircleGene {
                    cx: 20, cy: 20, radius: 5,
                    oklab: [0.5, 0.0, 0.0],
                    alpha: 50,
                    z_order: 200,
                }),
                ShapeGene::Circle(CircleGene {
                    cx: 10, cy: 10, radius: 5,
                    oklab: [0.5, 0.0, 0.0],
                    alpha: 50,
                    z_order: 100,
                }),
            ],
            blur: None,
            background: BackgroundGene::default(),
        };
        let sorted = genome.sorted_genes();
        assert_eq!(sorted[0].z_order(), 100);
        assert_eq!(sorted[1].z_order(), 200);
    }

    #[test]
    fn gene_costs_sum_correctly() {
        let genome = sample_genome();
        let expected = TRIANGLE_COST * 2;
        assert_eq!(genome.total_cost(), expected);
    }
}
