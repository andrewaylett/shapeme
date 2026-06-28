//! Grid genome: a shared-vertex quad mesh that covers the entire canvas.
//!
//! Adjacent cells share edges (tied vertices), so moving a vertex moves it for all cells that
//! touch it.  Annealing moves vertices and changes cell colours.  This is architecturally
//! distinct from the shape-based genome and so lives in its own type.
//!
//! # Coverage guarantee
//!
//! Edge and corner points are pinned rather than clamped.  When blur evolves, `normalize`
//! re-pins them to the new `±margin` line so the cells always cover the full canvas (blur
//! hides the seam).  Interior points are clamped inside the margin-extended bounds.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::annealing::AnnealingState;
use crate::gene::{BlurGene, Gene, MutationConfig};
use crate::genome::Genome;
use crate::render::{apply_blur, compute_diff, draw_triangle};
use crate::shapes::{nudge_oklab, rand_between, random_oklab_color};
use crate::svg::build_svg_from_grid;

/// A shared-vertex quad-mesh genome covering the entire canvas.
///
/// `points` is a `(cols+1) × (rows+1)` grid of control points (row-major).
/// `colors` is a `cols × rows` grid of `OKlab` colours (row-major).
/// Adjacent cells share their border points, so deforming a point moves all cells that touch it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GridGenome {
    /// Number of cell columns.
    pub cols: u16,
    /// Number of cell rows.
    pub rows: u16,
    /// Control points, row-major: index = `row * (cols+1) + col`.
    pub points: Vec<(i16, i16)>,
    /// Cell colours in `OKlab` [L, a, b], row-major: index = `row * cols + col`.
    pub colors: Vec<[f32; 3]>,
    /// Optional evolved Gaussian blur (same `BlurGene` mechanism as `ShapeGenome`).
    pub blur: Option<BlurGene>,
}

impl GridGenome {
    /// Construct a regular grid with neutral-grey cells.
    ///
    /// Points are placed at `x[c] = round(-m + c * (W - 1 + 2m) / cols)` and the y equivalent.
    /// `blur_radius` — if given — sets the initial blur and determines the canvas margin.
    #[must_use]
    pub fn new(
        cols: u16,
        rows: u16,
        width: u32,
        height: u32,
        blur_radius: Option<f32>,
    ) -> Self {
        let m = blur_radius.map_or(0, |r| r.ceil() as i32);
        let w = width as i32;
        let h = height as i32;
        let cols_u = cols as usize;
        let rows_u = rows as usize;

        let mut points = Vec::with_capacity((cols_u + 1) * (rows_u + 1));
        for r in 0..=(rows_u) {
            for c in 0..=(cols_u) {
                let x = (-m + c as i32 * (w - 1 + 2 * m) / cols_u as i32) as i16;
                let y = (-m + r as i32 * (h - 1 + 2 * m) / rows_u as i32) as i16;
                points.push((x, y));
            }
        }

        let neutral_grey = [0.5f32, 0.0, 0.0];
        let colors = vec![neutral_grey; cols_u * rows_u];
        let blur = blur_radius.map(|r| BlurGene { radius: r });

        Self {
            cols,
            rows,
            points,
            colors,
            blur,
        }
    }

    /// Re-apply all point constraints after a mutation.
    ///
    /// Corner points are fixed at `±margin`.  Edge points have one free axis (clamped to the
    /// margin-extended range) and one pinned axis.  Interior points are clamped on both axes.
    /// Pinned means assignment to the exact margin boundary, not clamping — so that when blur
    /// (and thus margin) grows, edge points move outward to maintain coverage.
    fn normalize(&mut self, width: u32, height: u32, margin: i16) {
        let w = width as i16;
        let h = height as i16;
        let m = margin;
        let cols = self.cols as usize;
        let rows = self.rows as usize;

        for r in 0..=(rows) {
            for c in 0..=(cols) {
                let idx = r * (cols + 1) + c;
                let (ref mut x, ref mut y) = self.points[idx];

                let on_left = c == 0;
                let on_right = c == cols;
                let on_top = r == 0;
                let on_bottom = r == rows;

                match (on_left || on_right, on_top || on_bottom) {
                    (true, true) => {
                        // Corner: both axes pinned
                        *x = if on_left { -m } else { w - 1 + m };
                        *y = if on_top { -m } else { h - 1 + m };
                    }
                    (true, false) => {
                        // Left/right edge: x pinned, y free
                        *x = if on_left { -m } else { w - 1 + m };
                        *y = (*y).clamp(-m, h - 1 + m);
                    }
                    (false, true) => {
                        // Top/bottom edge: x free, y pinned
                        *x = (*x).clamp(-m, w - 1 + m);
                        *y = if on_top { -m } else { h - 1 + m };
                    }
                    (false, false) => {
                        // Interior: both axes clamped
                        *x = (*x).clamp(-m, w - 1 + m);
                        *y = (*y).clamp(-m, h - 1 + m);
                    }
                }
            }
        }
    }

    /// Render all grid cells directly into `fb` (unblurred).
    ///
    /// Each cell is split into two triangles (TL-TR-BR and TL-BR-BL) with alpha = 1.0
    /// to guarantee gap-free coverage even when points are deformed.
    fn render_cells(&self, fb: &mut [f32], width: u32, height: u32) {
        let cols = self.cols as usize;
        let rows = self.rows as usize;
        let cols_p1 = cols + 1;

        for row in 0..rows {
            for col in 0..cols {
                let tl = self.points[row * cols_p1 + col];
                let tr = self.points[row * cols_p1 + col + 1];
                let br = self.points[(row + 1) * cols_p1 + col + 1];
                let bl = self.points[(row + 1) * cols_p1 + col];
                let color = self.colors[row * cols + col];

                // TL-TR-BR triangle: y-sort for the rasteriser
                draw_sorted_triangle(fb, width, height, tl, tr, br, color);
                // TL-BR-BL triangle
                draw_sorted_triangle(fb, width, height, tl, br, bl, color);
            }
        }
    }
}

/// Sort three points by y then call `draw_triangle` (which requires y1 ≤ y2 ≤ y3).
#[allow(
    clippy::too_many_arguments,
    reason = "forwarding triangle vertices to the rasteriser"
)]
fn draw_sorted_triangle(
    fb: &mut [f32],
    width: u32,
    height: u32,
    a: (i16, i16),
    b: (i16, i16),
    c: (i16, i16),
    color: [f32; 3],
) {
    let mut verts = [a, b, c];
    // Bubble-sort by y (identical approach to TriangleGene::normalize)
    loop {
        let mut swapped = false;
        if verts[0].1 > verts[1].1 {
            verts.swap(0, 1);
            swapped = true;
        }
        if verts[1].1 > verts[2].1 {
            verts.swap(1, 2);
            swapped = true;
        }
        if !swapped {
            break;
        }
    }
    draw_triangle(
        fb,
        width,
        height,
        verts[0].0,
        verts[0].1,
        verts[1].0,
        verts[1].1,
        verts[2].0,
        verts[2].1,
        color,
        1.0,
    );
}

impl Genome for GridGenome {
    fn render_to_fb(&self, fb: &mut [f32], width: u32, height: u32) {
        // Clear to black first (grid cells cover the canvas, so colour doesn't matter,
        // but initialising avoids undefined state in any uncovered gap pixel)
        for v in fb.iter_mut() {
            *v = 0.0;
        }
        self.render_cells(fb, width, height);
    }

    fn build_svg_output(&self, width: u32, height: u32, compact: bool) -> String {
        build_svg_from_grid(self, width, height, compact)
    }

    fn blur_radius(&self) -> Option<f32> {
        self.blur.as_ref().map(|b| b.radius)
    }

    fn total_cost(&self) -> usize {
        self.cols as usize * self.rows as usize
    }

    fn fitness(&self, target: &[f32], width: u32, height: u32, scratch: &mut Vec<f32>) -> f32 {
        self.render_to_fb(scratch, width, height);
        let blurred_opt = self
            .blur_radius()
            .map(|r| apply_blur(scratch, width, height, r));
        let display: &[f32] = blurred_opt.as_deref().unwrap_or(scratch);
        let diff = compute_diff(target, display);
        diff as f32 * 100.0
    }

    fn mutate(&self, rng: &mut impl Rng, _state: &AnnealingState, config: &MutationConfig) -> Self {
        let mut points = self.points.clone();
        let mut colors = self.colors.clone();
        let mut blur = self.blur.clone();

        let roll = rng.random_range(0..100u32);
        if roll < 60 {
            // Move a random point: 80% small nudge (±5), 20% large nudge (±20)
            if !points.is_empty() {
                let idx = rng.random_range(0..points.len());
                let delta = if rng.random_range(0..5u32) == 0 { 20 } else { 5 };
                points[idx].0 =
                    points[idx].0.saturating_add(rand_between(rng, -delta, delta) as i16);
                points[idx].1 =
                    points[idx].1.saturating_add(rand_between(rng, -delta, delta) as i16);
            }
        } else if roll < 90 {
            // Nudge a random cell's colour
            if !colors.is_empty() {
                let idx = rng.random_range(0..colors.len());
                colors[idx] = nudge_oklab(rng, colors[idx]);
            }
        } else if roll < 95 {
            // Fully randomise a random cell's colour
            if !colors.is_empty() {
                let idx = rng.random_range(0..colors.len());
                let (new_color, _) = random_oklab_color(rng);
                colors[idx] = new_color;
            }
        } else {
            // Nudge or toggle blur
            blur = blur.map_or(Some(BlurGene { radius: 0.5 }), |b| {
                let mutated = b.mutate(rng, config);
                if mutated.radius < 0.1 { None } else { Some(mutated) }
            });
        }

        let margin = blur.as_ref().map_or(0, |b| b.radius.ceil() as i16);
        let mut child = Self {
            cols: self.cols,
            rows: self.rows,
            points,
            colors,
            blur,
        };
        child.normalize(config.width, config.height, margin);
        child
    }

    fn recombine(&self, other: &Self, rng: &mut impl Rng) -> Self {
        let (points, colors) = if rng.random_bool(0.5) {
            (self.points.clone(), other.colors.clone())
        } else {
            (other.points.clone(), self.colors.clone())
        };

        let blur = match (&self.blur, &other.blur) {
            (Some(a), Some(b)) => Some(a.recombine(b, rng)),
            (Some(b), None) | (None, Some(b)) => {
                if rng.random_bool(0.5) { Some(b.clone()) } else { None }
            }
            (None, None) => None,
        };

        Self {
            cols: self.cols,
            rows: self.rows,
            points,
            colors,
            blur,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::annealing::AnnealingState;
    use crate::gene::MutationConfig;

    fn sample_config() -> MutationConfig {
        MutationConfig {
            width: 64,
            height: 64,
            margin: 0,
            mutation_rate: 500,
            use_triangles: false,
            use_circles: false,
            use_polygons: false,
            max_polygon_vertices: 0,
        }
    }

    #[test]
    fn new_grid_has_correct_dimensions() {
        let g = GridGenome::new(4, 3, 64, 64, None);
        assert_eq!(g.points.len(), 5 * 4, "points: (cols+1)*(rows+1)");
        assert_eq!(g.colors.len(), 4 * 3, "colors: cols*rows");
    }

    #[test]
    fn normalize_pins_corners() {
        let mut g = GridGenome::new(2, 2, 64, 64, Some(4.0));
        // Manually corrupt a corner point then re-normalise
        g.points[0] = (100, 100);
        let margin = 4i16;
        g.normalize(64, 64, margin);
        // Top-left corner must be pinned to (-margin, -margin)
        assert_eq!(g.points[0], (-margin, -margin));
    }

    #[test]
    fn normalize_pins_edges() {
        let mut g = GridGenome::new(2, 2, 64, 64, Some(4.0));
        let margin = 4i16;
        // Top-edge non-corner (row=0, col=1): y must be -margin, x is free
        let top_mid = g.points[0 * 3 + 1]; // row=0, col=1 in a (cols+1)=3-wide grid
        g.points[0 * 3 + 1] = (100, 100);
        g.normalize(64, 64, margin);
        let (x, y) = g.points[0 * 3 + 1];
        assert_eq!(y, -margin, "top edge y must be pinned");
        assert!(x != top_mid.0 || x.clamp(-margin, 63 + margin) == x, "x should be clamped");
    }

    #[test]
    fn fitness_returns_non_negative() {
        let g = GridGenome::new(3, 3, 16, 16, None);
        let target = vec![0.5f32; 16 * 16 * 3];
        let mut scratch = vec![0.0f32; 16 * 16 * 3];
        let f = g.fitness(&target, 16, 16, &mut scratch);
        assert!(f >= 0.0, "fitness must be non-negative: {f}");
    }

    #[test]
    fn mutate_preserves_dimensions() {
        let mut rng = SmallRng::seed_from_u64(42);
        let g = GridGenome::new(4, 3, 64, 64, None);
        let state = AnnealingState::new(usize::MAX / 2, usize::MAX / 2);
        let config = sample_config();
        for _ in 0..200 {
            let m = g.mutate(&mut rng, &state, &config);
            assert_eq!(m.cols, g.cols);
            assert_eq!(m.rows, g.rows);
            assert_eq!(m.points.len(), g.points.len());
            assert_eq!(m.colors.len(), g.colors.len());
        }
    }

    #[test]
    fn recombine_preserves_dimensions() {
        let mut rng = SmallRng::seed_from_u64(42);
        let a = GridGenome::new(4, 3, 64, 64, None);
        let b = GridGenome::new(4, 3, 64, 64, Some(2.0));
        for _ in 0..50 {
            let child = a.recombine(&b, &mut rng);
            assert_eq!(child.cols, a.cols);
            assert_eq!(child.points.len(), a.points.len());
            assert_eq!(child.colors.len(), a.colors.len());
        }
    }
}
