use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::oklab;

pub(crate) const MINALPHA: u8 = 1;
pub(crate) const MAXALPHA: u8 = 100;

/// A geometric primitive drawn into the framebuffer with alpha blending.
///
/// Colour is stored as `OKlab` [L, a, b] f32 values for perceptually uniform mutation
/// and interpolation.  Alpha is still stored as an integer percentage (1–100) and
/// divided by 100.0 on render.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Shape {
    /// Filled triangle with vertices sorted by y-coordinate for the rasteriser.
    Triangle {
        /// First vertex x coordinate.
        x1: i16,
        /// First vertex y coordinate.
        y1: i16,
        /// Second vertex x coordinate.
        x2: i16,
        /// Second vertex y coordinate.
        y2: i16,
        /// Third vertex x coordinate.
        x3: i16,
        /// Third vertex y coordinate.
        y3: i16,
        /// Colour in `OKlab` [L, a, b].
        oklab: [f32; 3],
        /// Alpha as an integer percentage (1–100).
        alpha: u8,
    },
    /// Filled circle; centre is clamped to the margin-extended bounds, radius is unclamped.
    ///
    /// The rasteriser clips scanlines at the framebuffer edge, so a large radius that extends
    /// beyond the image is rendered correctly — only the visible arc is drawn.
    Circle {
        /// Centre x coordinate.
        cx: i16,
        /// Centre y coordinate.
        cy: i16,
        /// Radius in pixels.
        radius: i16,
        /// Colour in `OKlab` [L, a, b].
        oklab: [f32; 3],
        /// Alpha as an integer percentage (1–100).
        alpha: u8,
    },
    /// Filled n-gon (≥ 3 vertices) using scanline rasterisation.
    Polygon {
        /// Vertices in order; must contain at least 3 entries.
        vertices: Vec<(i16, i16)>,
        /// Colour in `OKlab` [L, a, b].
        oklab: [f32; 3],
        /// Alpha as an integer percentage (1–100).
        alpha: u8,
    },
}

fn rand_between(rng: &mut impl Rng, min: i32, max: i32) -> i32 {
    rng.random_range(min..=max)
}

fn clamp_coord(v: i16, min: i16, max: i16) -> i16 {
    v.clamp(min, max)
}

fn normalize(shape: &mut Shape, width: u32, height: u32, margin: i16) {
    let w = width as i16;
    let h = height as i16;
    match shape {
        Shape::Triangle {
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            ..
        } => {
            // Bubble-sort vertices by y, keeping (x,y) pairs together.
            loop {
                let mut swapped = false;
                if *y1 > *y2 {
                    std::mem::swap(y1, y2);
                    std::mem::swap(x1, x2);
                    swapped = true;
                }
                if *y2 > *y3 {
                    std::mem::swap(y2, y3);
                    std::mem::swap(x2, x3);
                    swapped = true;
                }
                if !swapped {
                    break;
                }
            }
            *x1 = clamp_coord(*x1, -margin, w - 1 + margin);
            *y1 = clamp_coord(*y1, -margin, h - 1 + margin);
            *x2 = clamp_coord(*x2, -margin, w - 1 + margin);
            *y2 = clamp_coord(*y2, -margin, h - 1 + margin);
            *x3 = clamp_coord(*x3, -margin, w - 1 + margin);
            *y3 = clamp_coord(*y3, -margin, h - 1 + margin);
        }
        Shape::Circle { cx, cy, .. } => {
            *cx = clamp_coord(*cx, -margin, w - 1 + margin);
            *cy = clamp_coord(*cy, -margin, h - 1 + margin);
        }
        Shape::Polygon { vertices, .. } => {
            for (vx, vy) in vertices.iter_mut() {
                *vx = clamp_coord(*vx, -margin, w - 1 + margin);
                *vy = clamp_coord(*vy, -margin, h - 1 + margin);
            }
            // Sort vertices by angle from centroid to eliminate self-intersecting edges.
            let n = vertices.len() as f32;
            let cx = vertices.iter().map(|(x, _)| *x as f32).sum::<f32>() / n;
            let cy = vertices.iter().map(|(_, y)| *y as f32).sum::<f32>() / n;
            vertices.sort_unstable_by(|(ax, ay), (bx, by)| {
                let a_angle = (*ay as f32 - cy).atan2(*ax as f32 - cx);
                let b_angle = (*by as f32 - cy).atan2(*bx as f32 - cx);
                a_angle.partial_cmp(&b_angle).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }
}

enum ShapeKind {
    Triangle,
    Circle,
    Polygon,
}

fn select_shape_type(
    rng: &mut impl Rng,
    use_triangles: bool,
    use_circles: bool,
    use_polygons: bool,
) -> ShapeKind {
    let mut kinds = Vec::with_capacity(3);
    if use_triangles {
        kinds.push(ShapeKind::Triangle);
    }
    if use_circles {
        kinds.push(ShapeKind::Circle);
    }
    if use_polygons {
        kinds.push(ShapeKind::Polygon);
    }
    debug_assert!(!kinds.is_empty(), "at least one shape type must be enabled");
    let idx = rng.random_range(0..kinds.len());
    kinds.remove(idx)
}

fn random_color(rng: &mut impl Rng) -> ([f32; 3], u8) {
    let r = rng.random::<u8>();
    let g = rng.random::<u8>();
    let b = rng.random::<u8>();
    let alpha = rand_between(rng, MINALPHA as i32, MAXALPHA as i32) as u8;
    (oklab::srgb_u8_to_oklab(r, g, b), alpha)
}

/// Nudge a single `OKlab` channel by a small random delta.
///
/// L is clamped to [0, 1]; a and b are left unclamped since out-of-gamut values
/// are silently clamped to the nearest sRGB colour at render/SVG time.
fn nudge_oklab(rng: &mut impl Rng, lab: [f32; 3]) -> [f32; 3] {
    let mut delta = || (rng.random::<f32>() * 2.0 - 1.0) * 0.02;
    [
        (lab[0] + delta()).clamp(0.0, 1.0),
        lab[1] + delta(),
        lab[2] + delta(),
    ]
}

/// Generate a random shape within the image bounds, extended by `margin` on all sides.
pub fn random_shape(
    rng: &mut impl Rng,
    width: u32,
    height: u32,
    use_triangles: bool,
    use_circles: bool,
    use_polygons: bool,
    margin: i16,
) -> Shape {
    let kind = select_shape_type(rng, use_triangles, use_circles, use_polygons);
    let (oklab, alpha) = random_color(rng);
    let mut shape = match kind {
        ShapeKind::Triangle => Shape::Triangle {
            x1: (rng.random::<u32>() % width) as i16,
            y1: (rng.random::<u32>() % height) as i16,
            x2: (rng.random::<u32>() % width) as i16,
            y2: (rng.random::<u32>() % height) as i16,
            x3: (rng.random::<u32>() % width) as i16,
            y3: (rng.random::<u32>() % height) as i16,
            oklab,
            alpha,
        },
        ShapeKind::Circle => Shape::Circle {
            cx: (rng.random::<u32>() % width) as i16,
            cy: (rng.random::<u32>() % height) as i16,
            radius: (rng.random::<u32>() % width) as i16,
            oklab,
            alpha,
        },
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
            Shape::Polygon {
                vertices,
                oklab,
                alpha,
            }
        }
    };
    normalize(&mut shape, width, height, margin);
    shape
}

/// Generate a random shape near a random anchor point, with all coordinates within `delta` pixels.
#[allow(
    clippy::too_many_arguments,
    reason = "shape configuration flags and spatial parameters are all necessary"
)]
pub(crate) fn random_small_shape(
    rng: &mut impl Rng,
    width: u32,
    height: u32,
    delta: i32,
    use_triangles: bool,
    use_circles: bool,
    use_polygons: bool,
    margin: i16,
) -> Shape {
    let kind = select_shape_type(rng, use_triangles, use_circles, use_polygons);
    let x = (rng.random::<u32>() % width) as i32;
    let y = (rng.random::<u32>() % height) as i32;
    let (oklab, alpha) = random_color(rng);
    let mut shape = match kind {
        ShapeKind::Triangle => Shape::Triangle {
            x1: (x + rand_between(rng, -delta, delta)) as i16,
            y1: (y + rand_between(rng, -delta, delta)) as i16,
            x2: (x + rand_between(rng, -delta, delta)) as i16,
            y2: (y + rand_between(rng, -delta, delta)) as i16,
            x3: (x + rand_between(rng, -delta, delta)) as i16,
            y3: (y + rand_between(rng, -delta, delta)) as i16,
            oklab,
            alpha,
        },
        ShapeKind::Circle => Shape::Circle {
            cx: x as i16,
            cy: y as i16,
            radius: rand_between(rng, 1, delta) as i16,
            oklab,
            alpha,
        },
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
            Shape::Polygon {
                vertices,
                oklab,
                alpha,
            }
        }
    };
    normalize(&mut shape, width, height, margin);
    shape
}

/// Apply a random mutation to a single shape in place.
pub(crate) fn mutate_shape(
    rng: &mut impl Rng,
    shape: &mut Shape,
    width: u32,
    height: u32,
    margin: i16,
    max_polygon_vertices: usize,
) {
    match shape {
        Shape::Triangle {
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            oklab,
            alpha,
        } => match rng.random_range(0..6u32) {
            0 => {
                *x1 = (rng.random::<u32>() % width) as i16;
                *y1 = (rng.random::<u32>() % height) as i16;
                *x2 = (rng.random::<u32>() % width) as i16;
                *y2 = (rng.random::<u32>() % height) as i16;
                *x3 = (rng.random::<u32>() % width) as i16;
                *y3 = (rng.random::<u32>() % height) as i16;
                normalize(shape, width, height, margin);
            }
            1 => {
                *x1 = x1.saturating_add(rand_between(rng, -20, 20) as i16);
                *y1 = y1.saturating_add(rand_between(rng, -20, 20) as i16);
                *x2 = x2.saturating_add(rand_between(rng, -20, 20) as i16);
                *y2 = y2.saturating_add(rand_between(rng, -20, 20) as i16);
                *x3 = x3.saturating_add(rand_between(rng, -20, 20) as i16);
                *y3 = y3.saturating_add(rand_between(rng, -20, 20) as i16);
                normalize(shape, width, height, margin);
            }
            2 => {
                *x1 = x1.saturating_add(rand_between(rng, -5, 5) as i16);
                *y1 = y1.saturating_add(rand_between(rng, -5, 5) as i16);
                *x2 = x2.saturating_add(rand_between(rng, -5, 5) as i16);
                *y2 = y2.saturating_add(rand_between(rng, -5, 5) as i16);
                *x3 = x3.saturating_add(rand_between(rng, -5, 5) as i16);
                *y3 = y3.saturating_add(rand_between(rng, -5, 5) as i16);
                normalize(shape, width, height, margin);
            }
            3 => {
                let (new_oklab, _) = random_color(rng);
                *oklab = new_oklab;
            }
            4 => {
                *oklab = nudge_oklab(rng, *oklab);
            }
            _ => {
                *alpha = rand_between(rng, MINALPHA as i32, MAXALPHA as i32) as u8;
            }
        },
        Shape::Circle {
            cx,
            cy,
            radius,
            oklab,
            alpha,
        } => match rng.random_range(0..6u32) {
            0 => {
                *cx = (rng.random::<u32>() % width) as i16;
                *cy = (rng.random::<u32>() % height) as i16;
                *radius = (rng.random::<u32>() % width) as i16;
                normalize(shape, width, height, margin);
            }
            1 => {
                *cx = cx.saturating_add(rand_between(rng, -20, 20) as i16);
                *cy = cy.saturating_add(rand_between(rng, -20, 20) as i16);
                *radius = radius.saturating_add(rand_between(rng, -20, 20) as i16);
                normalize(shape, width, height, margin);
            }
            2 => {
                *cx = cx.saturating_add(rand_between(rng, -5, 5) as i16);
                *cy = cy.saturating_add(rand_between(rng, -5, 5) as i16);
                *radius = radius.saturating_add(rand_between(rng, -5, 5) as i16);
                normalize(shape, width, height, margin);
            }
            3 => {
                let (new_oklab, _) = random_color(rng);
                *oklab = new_oklab;
            }
            4 => {
                *oklab = nudge_oklab(rng, *oklab);
            }
            _ => {
                *alpha = rand_between(rng, MINALPHA as i32, MAXALPHA as i32) as u8;
            }
        },
        Shape::Polygon {
            vertices,
            oklab,
            alpha,
        } => match rng.random_range(0..8u32) {
            0 => {
                let n = vertices.len();
                *vertices = (0..n)
                    .map(|_| {
                        (
                            (rng.random::<u32>() % width) as i16,
                            (rng.random::<u32>() % height) as i16,
                        )
                    })
                    .collect();
                normalize(shape, width, height, margin);
            }
            1 => {
                for (vx, vy) in vertices.iter_mut() {
                    *vx = vx.saturating_add(rand_between(rng, -20, 20) as i16);
                    *vy = vy.saturating_add(rand_between(rng, -20, 20) as i16);
                }
                normalize(shape, width, height, margin);
            }
            2 => {
                for (vx, vy) in vertices.iter_mut() {
                    *vx = vx.saturating_add(rand_between(rng, -5, 5) as i16);
                    *vy = vy.saturating_add(rand_between(rng, -5, 5) as i16);
                }
                normalize(shape, width, height, margin);
            }
            3 => {
                let (new_oklab, _) = random_color(rng);
                *oklab = new_oklab;
            }
            4 => {
                *oklab = nudge_oklab(rng, *oklab);
            }
            5 => {
                *alpha = rand_between(rng, MINALPHA as i32, MAXALPHA as i32) as u8;
            }
            6 => {
                // Split a random edge by inserting its midpoint (nudged ±20 px).
                // Falls back to a small nudge when the vertex cap has been reached.
                if vertices.len() < max_polygon_vertices {
                    let n = vertices.len();
                    let edge = rng.random_range(0..n);
                    let (x1, y1) = vertices[edge];
                    let (x2, y2) = vertices[(edge + 1) % n];
                    let mx = ((x1 as i32 + x2 as i32) / 2 + rand_between(rng, -20, 20))
                        .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    let my = ((y1 as i32 + y2 as i32) / 2 + rand_between(rng, -20, 20))
                        .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    vertices.insert(edge + 1, (mx, my));
                } else {
                    for (vx, vy) in vertices.iter_mut() {
                        *vx = vx.saturating_add(rand_between(rng, -5, 5) as i16);
                        *vy = vy.saturating_add(rand_between(rng, -5, 5) as i16);
                    }
                }
                normalize(shape, width, height, margin);
            }
            _ => {
                // Remove a random vertex; fall through to a small nudge if already at minimum.
                if vertices.len() > 3 {
                    let idx = rng.random_range(0..vertices.len());
                    vertices.remove(idx);
                } else {
                    for (vx, vy) in vertices.iter_mut() {
                        *vx = vx.saturating_add(rand_between(rng, -5, 5) as i16);
                        *vy = vy.saturating_add(rand_between(rng, -5, 5) as i16);
                    }
                }
                normalize(shape, width, height, margin);
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn seeded() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    fn black_oklab() -> [f32; 3] {
        [0.0, 0.0, 0.0]
    }

    #[test]
    fn triangle_normalise_sorts_by_y() {
        let mut shape = Shape::Triangle {
            x1: 10,
            y1: 30,
            x2: 20,
            y2: 10,
            x3: 30,
            y3: 20,
            oklab: black_oklab(),
            alpha: 50,
        };
        normalize(&mut shape, 100, 100, 0);
        if let Shape::Triangle { y1, y2, y3, .. } = shape {
            assert!(
                y1 <= y2 && y2 <= y3,
                "y values must be non-decreasing after normalise"
            );
        } else {
            panic!("expected Triangle");
        }
    }

    #[test]
    fn circle_normalise_clamps_centre_only() {
        let mut shape = Shape::Circle {
            cx: -200,
            cy: 300,
            radius: 100,
            oklab: black_oklab(),
            alpha: 50,
        };
        normalize(&mut shape, 50, 50, 0);
        if let Shape::Circle { cx, cy, radius, .. } = shape {
            assert_eq!(cx, 0, "cx should be clamped to 0");
            assert_eq!(cy, 49, "cy should be clamped to h-1");
            assert_eq!(radius, 100, "radius must be unchanged");
        } else {
            panic!("expected Circle");
        }
    }

    #[test]
    fn polygon_normalise_clamps_vertices() {
        let mut shape = Shape::Polygon {
            vertices: vec![(-200, 300), (100, 100), (50, -100)],
            oklab: black_oklab(),
            alpha: 50,
        };
        normalize(&mut shape, 50, 50, 0);
        if let Shape::Polygon { vertices, .. } = shape {
            for (vx, vy) in &vertices {
                assert!(*vx >= 0 && *vx <= 49, "vx {vx} out of bounds [0,49]");
                assert!(*vy >= 0 && *vy <= 49, "vy {vy} out of bounds [0,49]");
            }
        } else {
            panic!("expected Polygon");
        }
    }

    #[test]
    fn alpha_mutation_stays_in_bounds() {
        let mut rng = seeded();
        let mut shape = Shape::Triangle {
            x1: 10,
            y1: 10,
            x2: 20,
            y2: 20,
            x3: 30,
            y3: 30,
            oklab: black_oklab(),
            alpha: 50,
        };
        for _ in 0..1000 {
            mutate_shape(&mut rng, &mut shape, 100, 100, 0, 100);
            if let Shape::Triangle { alpha, .. } = shape {
                assert!(
                    (MINALPHA..=MAXALPHA).contains(&alpha),
                    "alpha {alpha} out of bounds"
                );
            }
        }
    }

    #[test]
    fn normalise_with_margin_allows_coords_outside_image() {
        let margin: i16 = 10;
        // A triangle vertex just outside the old bound should be preserved when margin is set.
        let mut shape = Shape::Triangle {
            x1: -5,
            y1: -5,
            x2: 104,
            y2: 5,
            x3: 5,
            y3: 104,
            oklab: black_oklab(),
            alpha: 50,
        };
        normalize(&mut shape, 100, 100, margin);
        if let Shape::Triangle {
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            ..
        } = shape
        {
            assert_eq!(x1, -5, "x1 within margin should be unchanged");
            assert_eq!(y1, -5, "y1 within margin should be unchanged");
            assert_eq!(x2, 104, "x2 within margin should be unchanged");
            assert_eq!(x3, 5, "x3 in-bounds should be unchanged");
            assert_eq!(y3, 104, "y3 within margin should be unchanged");
            // y-sort: -5 <= 5 <= 104 after the sort
            assert!(y1 <= y2 && y2 <= y3, "y values must be non-decreasing");
        } else {
            panic!("expected Triangle");
        }
    }

    #[test]
    fn polygon_mutation_vertex_count_minimum() {
        let mut rng = seeded();
        let mut shape = Shape::Polygon {
            vertices: vec![(0, 0), (50, 0), (25, 50)],
            oklab: black_oklab(),
            alpha: 50,
        };
        for _ in 0..1000 {
            mutate_shape(&mut rng, &mut shape, 100, 100, 0, 100);
            if let Shape::Polygon { ref vertices, .. } = shape {
                assert!(
                    vertices.len() >= 3,
                    "polygon must never have fewer than 3 vertices, got {}",
                    vertices.len()
                );
            } else {
                panic!("shape changed type unexpectedly");
            }
        }
    }

    #[test]
    fn polygon_mutation_split_increases_count() {
        let mut rng = seeded();
        let mut shape = Shape::Polygon {
            vertices: vec![(0, 0), (50, 0), (25, 50)],
            oklab: black_oklab(),
            alpha: 50,
        };
        let mut saw_split = false;
        for _ in 0..1000 {
            let before = if let Shape::Polygon { ref vertices, .. } = shape {
                vertices.len()
            } else {
                panic!("shape changed type unexpectedly");
            };
            mutate_shape(&mut rng, &mut shape, 100, 100, 0, 100);
            let after = if let Shape::Polygon { ref vertices, .. } = shape {
                vertices.len()
            } else {
                panic!("shape changed type unexpectedly");
            };
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
        let mut shape = Shape::Polygon {
            vertices: vec![(0, 0), (50, 0), (25, 50)],
            oklab: black_oklab(),
            alpha: 50,
        };
        for _ in 0..5000 {
            mutate_shape(&mut rng, &mut shape, 100, 100, 0, cap);
            if let Shape::Polygon { ref vertices, .. } = shape {
                assert!(
                    vertices.len() <= cap,
                    "vertex count {} exceeds cap {cap}",
                    vertices.len()
                );
            }
        }
    }

    #[test]
    fn polygon_normalise_sorts_by_angle() {
        // Vertices in clockwise order (wrong); normalise should sort them by angle.
        let mut shape = Shape::Polygon {
            vertices: vec![(50, 0), (100, 50), (50, 100), (0, 50)],
            oklab: black_oklab(),
            alpha: 50,
        };
        normalize(&mut shape, 200, 200, 0);
        if let Shape::Polygon { vertices, .. } = shape {
            // After sort the angles from centroid (50,50) should be non-decreasing.
            let cx = 50.0f32;
            let cy = 50.0f32;
            let angles: Vec<f32> = vertices
                .iter()
                .map(|(x, y)| (*y as f32 - cy).atan2(*x as f32 - cx))
                .collect();
            for w in angles.windows(2) {
                assert!(
                    w[0] <= w[1],
                    "angles not sorted: {:?} > {:?}",
                    w[0],
                    w[1]
                );
            }
        } else {
            panic!("expected Polygon");
        }
    }
}
