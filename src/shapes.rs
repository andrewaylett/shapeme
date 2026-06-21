use rand::Rng;
use serde::{Deserialize, Serialize};

pub(crate) const MINALPHA: u8 = 1;
pub(crate) const MAXALPHA: u8 = 100;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum Shape {
    Triangle {
        x1: i16,
        y1: i16,
        x2: i16,
        y2: i16,
        x3: i16,
        y3: i16,
        r: u8,
        g: u8,
        b: u8,
        alpha: u8,
    },
    Circle {
        cx: i16,
        cy: i16,
        radius: i16,
        r: u8,
        g: u8,
        b: u8,
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
        Shape::Circle { cx, cy, radius, .. } => {
            *cx = clamp_coord(*cx, -margin, w - 1 + margin);
            *cy = clamp_coord(*cy, -margin, h - 1 + margin);
            while *radius > 0
                && (*cx - *radius < -margin
                    || *cx + *radius >= w + margin
                    || *cy - *radius < -margin
                    || *cy + *radius >= h + margin)
            {
                *radius -= 1;
            }
        }
    }
}

fn select_shape_type(rng: &mut impl Rng, use_triangles: bool, use_circles: bool) -> bool {
    if use_triangles && use_circles {
        rng.random_bool(0.5)
    } else {
        use_triangles
    }
}

fn random_color(rng: &mut impl Rng) -> (u8, u8, u8, u8) {
    let r = rng.random::<u8>();
    let g = rng.random::<u8>();
    let b = rng.random::<u8>();
    let alpha = rand_between(rng, MINALPHA as i32, MAXALPHA as i32) as u8;
    (r, g, b, alpha)
}

pub(crate) fn random_shape(
    rng: &mut impl Rng,
    width: u32,
    height: u32,
    use_triangles: bool,
    use_circles: bool,
    margin: i16,
) -> Shape {
    let is_triangle = select_shape_type(rng, use_triangles, use_circles);
    let (r, g, b, alpha) = random_color(rng);
    let mut shape = if is_triangle {
        Shape::Triangle {
            x1: (rng.random::<u32>() % width) as i16,
            y1: (rng.random::<u32>() % height) as i16,
            x2: (rng.random::<u32>() % width) as i16,
            y2: (rng.random::<u32>() % height) as i16,
            x3: (rng.random::<u32>() % width) as i16,
            y3: (rng.random::<u32>() % height) as i16,
            r,
            g,
            b,
            alpha,
        }
    } else {
        Shape::Circle {
            cx: (rng.random::<u32>() % width) as i16,
            cy: (rng.random::<u32>() % height) as i16,
            radius: (rng.random::<u32>() % width) as i16,
            r,
            g,
            b,
            alpha,
        }
    };
    normalize(&mut shape, width, height, margin);
    shape
}

pub(crate) fn random_small_shape(
    rng: &mut impl Rng,
    width: u32,
    height: u32,
    delta: i32,
    use_triangles: bool,
    use_circles: bool,
    margin: i16,
) -> Shape {
    let is_triangle = select_shape_type(rng, use_triangles, use_circles);
    let x = (rng.random::<u32>() % width) as i32;
    let y = (rng.random::<u32>() % height) as i32;
    let (r, g, b, alpha) = random_color(rng);
    let mut shape = if is_triangle {
        Shape::Triangle {
            x1: (x + rand_between(rng, -delta, delta)) as i16,
            y1: (y + rand_between(rng, -delta, delta)) as i16,
            x2: (x + rand_between(rng, -delta, delta)) as i16,
            y2: (y + rand_between(rng, -delta, delta)) as i16,
            x3: (x + rand_between(rng, -delta, delta)) as i16,
            y3: (y + rand_between(rng, -delta, delta)) as i16,
            r,
            g,
            b,
            alpha,
        }
    } else {
        Shape::Circle {
            cx: x as i16,
            cy: y as i16,
            radius: rand_between(rng, 1, delta) as i16,
            r,
            g,
            b,
            alpha,
        }
    };
    normalize(&mut shape, width, height, margin);
    shape
}

pub(crate) fn mutate_shape(
    rng: &mut impl Rng,
    shape: &mut Shape,
    width: u32,
    height: u32,
    margin: i16,
) {
    let choice = rng.random_range(0..6u32);
    match shape {
        Shape::Triangle {
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            r,
            g,
            b,
            alpha,
        } => match choice {
            0 => {
                // randomize vertices
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
                *r = rng.random();
                *g = rng.random();
                *b = rng.random();
            }
            4 => {
                *r = (*r as i32 + rand_between(rng, -5, 5)).clamp(0, 255) as u8;
                *g = (*g as i32 + rand_between(rng, -5, 5)).clamp(0, 255) as u8;
                *b = (*b as i32 + rand_between(rng, -5, 5)).clamp(0, 255) as u8;
            }
            _ => {
                *alpha = rand_between(rng, MINALPHA as i32, MAXALPHA as i32) as u8;
            }
        },
        Shape::Circle {
            cx,
            cy,
            radius,
            r,
            g,
            b,
            alpha,
        } => match choice {
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
                *r = rng.random();
                *g = rng.random();
                *b = rng.random();
            }
            4 => {
                *r = (*r as i32 + rand_between(rng, -5, 5)).clamp(0, 255) as u8;
                *g = (*g as i32 + rand_between(rng, -5, 5)).clamp(0, 255) as u8;
                *b = (*b as i32 + rand_between(rng, -5, 5)).clamp(0, 255) as u8;
            }
            _ => {
                *alpha = rand_between(rng, MINALPHA as i32, MAXALPHA as i32) as u8;
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

    #[test]
    fn triangle_normalise_sorts_by_y() {
        let mut shape = Shape::Triangle {
            x1: 10,
            y1: 30,
            x2: 20,
            y2: 10,
            x3: 30,
            y3: 20,
            r: 0,
            g: 0,
            b: 0,
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
    fn circle_normalise_clamps_radius() {
        let mut shape = Shape::Circle {
            cx: 5,
            cy: 5,
            radius: 100,
            r: 0,
            g: 0,
            b: 0,
            alpha: 50,
        };
        normalize(&mut shape, 50, 50, 0);
        if let Shape::Circle { cx, cy, radius, .. } = shape {
            assert!(cx - radius >= 0);
            assert!(cx + radius < 50);
            assert!(cy - radius >= 0);
            assert!(cy + radius < 50);
        } else {
            panic!("expected Circle");
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
            r: 128,
            g: 128,
            b: 128,
            alpha: 50,
        };
        for _ in 0..1000 {
            mutate_shape(&mut rng, &mut shape, 100, 100, 0);
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
            r: 0,
            g: 0,
            b: 0,
            alpha: 50,
        };
        normalize(&mut shape, 100, 100, margin);
        if let Shape::Triangle { x1, y1, x2, y2, x3, y3, .. } = shape {
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
}
