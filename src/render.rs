use crate::shapes::Shape;

/// Draw a horizontal span into an RGB24 framebuffer with alpha blending.
/// Out-of-bounds y is silently ignored; x coordinates are clamped to the row.
#[allow(clippy::too_many_arguments)]
pub fn draw_hline(
    fb: &mut [u8],
    width: u32,
    height: u32,
    mut x1: i32,
    mut x2: i32,
    y: i32,
    r: u8,
    g: u8,
    b: u8,
    alpha: f32,
) {
    if y < 0 || y >= height as i32 {
        return;
    }
    if x1 > x2 {
        std::mem::swap(&mut x1, &mut x2);
    }
    let x1 = x1.max(0) as u32;
    let x2 = x2.min(width as i32 - 1) as u32;
    if x1 > x2 {
        return;
    }
    let ar = alpha * r as f32;
    let ag = alpha * g as f32;
    let ab = alpha * b as f32;
    let inv = 1.0 - alpha;
    let row = y as u32 * width;
    for x in x1..=x2 {
        let base = ((row + x) * 3) as usize;
        fb[base] = (ar + inv * fb[base] as f32) as u8;
        fb[base + 1] = (ag + inv * fb[base + 1] as f32) as u8;
        fb[base + 2] = (ab + inv * fb[base + 2] as f32) as u8;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn draw_circle(
    fb: &mut [u8],
    width: u32,
    height: u32,
    cx: i16,
    cy: i16,
    radius: i16,
    r: u8,
    g: u8,
    b: u8,
    alpha: f32,
) {
    let xc = cx as f64;
    let yc = cy as f64;
    let rad = radius as f64;
    let y_start = (yc - rad) as i32;
    let y_end = (yc + rad) as i32;
    for y in y_start..=y_end {
        let dy = y as f64 - yc;
        let dx = (rad * rad - dy * dy).sqrt();
        let x1 = (xc + dx).round() as i32;
        let x2 = (xc - dx).round() as i32;
        draw_hline(fb, width, height, x1, x2, y, r, g, b, alpha);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn draw_triangle(
    fb: &mut [u8],
    width: u32,
    height: u32,
    x1: i16,
    y1: i16,
    x2: i16,
    y2: i16,
    x3: i16,
    y3: i16,
    r: u8,
    g: u8,
    b: u8,
    alpha: f32,
) {
    // Vertices are pre-sorted by y (normalise guarantees this).
    let (ax, ay) = (x1 as f32, y1 as f32);
    let (bx, by) = (x2 as f32, y2 as f32);
    let (cx, cy) = (x3 as f32, y3 as f32);

    let dx1 = if by - ay > 0.0 {
        (bx - ax) / (by - ay)
    } else {
        bx - ax
    };
    let dx2 = if cy - ay > 0.0 {
        (cx - ax) / (cy - ay)
    } else {
        0.0
    };
    let dx3 = if cy - by > 0.0 {
        (cx - bx) / (cy - by)
    } else {
        0.0
    };

    let mut sx = ax;
    let mut sy = ay;
    let mut ex = ax;

    if dx1 > dx2 {
        while sy <= by {
            draw_hline(
                fb, width, height, sx as i32, ex as i32, sy as i32, r, g, b, alpha,
            );
            sy += 1.0;
            sx += dx2;
            ex += dx1;
        }
        ex = bx;
        while sy <= cy {
            draw_hline(
                fb, width, height, sx as i32, ex as i32, sy as i32, r, g, b, alpha,
            );
            sy += 1.0;
            sx += dx2;
            ex += dx3;
        }
    } else {
        while sy <= by {
            draw_hline(
                fb, width, height, sx as i32, ex as i32, sy as i32, r, g, b, alpha,
            );
            sy += 1.0;
            sx += dx1;
            ex += dx2;
        }
        sx = bx;
        sy = by + 1.0;
        while sy <= cy {
            draw_hline(
                fb, width, height, sx as i32, ex as i32, sy as i32, r, g, b, alpha,
            );
            sy += 1.0;
            sx += dx3;
            ex += dx2;
        }
    }
}

pub fn draw_shapes(fb: &mut [u8], width: u32, height: u32, shapes: &[Shape]) {
    for shape in shapes {
        match *shape {
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
            } => {
                draw_triangle(
                    fb,
                    width,
                    height,
                    x1,
                    y1,
                    x2,
                    y2,
                    x3,
                    y3,
                    r,
                    g,
                    b,
                    alpha as f32 / 100.0,
                );
            }
            Shape::Circle {
                cx,
                cy,
                radius,
                r,
                g,
                b,
                alpha,
            } => {
                draw_circle(
                    fb,
                    width,
                    height,
                    cx,
                    cy,
                    radius,
                    r,
                    g,
                    b,
                    alpha as f32 / 100.0,
                );
            }
        }
    }
}

/// Sum of per-pixel Euclidean RGB distances between two RGB24 buffers.
/// Max per-pixel distance is sqrt(255²×3) ≈ 441.67, capped at 442 in
/// percentage calculations. (The original C comment said 422, which was
/// a typo; the code correctly used 442.)
pub fn compute_diff(a: &[u8], b: &[u8]) -> i64 {
    assert_eq!(a.len(), b.len());
    let mut total: i64 = 0;
    let chunks = a.len() / 3;
    for i in 0..chunks {
        let base = i * 3;
        let dr = a[base] as i64 - b[base] as i64;
        let dg = a[base + 1] as i64 - b[base + 1] as i64;
        let db = a[base + 2] as i64 - b[base + 2] as i64;
        total += ((dr * dr + dg * dg + db * db) as f64).sqrt() as i64;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_diff_identical() {
        let buf = vec![128u8; 300];
        assert_eq!(compute_diff(&buf, &buf), 0);
    }

    #[test]
    fn compute_diff_max_distance() {
        let black = vec![0u8; 300];
        let white = vec![255u8; 300];
        let pixels = 300 / 3;
        let per_pixel = ((255i64 * 255 * 3) as f64).sqrt() as i64;
        assert_eq!(compute_diff(&black, &white), per_pixel * pixels as i64);
    }

    #[test]
    fn draw_hline_out_of_bounds_y_noop() {
        let mut fb = vec![0u8; 300]; // 10×10 RGB
        let before = fb.clone();
        draw_hline(&mut fb, 10, 10, 0, 9, -1, 255, 0, 0, 1.0);
        draw_hline(&mut fb, 10, 10, 0, 9, 10, 255, 0, 0, 1.0);
        assert_eq!(fb, before, "out-of-bounds y must not write to framebuffer");
    }
}
