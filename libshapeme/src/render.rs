use image::{ImageBuffer, Rgb, imageops};

use crate::shapes::Shape;

/// Draw a horizontal span into an RGB24 framebuffer with alpha blending.
/// Out-of-bounds y is silently ignored; x coordinates are clamped to the row.
#[allow(
    clippy::too_many_arguments,
    reason = "pixel coordinate + colour + alpha parameters are unavoidable in a rasterizer"
)]
fn draw_hline(
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
    let x1 = x1.max(0);
    let x2 = x2.min(width as i32 - 1);
    if x1 > x2 {
        return;
    }
    let x1 = x1 as u32;
    let x2 = x2 as u32;
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

#[allow(
    clippy::too_many_arguments,
    reason = "pixel coordinate + colour + alpha parameters are unavoidable in a rasterizer"
)]
fn draw_circle(
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

#[allow(
    clippy::too_many_arguments,
    reason = "pixel coordinate + colour + alpha parameters are unavoidable in a rasterizer"
)]
fn draw_triangle(
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

#[allow(
    clippy::too_many_arguments,
    reason = "pixel coordinate + colour + alpha parameters are unavoidable in a rasterizer"
)]
fn draw_polygon(
    fb: &mut [u8],
    width: u32,
    height: u32,
    vertices: &[(i16, i16)],
    r: u8,
    g: u8,
    b: u8,
    alpha: f32,
) {
    let n = vertices.len();
    if n < 3 {
        return;
    }
    let y_min = vertices.iter().map(|v| v.1).min().expect("vertices is non-empty") as i32;
    let y_max = vertices.iter().map(|v| v.1).max().expect("vertices is non-empty") as i32;
    for y in y_min..=y_max {
        let mut xs: Vec<i32> = Vec::new();
        for i in 0..n {
            let (x1, y1) = (vertices[i].0 as i32, vertices[i].1 as i32);
            let (x2, y2) = (vertices[(i + 1) % n].0 as i32, vertices[(i + 1) % n].1 as i32);
            // Include the lower endpoint, exclude the upper — avoids double-counting at vertices.
            if (y1 <= y && y < y2) || (y2 <= y && y < y1) {
                let x = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
                xs.push(x);
            }
        }
        xs.sort_unstable();
        for pair in xs.chunks(2) {
            if pair.len() == 2 {
                draw_hline(fb, width, height, pair[0], pair[1], y, r, g, b, alpha);
            }
        }
    }
}

/// Rasterise a slice of shapes into an RGB24 framebuffer.
pub fn draw_shapes(fb: &mut [u8], width: u32, height: u32, shapes: &[Shape]) {
    for shape in shapes {
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
            } => {
                draw_triangle(
                    fb,
                    width,
                    height,
                    *x1,
                    *y1,
                    *x2,
                    *y2,
                    *x3,
                    *y3,
                    *r,
                    *g,
                    *b,
                    *alpha as f32 / 100.0,
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
                    *cx,
                    *cy,
                    *radius,
                    *r,
                    *g,
                    *b,
                    *alpha as f32 / 100.0,
                );
            }
            Shape::Polygon {
                vertices,
                r,
                g,
                b,
                alpha,
            } => {
                draw_polygon(
                    fb,
                    width,
                    height,
                    vertices,
                    *r,
                    *g,
                    *b,
                    *alpha as f32 / 100.0,
                );
            }
        }
    }
}

/// Apply a Gaussian blur (sigma = `radius`) to an RGB24 framebuffer.
#[must_use]
pub fn apply_blur(fb: &[u8], width: u32, height: u32, radius: f32) -> Vec<u8> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, fb.to_vec())
        .expect("framebuffer dimensions must match pixel data");
    imageops::blur(&img, radius).into_raw()
}

/// Sum of per-pixel Euclidean RGB distances between two RGB24 buffers.
///
/// Max per-pixel distance is sqrt(255²×3) ≈ 441.67, capped at 442 in
/// percentage calculations. (The original C comment said 422, which was
/// a typo; the code correctly used 442.)
#[must_use]
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

/// Downsample `pixels` so that `max(width, height)` ≤ `max_dim` using Lanczos3 resampling.
/// Returns unchanged if already within the limit.
#[must_use]
pub fn scale_image(pixels: Vec<u8>, width: u32, height: u32, max_dim: u32) -> (Vec<u8>, u32, u32) {
    if width.max(height) <= max_dim {
        return (pixels, width, height);
    }
    let (new_w, new_h) = if width >= height {
        // Divides by width (>= height and > 0 since max(w,h) > max_dim ≥ 0)
        let new_h = (u64::from(height) * u64::from(max_dim) / u64::from(width)).max(1);
        (max_dim, u32::try_from(new_h).unwrap_or(max_dim))
    } else {
        let new_w = (u64::from(width) * u64::from(max_dim) / u64::from(height)).max(1);
        (u32::try_from(new_w).unwrap_or(max_dim), max_dim)
    };
    tracing::debug!(
        original_width = width,
        original_height = height,
        scaled_width = new_w,
        scaled_height = new_h,
        "image scaled"
    );
    let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width, height, pixels)
        .expect("pixel buffer must match declared dimensions");
    let resized =
        image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Lanczos3);
    (resized.into_raw(), new_w, new_h)
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
    fn apply_blur_preserves_buffer_length() {
        let pixels = vec![128u8; 30 * 30 * 3];
        let blurred = apply_blur(&pixels, 30, 30, 2.0);
        assert_eq!(blurred.len(), pixels.len());
    }

    #[test]
    fn draw_hline_out_of_bounds_y_noop() {
        let mut fb = vec![0u8; 300]; // 10×10 RGB
        let before = fb.clone();
        draw_hline(&mut fb, 10, 10, 0, 9, -1, 255, 0, 0, 1.0);
        draw_hline(&mut fb, 10, 10, 0, 9, 10, 255, 0, 0, 1.0);
        assert_eq!(fb, before, "out-of-bounds y must not write to framebuffer");
    }

    #[test]
    fn draw_hline_entirely_left_of_viewport_noop() {
        let mut fb = vec![0u8; 300]; // 10×10 RGB
        let before = fb.clone();
        // Both x values negative — span is entirely off-screen to the left.
        // Casting a negative x2 to u32 wraps to a huge index; this must not panic.
        draw_hline(&mut fb, 10, 10, -5, -1, 5, 255, 0, 0, 1.0);
        assert_eq!(
            fb, before,
            "entirely off-screen-left span must not write to framebuffer"
        );
    }

    #[test]
    fn scale_image_noop_when_within_max() {
        let pixels = vec![255u8; 10 * 3];
        let (out, w, h) = scale_image(pixels.clone(), 10, 1, 256);
        assert_eq!((w, h), (10, 1));
        assert_eq!(out, pixels);
    }

    #[test]
    fn scale_image_reduces_max_dimension() {
        let pixels = vec![0u8; 400 * 300 * 3];
        let (_, w, h) = scale_image(pixels, 400, 300, 100);
        assert_eq!(w, 100, "width should be capped at max_dim");
        assert!(h > 0 && h <= 100, "height {h} should be > 0 and <= 100");
    }

    #[test]
    fn draw_polygon_fills_square() {
        // 4-vertex square: x ∈ [1,8], y ∈ [1,7] (upper endpoint excluded by scanline rule).
        let mut fb = vec![0u8; 10 * 10 * 3];
        let vertices: &[(i16, i16)] = &[(1, 1), (8, 1), (8, 8), (1, 8)];
        draw_polygon(&mut fb, 10, 10, vertices, 255, 255, 255, 1.0);
        // Check interior pixels are filled (well inside all four edges).
        for y in 2i32..7 {
            for x in 2i32..7 {
                let base = ((y * 10 + x) * 3) as usize;
                assert_eq!(fb[base], 255, "interior pixel ({x},{y}) should be filled");
            }
        }
        // Corners outside the polygon must be untouched.
        assert_eq!(fb[0], 0, "top-left corner (0,0) should not be filled");
        assert_eq!(fb[(9 * 10 + 9) * 3], 0, "bottom-right corner (9,9) should not be filled");
    }
}
