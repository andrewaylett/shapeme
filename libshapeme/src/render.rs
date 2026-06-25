use image::{ImageBuffer, Rgb, imageops};

use crate::gene::{CircleGene, PolygonGene, ShapeGene, TriangleGene};

/// Draw a horizontal span into an `OKlab` f32 framebuffer with alpha blending.
/// Out-of-bounds y is silently ignored; x coordinates are clamped to the row.
#[allow(
    clippy::too_many_arguments,
    reason = "pixel coordinate + colour + alpha parameters are unavoidable in a rasterizer"
)]
fn draw_hline(
    fb: &mut [f32],
    width: u32,
    height: u32,
    mut x1: i32,
    mut x2: i32,
    y: i32,
    color: [f32; 3],
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
    let inv = 1.0 - alpha;
    let row = y as u32 * width;
    for x in x1..=x2 {
        let base = ((row + x) * 3) as usize;
        fb[base]     = alpha * color[0] + inv * fb[base];
        fb[base + 1] = alpha * color[1] + inv * fb[base + 1];
        fb[base + 2] = alpha * color[2] + inv * fb[base + 2];
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "pixel coordinate + colour + alpha parameters are unavoidable in a rasterizer"
)]
fn draw_circle(
    fb: &mut [f32],
    width: u32,
    height: u32,
    cx: i16,
    cy: i16,
    radius: i16,
    color: [f32; 3],
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
        draw_hline(fb, width, height, x1, x2, y, color, alpha);
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "pixel coordinate + colour + alpha parameters are unavoidable in a rasterizer"
)]
fn draw_triangle(
    fb: &mut [f32],
    width: u32,
    height: u32,
    x1: i16,
    y1: i16,
    x2: i16,
    y2: i16,
    x3: i16,
    y3: i16,
    color: [f32; 3],
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
            draw_hline(fb, width, height, sx as i32, ex as i32, sy as i32, color, alpha);
            sy += 1.0;
            sx += dx2;
            ex += dx1;
        }
        ex = bx;
        while sy <= cy {
            draw_hline(fb, width, height, sx as i32, ex as i32, sy as i32, color, alpha);
            sy += 1.0;
            sx += dx2;
            ex += dx3;
        }
    } else {
        while sy <= by {
            draw_hline(fb, width, height, sx as i32, ex as i32, sy as i32, color, alpha);
            sy += 1.0;
            sx += dx1;
            ex += dx2;
        }
        sx = bx;
        sy = by + 1.0;
        while sy <= cy {
            draw_hline(fb, width, height, sx as i32, ex as i32, sy as i32, color, alpha);
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
    fb: &mut [f32],
    width: u32,
    height: u32,
    vertices: &[(i16, i16)],
    color: [f32; 3],
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
                draw_hline(fb, width, height, pair[0], pair[1], y, color, alpha);
            }
        }
    }
}

/// Rasterise a slice of `ShapeGene`s into an `OKlab` f32 framebuffer, sorted by `z_order`.
///
/// Sorting n ≤ 64 genes per call is negligible; correctness requires stable z-ordering
/// so recombined genomes render identically regardless of `Vec` insertion order.
pub fn draw_genes(fb: &mut [f32], width: u32, height: u32, genes: &[ShapeGene]) {
    let mut sorted: Vec<&ShapeGene> = genes.iter().collect();
    sorted.sort_unstable_by_key(|g| g.z_order());
    for g in sorted {
        draw_gene_inner(fb, width, height, g);
    }
}

fn draw_gene_inner(fb: &mut [f32], width: u32, height: u32, gene: &ShapeGene) {
    match gene {
        ShapeGene::Triangle(TriangleGene { x1, y1, x2, y2, x3, y3, oklab, alpha, .. }) => {
            draw_triangle(fb, width, height, *x1, *y1, *x2, *y2, *x3, *y3, *oklab, *alpha as f32 / 100.0);
        }
        ShapeGene::Circle(CircleGene { cx, cy, radius, oklab, alpha, .. }) => {
            draw_circle(fb, width, height, *cx, *cy, *radius, *oklab, *alpha as f32 / 100.0);
        }
        ShapeGene::Polygon(PolygonGene { vertices, oklab, alpha, .. }) => {
            draw_polygon(fb, width, height, vertices, *oklab, *alpha as f32 / 100.0);
        }
    }
}

/// Apply a Gaussian blur (sigma = `radius`) to an `OKlab` f32 framebuffer.
///
/// Gaussian blur is a linear operation; applying it in `OKlab` (a perceptually uniform space)
/// gives perceptually correct results.
#[must_use]
pub fn apply_blur(fb: &[f32], width: u32, height: u32, radius: f32) -> Vec<f32> {
    let img: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::from_raw(width, height, fb.to_vec())
        .expect("framebuffer dimensions must match pixel data");
    imageops::blur(&img, radius).into_raw()
}

/// RMSE of per-pixel `OKlab` distances between two f32 framebuffers.
///
/// Returns the root-mean-square error per pixel, in `OKlab` units ([0, ~1]).
/// RMSE penalises large outlier errors more than MAE, which matches the goal of
/// minimising visually salient differences.
#[must_use]
pub fn compute_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let chunks = a.len() / 3;
    let sum_sq: f64 = (0..chunks)
        .map(|i| {
            let base = i * 3;
            let dl = f64::from(a[base] - b[base]);
            let da = f64::from(a[base + 1] - b[base + 1]);
            let db = f64::from(a[base + 2] - b[base + 2]);
            dl * dl + da * da + db * db
        })
        .sum();
    (sum_sq / chunks as f64).sqrt()
}

/// Downsample an sRGB u8 pixel buffer so that `max(width, height)` ≤ `max_dim`
/// using Lanczos3 resampling.  Returns unchanged if already within the limit.
///
/// Scaling is done in sRGB u8 space rather than `OKlab` because `imageops::resize`
/// clips negative f32 values to 0, which would destroy the negative `a`/`b` channels
/// that represent cool/blue `OKlab` colours and introduce a warm (sepia) bias.
#[must_use]
pub fn scale_image(
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    max_dim: u32,
) -> (Vec<u8>, u32, u32) {
    if width.max(height) <= max_dim {
        return (pixels, width, height);
    }
    let (new_w, new_h) = if width >= height {
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
    let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, pixels)
        .expect("pixel buffer must match declared dimensions");
    let resized = imageops::resize(&img, new_w, new_h, imageops::FilterType::Lanczos3);
    (resized.into_raw(), new_w, new_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_diff_identical() {
        let buf = vec![0.5f32; 300];
        assert_eq!(compute_diff(&buf, &buf), 0.0);
    }

    #[test]
    fn compute_diff_white_vs_black() {
        let black = vec![0.0f32; 300];
        let white = vec![1.0f32; 300];
        // Per-pixel squared distance: 1^2 + 1^2 + 1^2 = 3. RMSE = sqrt(3) ≈ 1.732.
        let diff = compute_diff(&black, &white);
        let expected = 3.0f64.sqrt();
        assert!(
            (diff - expected).abs() < 1e-6,
            "expected {expected}, got {diff}"
        );
    }

    #[test]
    fn apply_blur_preserves_buffer_length() {
        let pixels = vec![0.5f32; 30 * 30 * 3];
        let blurred = apply_blur(&pixels, 30, 30, 2.0);
        assert_eq!(blurred.len(), pixels.len());
    }

    #[test]
    fn apply_blur_preserves_negative_oklab_channels() {
        use crate::oklab;
        // Verify Gaussian blur does not clip negative OKlab a/b channels.
        let blue = oklab::srgb_u8_to_oklab(0, 0, 255);
        let pixels: Vec<f32> = (0..30 * 30).flat_map(|_| blue).collect();
        let blurred = apply_blur(&pixels, 30, 30, 2.0);
        // Interior pixels (away from edges) should still have negative b
        let centre = (15 * 30 + 15) * 3;
        let b_channel = blurred[centre + 2];
        assert!(
            b_channel < -0.1,
            "blur clipped negative b channel: {b_channel:.4}, expected ≈ {:.4}",
            blue[2]
        );
    }

    #[test]
    fn draw_hline_out_of_bounds_y_noop() {
        let mut fb = vec![0.0f32; 300]; // 10×10 OKlab
        let before = fb.clone();
        draw_hline(&mut fb, 10, 10, 0, 9, -1, [1.0, 0.0, 0.0], 1.0);
        draw_hline(&mut fb, 10, 10, 0, 9, 10, [1.0, 0.0, 0.0], 1.0);
        assert_eq!(fb, before, "out-of-bounds y must not write to framebuffer");
    }

    #[test]
    fn draw_hline_entirely_left_of_viewport_noop() {
        let mut fb = vec![0.0f32; 300]; // 10×10 OKlab
        let before = fb.clone();
        draw_hline(&mut fb, 10, 10, -5, -1, 5, [1.0, 0.0, 0.0], 1.0);
        assert_eq!(
            fb, before,
            "entirely off-screen-left span must not write to framebuffer"
        );
    }

    #[test]
    fn scale_image_noop_when_within_max() {
        let pixels = vec![128u8; 10 * 3];
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
        let mut fb = vec![0.0f32; 10 * 10 * 3];
        let vertices: &[(i16, i16)] = &[(1, 1), (8, 1), (8, 8), (1, 8)];
        draw_polygon(&mut fb, 10, 10, vertices, [1.0, 0.0, 0.0], 1.0);
        // Check interior pixels are filled (well inside all four edges).
        for y in 2i32..7 {
            for x in 2i32..7 {
                let base = ((y * 10 + x) * 3) as usize;
                assert!(
                    (fb[base] - 1.0).abs() < 1e-6,
                    "interior pixel ({x},{y}) L channel should be 1.0"
                );
            }
        }
        // Corners outside the polygon must be untouched.
        assert_eq!(fb[0], 0.0, "top-left corner (0,0) should not be filled");
        assert_eq!(
            fb[(9 * 10 + 9) * 3],
            0.0,
            "bottom-right corner (9,9) should not be filled"
        );
    }

    #[test]
    fn scale_image_preserves_blue_colour() {
        // Scaling in sRGB u8 avoids the negative-f32-clipping issue that caused sepia.
        // Verify a blue image stays blue after scaling.
        let pixels: Vec<u8> = (0..8 * 8).flat_map(|_| [0u8, 0u8, 255u8]).collect();
        let (scaled, w, h) = scale_image(pixels, 8, 8, 4);
        assert_eq!((w, h), (4, 4));
        // All pixels should still be approximately blue
        for i in 0..(w * h) as usize {
            let r = scaled[i * 3];
            let b = scaled[i * 3 + 2];
            assert!(b > 200, "blue channel should survive scaling: R={r} B={b}");
            assert!(r < 20, "red should not dominate: R={r} B={b}");
        }
    }
}
