use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use image::imageops::FilterType;

use crate::annealing::AnnealingState;
use crate::shapes::Shape;

/// Load an image (any format supported by the image crate) and return it as an
/// RGB24 byte buffer plus dimensions.
pub(crate) fn load_png(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .into_rgb8();
    let width = img.width();
    let height = img.height();
    Ok((img.into_raw(), width, height))
}

/// Downsample `pixels` so that `max(width, height)` ≤ `max_dim` using Lanczos3 resampling.
/// Returns unchanged if already within the limit.
pub(crate) fn scale_image(
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    max_dim: u32,
) -> (Vec<u8>, u32, u32) {
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
    let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width, height, pixels)
        .expect("pixel buffer must match declared dimensions");
    let resized = image::imageops::resize(&img, new_w, new_h, FilterType::Lanczos3);
    (resized.into_raw(), new_w, new_h)
}

/// Build an SVG string representing the given shapes.
///
/// `blur_radius` (Gaussian sigma) wraps all content in a `<feGaussianBlur>` filter.
/// `compact` omits the XML declaration and DOCTYPE, uses single-quoted attributes,
/// and produces a single line — suitable for embedding in a data URL.
pub(crate) fn build_svg(
    shapes: &[Shape],
    width: u32,
    height: u32,
    blur_radius: Option<f32>,
    compact: bool,
) -> String {
    let mut s = String::new();

    if compact {
        write!(
            s,
            "<svg xmlns='http://www.w3.org/2000/svg' \
             viewBox='0 0 {width} {height}' \
             preserveAspectRatio='xMidYMid slice'>"
        )
        .expect("String write is infallible");

        if let Some(r) = blur_radius {
            write!(
                s,
                "<defs><filter id='b'>\
                 <feGaussianBlur stdDeviation='{r}'/>\
                 </filter></defs>\
                 <g filter='url(#b)'>"
            )
            .expect("String write is infallible");
        }

        write!(s, "<rect width='{width}' height='{height}' fill='#000'/>")
            .expect("String write is infallible");
        push_shapes_compact(&mut s, shapes);

        if blur_radius.is_some() {
            s.push_str("</g>");
        }
        s.push_str("</svg>");
    } else {
        s.push_str("<?xml version=\"1.0\" standalone=\"no\"?>\n");
        s.push_str("<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" ");
        s.push_str("\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n");
        writeln!(
            s,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" \
             viewBox=\"0 0 {width} {height}\" \
             preserveAspectRatio=\"xMidYMid slice\" \
             version=\"1.1\">"
        )
        .expect("String write is infallible");

        if let Some(r) = blur_radius {
            writeln!(
                s,
                "<defs><filter id=\"b\">\
                 <feGaussianBlur stdDeviation=\"{r}\"/>\
                 </filter></defs>"
            )
            .expect("String write is infallible");
            writeln!(s, "<g filter=\"url(#b)\">").expect("String write is infallible");
        }

        writeln!(
            s,
            "<rect width=\"{width}\" height=\"{height}\" fill=\"#000000\"/>"
        )
        .expect("String write is infallible");
        push_shapes_verbose(&mut s, shapes);

        if blur_radius.is_some() {
            s.push_str("</g>\n");
        }
        s.push_str("</svg>\n");
    }

    s
}

fn push_shapes_compact(s: &mut String, shapes: &[Shape]) {
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
                write!(
                    s,
                    "<polygon points='{x1},{y1} {x2},{y2} {x3},{y3}' \
                     style='fill:#{r:02x}{g:02x}{b:02x};\
                     fill-opacity:{:.2}'/>",
                    f32::from(*alpha) / 100.0
                )
                .expect("String write is infallible");
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
                write!(
                    s,
                    "<circle cx='{cx}' cy='{cy}' r='{radius}' \
                     style='fill:#{r:02x}{g:02x}{b:02x};\
                     fill-opacity:{:.2}'/>",
                    f32::from(*alpha) / 100.0
                )
                .expect("String write is infallible");
            }
        }
    }
}

fn push_shapes_verbose(s: &mut String, shapes: &[Shape]) {
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
                writeln!(
                    s,
                    "<polygon points=\"{x1},{y1} {x2},{y2} {x3},{y3}\" \
                     style=\"fill:#{r:02x}{g:02x}{b:02x};\
                     fill-opacity:{:.2}\"/>",
                    f32::from(*alpha) / 100.0
                )
                .expect("String write is infallible");
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
                writeln!(
                    s,
                    "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{radius}\" \
                     style=\"fill:#{r:02x}{g:02x}{b:02x};\
                     fill-opacity:{:.2}\"/>",
                    f32::from(*alpha) / 100.0
                )
                .expect("String write is infallible");
            }
        }
    }
}

/// Write an SVG string to a file.
pub(crate) fn write_svg(path: &Path, svg: &str) -> Result<()> {
    fs::write(path, svg).with_context(|| format!("failed to write {}", path.display()))
}

/// Percent-encode a compact SVG string for use as a `data:image/svg+xml;charset=utf-8,` URL.
///
/// `%` is encoded first to prevent double-encoding subsequent replacements.
pub(crate) fn svg_to_data_url(svg: &str) -> String {
    let encoded = svg
        .replace('%', "%25")
        .replace('<', "%3C")
        .replace('>', "%3E")
        .replace('#', "%23")
        .replace('&', "%26")
        .replace('\n', "");
    format!("data:image/svg+xml;charset=utf-8,{encoded}")
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Checkpoint {
    state: AnnealingState,
    shapes: Vec<Shape>,
}

/// Save annealing state and shapes to a binary checkpoint file.
pub(crate) fn save_binary(path: &Path, state: &AnnealingState, shapes: &[Shape]) -> Result<()> {
    let checkpoint = Checkpoint {
        state: state.clone(),
        shapes: shapes.to_vec(),
    };
    let encoded = bincode::serde::encode_to_vec(&checkpoint, bincode::config::standard())
        .context("failed to encode checkpoint")?;
    fs::write(path, encoded).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Load a binary checkpoint. Returns `None` if the file does not exist
/// (fresh start). Fails on corrupt data.
pub(crate) fn load_binary(path: &Path) -> Result<Option<(AnnealingState, Vec<Shape>)>> {
    if !path.exists() {
        return Ok(None);
    }
    let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let (checkpoint, _): (Checkpoint, _) =
        bincode::serde::decode_from_slice(&data, bincode::config::standard())
            .context("failed to decode checkpoint")?;
    Ok(Some((checkpoint.state, checkpoint.shapes)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::Shape;

    fn sample_triangle() -> Vec<Shape> {
        vec![Shape::Triangle {
            x1: 0,
            y1: 0,
            x2: 10,
            y2: 0,
            x3: 5,
            y3: 10,
            r: 255,
            g: 0,
            b: 0,
            alpha: 50,
        }]
    }

    #[test]
    fn build_svg_compact_has_viewbox_and_preserveaspectratio() {
        let svg = build_svg(&sample_triangle(), 100, 200, None, true);
        assert!(
            svg.contains("viewBox='0 0 100 200'"),
            "missing viewBox: {svg}"
        );
        assert!(
            svg.contains("preserveAspectRatio='xMidYMid slice'"),
            "missing preserveAspectRatio: {svg}"
        );
    }

    #[test]
    fn build_svg_compact_is_single_line() {
        let svg = build_svg(&sample_triangle(), 100, 100, Some(4.0), true);
        assert!(!svg.contains('\n'), "compact SVG must not contain newlines");
    }

    #[test]
    fn build_svg_compact_includes_blur_filter() {
        let svg = build_svg(&sample_triangle(), 100, 100, Some(8.0), true);
        assert!(svg.contains("feGaussianBlur"), "missing blur filter: {svg}");
        assert!(
            svg.contains("stdDeviation='8'"),
            "wrong stdDeviation: {svg}"
        );
    }

    #[test]
    fn build_svg_verbose_has_viewbox() {
        let svg = build_svg(&sample_triangle(), 100, 200, None, false);
        assert!(
            svg.contains("viewBox=\"0 0 100 200\""),
            "missing viewBox: {svg}"
        );
        assert!(
            svg.contains("preserveAspectRatio=\"xMidYMid slice\""),
            "missing preserveAspectRatio: {svg}"
        );
    }

    #[test]
    fn svg_to_data_url_encodes_special_chars() {
        let url = svg_to_data_url("<svg><rect fill='#fff'/></svg>");
        assert!(
            url.starts_with("data:image/svg+xml;charset=utf-8,"),
            "wrong prefix: {url}"
        );
        assert!(url.contains("%3C"), "< not encoded: {url}");
        assert!(url.contains("%3E"), "> not encoded: {url}");
        assert!(url.contains("%23"), "# not encoded: {url}");
        assert!(!url.contains('#'), "raw # in url: {url}");
        assert!(!url.contains('<'), "raw < in url: {url}");
        assert!(!url.contains('>'), "raw > in url: {url}");
    }

    #[test]
    fn svg_to_data_url_no_double_encoding() {
        // < must become %3C, not %253C (which happens if % is encoded after <)
        let url = svg_to_data_url("<tag/>");
        assert!(url.contains("%3C"), "< not encoded: {url}");
        assert!(!url.contains("%253C"), "double-encoded <: {url}");
    }

    #[test]
    fn svg_to_data_url_encodes_percent_sign() {
        let url = svg_to_data_url("50%");
        assert!(url.contains("%25"), "% not encoded as %25: {url}");
    }

    #[test]
    fn scale_image_noop_when_within_max() {
        let pixels = vec![255u8; 10 * 1 * 3];
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
}
