use std::fmt::Write as _;

use crate::shapes::Shape;

/// Build an SVG string representing the given shapes.
///
/// `blur_radius` (Gaussian sigma) wraps all content in a `<feGaussianBlur>` filter.
/// `compact` omits the XML declaration and DOCTYPE, uses single-quoted attributes,
/// and produces a single line — suitable for embedding in a data URL.
#[must_use]
pub fn build_svg(
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
        for shape in shapes {
            push_shape(&mut s, shape, true);
        }

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
        for shape in shapes {
            push_shape(&mut s, shape, false);
        }

        if blur_radius.is_some() {
            s.push_str("</g>\n");
        }
        s.push_str("</svg>\n");
    }

    s
}

fn push_shape(s: &mut String, shape: &Shape, compact: bool) {
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
            let a = f32::from(*alpha) / 100.0;
            if compact {
                write!(
                    s,
                    "<polygon points='{x1},{y1} {x2},{y2} {x3},{y3}' \
                     style='fill:#{r:02x}{g:02x}{b:02x};\
                     fill-opacity:{a:.2}'/>",
                )
                .expect("String write is infallible");
            } else {
                writeln!(
                    s,
                    "<polygon points=\"{x1},{y1} {x2},{y2} {x3},{y3}\" \
                     style=\"fill:#{r:02x}{g:02x}{b:02x};\
                     fill-opacity:{a:.2}\"/>",
                )
                .expect("String write is infallible");
            }
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
            let a = f32::from(*alpha) / 100.0;
            if compact {
                write!(
                    s,
                    "<circle cx='{cx}' cy='{cy}' r='{radius}' \
                     style='fill:#{r:02x}{g:02x}{b:02x};\
                     fill-opacity:{a:.2}'/>",
                )
                .expect("String write is infallible");
            } else {
                writeln!(
                    s,
                    "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{radius}\" \
                     style=\"fill:#{r:02x}{g:02x}{b:02x};\
                     fill-opacity:{a:.2}\"/>",
                )
                .expect("String write is infallible");
            }
        }
        Shape::Polygon {
            vertices,
            r,
            g,
            b,
            alpha,
        } => {
            let a = f32::from(*alpha) / 100.0;
            let pts: String = vertices
                .iter()
                .map(|(x, y)| format!("{x},{y}"))
                .collect::<Vec<_>>()
                .join(" ");
            if compact {
                write!(
                    s,
                    "<polygon points='{pts}' \
                     style='fill:#{r:02x}{g:02x}{b:02x};\
                     fill-opacity:{a:.2}'/>",
                )
                .expect("String write is infallible");
            } else {
                writeln!(
                    s,
                    "<polygon points=\"{pts}\" \
                     style=\"fill:#{r:02x}{g:02x}{b:02x};\
                     fill-opacity:{a:.2}\"/>",
                )
                .expect("String write is infallible");
            }
        }
    }
}

/// Percent-encode a compact SVG string for use as a `data:image/svg+xml,` URL.
///
/// `%` is encoded first to prevent double-encoding subsequent replacements.
#[must_use]
pub fn svg_to_data_url(svg: &str) -> String {
    let encoded = svg
        .replace('%', "%25")
        .replace('<', "%3C")
        .replace('>', "%3E")
        .replace('#', "%23")
        .replace('&', "%26")
        .replace('\n', "");
    format!("data:image/svg+xml,{encoded}")
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
            url.starts_with("data:image/svg+xml,"),
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
}
