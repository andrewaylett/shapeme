use std::fmt::Write as _;

use crate::gene::{CircleGene, PolygonGene, ShapeGene, TriangleGene};
use crate::genome::{Genome, ShapeGenome};
use crate::grid::GridGenome;
use crate::oklab;

/// Build an SVG string representing the given shape genes.
///
/// `blur_radius` (Gaussian sigma) wraps all content in a `<feGaussianBlur>` filter.
/// `compact` omits the XML declaration and DOCTYPE, uses single-quoted attributes,
/// and produces a single line — suitable for embedding in a data URL.
/// Shape colours are converted from `OKlab` to sRGB at output time.
#[must_use]
pub fn build_svg(
    genes: &[ShapeGene],
    width: u32,
    height: u32,
    blur_radius: Option<f32>,
    background: [f32; 3],
    compact: bool,
) -> String {
    let [bg_r, bg_g, bg_b] = oklab::oklab_to_srgb_u8(background);
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

        write!(
            s,
            "<rect width='{width}' height='{height}' fill='#{bg_r:02x}{bg_g:02x}{bg_b:02x}'/>"
        )
        .expect("String write is infallible");
        for gene in genes {
            push_gene(&mut s, gene, true);
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
            "<rect width=\"{width}\" height=\"{height}\" fill=\"#{bg_r:02x}{bg_g:02x}{bg_b:02x}\"/>"
        )
        .expect("String write is infallible");
        for gene in genes {
            push_gene(&mut s, gene, false);
        }

        if blur_radius.is_some() {
            s.push_str("</g>\n");
        }
        s.push_str("</svg>\n");
    }

    s
}

fn push_gene(s: &mut String, gene: &ShapeGene, compact: bool) {
    match gene {
        ShapeGene::Triangle(TriangleGene {
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            oklab,
            alpha,
            ..
        }) => {
            let [r, g, b] = oklab::oklab_to_srgb_u8(*oklab);
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
        ShapeGene::Circle(CircleGene {
            cx,
            cy,
            radius,
            oklab,
            alpha,
            ..
        }) => {
            let [r, g, b] = oklab::oklab_to_srgb_u8(*oklab);
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
        ShapeGene::Polygon(PolygonGene {
            vertices,
            oklab,
            alpha,
            ..
        }) => {
            let [r, g, b] = oklab::oklab_to_srgb_u8(*oklab);
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

/// Build an SVG string from a `ShapeGenome`, respecting z-order.
///
/// Thin wrapper around `build_svg` that extracts genes in z-order and the blur radius.
#[must_use]
pub fn build_svg_from_genome(
    genome: &ShapeGenome,
    width: u32,
    height: u32,
    compact: bool,
) -> String {
    let genes: Vec<&ShapeGene> = genome.sorted_genes();
    let owned: Vec<ShapeGene> = genes.into_iter().cloned().collect();
    build_svg(
        &owned,
        width,
        height,
        genome.blur_radius(),
        genome.background_oklab(),
        compact,
    )
}

/// Build an SVG string from a `GridGenome`.
///
/// Each grid cell is rendered as a `<polygon>` with four vertices (TL, TR, BR, BL).
/// Cells are fully opaque — no `fill-opacity` attribute.
/// If the genome has blur, wraps all content in a `<feGaussianBlur>` filter.
#[must_use]
pub fn build_svg_from_grid(grid: &GridGenome, width: u32, height: u32, compact: bool) -> String {
    let mut s = String::new();

    if compact {
        write!(
            s,
            "<svg xmlns='http://www.w3.org/2000/svg' \
             viewBox='0 0 {width} {height}' \
             preserveAspectRatio='xMidYMid slice'>"
        )
        .expect("String write is infallible");
        if let Some(r) = grid.blur_radius() {
            write!(
                s,
                "<defs><filter id='b'>\
                 <feGaussianBlur stdDeviation='{r}'/>\
                 </filter></defs>\
                 <g filter='url(#b)'>"
            )
            .expect("String write is infallible");
        }
        for row in 0..grid.rows as usize {
            for col in 0..grid.cols as usize {
                push_grid_cell(&mut s, grid, row, col, true);
            }
        }
        if grid.blur_radius().is_some() {
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
        if let Some(r) = grid.blur_radius() {
            writeln!(
                s,
                "<defs><filter id=\"b\">\
                 <feGaussianBlur stdDeviation=\"{r}\"/>\
                 </filter></defs>"
            )
            .expect("String write is infallible");
            writeln!(s, "<g filter=\"url(#b)\">").expect("String write is infallible");
        }
        for row in 0..grid.rows as usize {
            for col in 0..grid.cols as usize {
                push_grid_cell(&mut s, grid, row, col, false);
            }
        }
        if grid.blur_radius().is_some() {
            s.push_str("</g>\n");
        }
        s.push_str("</svg>\n");
    }

    s
}

fn push_grid_cell(s: &mut String, grid: &GridGenome, row: usize, col: usize, compact: bool) {
    let cols_p1 = grid.cols as usize + 1;
    let tl = grid.points[row * cols_p1 + col];
    let tr = grid.points[row * cols_p1 + col + 1];
    let br = grid.points[(row + 1) * cols_p1 + col + 1];
    let bl = grid.points[(row + 1) * cols_p1 + col];
    let color = grid.colors[row * grid.cols as usize + col];
    let [r, g, b] = oklab::oklab_to_srgb_u8(color);
    let pts = format!(
        "{},{} {},{} {},{} {},{}",
        tl.0, tl.1, tr.0, tr.1, br.0, br.1, bl.0, bl.1
    );
    if compact {
        write!(s, "<polygon points='{pts}' fill='#{r:02x}{g:02x}{b:02x}'/>")
            .expect("String write is infallible");
    } else {
        writeln!(
            s,
            "<polygon points=\"{pts}\" fill=\"#{r:02x}{g:02x}{b:02x}\"/>"
        )
        .expect("String write is infallible");
    }
}

/// Build a React JSX-compatible SVG string from shape genes.
///
/// Colours use 8-digit hex RGBA (`fill="#rrggbbaa"`) for semi-transparent shapes; fully
/// opaque shapes (alpha == 100) use 6-digit `fill="#rrggbb"`.
/// The outer `<svg>` uses `width="100%"` `height="100%"` for responsive embedding.
/// `filter_id` is used as the `<filter id>` (typically the first char of the output filename).
#[must_use]
pub fn build_svg_react(
    genes: &[ShapeGene],
    width: u32,
    height: u32,
    blur_radius: Option<f32>,
    background: [f32; 3],
    filter_id: char,
) -> String {
    let [bg_r, bg_g, bg_b] = oklab::oklab_to_srgb_u8(background);
    let mut s = String::new();

    writeln!(s, "<svg").expect("String write is infallible");
    writeln!(s, "  viewBox=\"0 0 {width} {height}\"").expect("String write is infallible");
    writeln!(s, "  preserveAspectRatio=\"xMidYMid slice\"").expect("String write is infallible");
    writeln!(s, "  width=\"100%\"").expect("String write is infallible");
    writeln!(s, "  height=\"100%\"").expect("String write is infallible");
    writeln!(s, ">").expect("String write is infallible");

    if let Some(r) = blur_radius {
        writeln!(s, "  <defs>").expect("String write is infallible");
        writeln!(s, "    <filter id=\"{filter_id}\">").expect("String write is infallible");
        writeln!(s, "      <feGaussianBlur stdDeviation=\"{r}\" />")
            .expect("String write is infallible");
        writeln!(s, "    </filter>").expect("String write is infallible");
        writeln!(s, "  </defs>").expect("String write is infallible");
        writeln!(s, "  <g filter=\"url(#{filter_id})\">").expect("String write is infallible");
    }

    let inner = if blur_radius.is_some() { "    " } else { "  " };
    writeln!(
        s,
        "{inner}<rect width=\"{width}\" height=\"{height}\" fill=\"#{bg_r:02x}{bg_g:02x}{bg_b:02x}\" />"
    )
    .expect("String write is infallible");

    for gene in genes {
        push_gene_react(&mut s, gene, inner);
    }

    if blur_radius.is_some() {
        writeln!(s, "  </g>").expect("String write is infallible");
    }
    s.push_str("</svg>\n");
    s
}

fn alpha_hex(a: u8) -> String {
    if a >= 100 {
        String::new()
    } else {
        format!("{:02x}", (f32::from(a) / 100.0 * 255.0).round() as u8)
    }
}

fn push_gene_react(s: &mut String, gene: &ShapeGene, indent: &str) {
    match gene {
        ShapeGene::Triangle(TriangleGene {
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            oklab,
            alpha,
            ..
        }) => {
            let [r, g, b] = oklab::oklab_to_srgb_u8(*oklab);
            let ah = alpha_hex(*alpha);
            let single = format!(
                "{indent}<polygon points=\"{x1},{y1} {x2},{y2} {x3},{y3}\" fill=\"#{r:02x}{g:02x}{b:02x}{ah}\" />"
            );
            if single.len() <= 80 {
                writeln!(s, "{single}").expect("String write is infallible");
            } else {
                writeln!(s, "{indent}<polygon").expect("String write is infallible");
                writeln!(
                    s,
                    "{indent}  points=\"{x1},{y1} {x2},{y2} {x3},{y3}\""
                )
                .expect("String write is infallible");
                writeln!(s, "{indent}  fill=\"#{r:02x}{g:02x}{b:02x}{ah}\"")
                    .expect("String write is infallible");
                writeln!(s, "{indent}/>").expect("String write is infallible");
            }
        }
        ShapeGene::Circle(CircleGene {
            cx,
            cy,
            radius,
            oklab,
            alpha,
            ..
        }) => {
            let [r, g, b] = oklab::oklab_to_srgb_u8(*oklab);
            let ah = alpha_hex(*alpha);
            let single = format!(
                "{indent}<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{radius}\" fill=\"#{r:02x}{g:02x}{b:02x}{ah}\" />"
            );
            if single.len() <= 80 {
                writeln!(s, "{single}").expect("String write is infallible");
            } else {
                writeln!(s, "{indent}<circle").expect("String write is infallible");
                writeln!(s, "{indent}  cx=\"{cx}\"").expect("String write is infallible");
                writeln!(s, "{indent}  cy=\"{cy}\"").expect("String write is infallible");
                writeln!(s, "{indent}  r=\"{radius}\"").expect("String write is infallible");
                writeln!(s, "{indent}  fill=\"#{r:02x}{g:02x}{b:02x}{ah}\"")
                    .expect("String write is infallible");
                writeln!(s, "{indent}/>").expect("String write is infallible");
            }
        }
        ShapeGene::Polygon(PolygonGene {
            vertices,
            oklab,
            alpha,
            ..
        }) => {
            let [r, g, b] = oklab::oklab_to_srgb_u8(*oklab);
            let ah = alpha_hex(*alpha);
            let pts: String = vertices
                .iter()
                .map(|(x, y)| format!("{x},{y}"))
                .collect::<Vec<_>>()
                .join(" ");
            let single = format!(
                "{indent}<polygon points=\"{pts}\" fill=\"#{r:02x}{g:02x}{b:02x}{ah}\" />"
            );
            if single.len() <= 80 {
                writeln!(s, "{single}").expect("String write is infallible");
            } else {
                writeln!(s, "{indent}<polygon").expect("String write is infallible");
                writeln!(s, "{indent}  points=\"{pts}\"").expect("String write is infallible");
                writeln!(s, "{indent}  fill=\"#{r:02x}{g:02x}{b:02x}{ah}\"")
                    .expect("String write is infallible");
                writeln!(s, "{indent}/>").expect("String write is infallible");
            }
        }
    }
}

/// Build a React JSX SVG string from a `ShapeGenome`, respecting z-order.
#[must_use]
pub fn build_svg_react_from_genome(
    genome: &ShapeGenome,
    width: u32,
    height: u32,
    filter_id: char,
) -> String {
    let genes: Vec<&ShapeGene> = genome.sorted_genes();
    let owned: Vec<ShapeGene> = genes.into_iter().cloned().collect();
    build_svg_react(
        &owned,
        width,
        height,
        genome.blur_radius(),
        genome.background_oklab(),
        filter_id,
    )
}

/// Build a React JSX SVG string from a `GridGenome`.
///
/// Grid cells are fully opaque, so no alpha suffix is emitted.
#[must_use]
pub fn build_svg_react_from_grid(
    grid: &GridGenome,
    width: u32,
    height: u32,
    filter_id: char,
) -> String {
    let mut s = String::new();

    writeln!(s, "<svg").expect("String write is infallible");
    writeln!(s, "  viewBox=\"0 0 {width} {height}\"").expect("String write is infallible");
    writeln!(s, "  preserveAspectRatio=\"xMidYMid slice\"").expect("String write is infallible");
    writeln!(s, "  width=\"100%\"").expect("String write is infallible");
    writeln!(s, "  height=\"100%\"").expect("String write is infallible");
    writeln!(s, ">").expect("String write is infallible");

    if let Some(r) = grid.blur_radius() {
        writeln!(s, "  <defs>").expect("String write is infallible");
        writeln!(s, "    <filter id=\"{filter_id}\">").expect("String write is infallible");
        writeln!(s, "      <feGaussianBlur stdDeviation=\"{r}\" />")
            .expect("String write is infallible");
        writeln!(s, "    </filter>").expect("String write is infallible");
        writeln!(s, "  </defs>").expect("String write is infallible");
        writeln!(s, "  <g filter=\"url(#{filter_id})\">").expect("String write is infallible");
    }

    let inner = if grid.blur_radius().is_some() { "    " } else { "  " };
    for row in 0..grid.rows as usize {
        for col in 0..grid.cols as usize {
            push_grid_cell_react(&mut s, grid, row, col, inner);
        }
    }

    if grid.blur_radius().is_some() {
        writeln!(s, "  </g>").expect("String write is infallible");
    }
    s.push_str("</svg>\n");
    s
}

fn push_grid_cell_react(
    s: &mut String,
    grid: &GridGenome,
    row: usize,
    col: usize,
    indent: &str,
) {
    let cols_p1 = grid.cols as usize + 1;
    let tl = grid.points[row * cols_p1 + col];
    let tr = grid.points[row * cols_p1 + col + 1];
    let br = grid.points[(row + 1) * cols_p1 + col + 1];
    let bl = grid.points[(row + 1) * cols_p1 + col];
    let color = grid.colors[row * grid.cols as usize + col];
    let [r, g, b] = oklab::oklab_to_srgb_u8(color);
    let pts = format!(
        "{},{} {},{} {},{} {},{}",
        tl.0, tl.1, tr.0, tr.1, br.0, br.1, bl.0, bl.1
    );
    let single = format!("{indent}<polygon points=\"{pts}\" fill=\"#{r:02x}{g:02x}{b:02x}\" />");
    if single.len() <= 80 {
        writeln!(s, "{single}").expect("String write is infallible");
    } else {
        writeln!(s, "{indent}<polygon").expect("String write is infallible");
        writeln!(s, "{indent}  points=\"{pts}\"").expect("String write is infallible");
        writeln!(s, "{indent}  fill=\"#{r:02x}{g:02x}{b:02x}\"")
            .expect("String write is infallible");
        writeln!(s, "{indent}/>").expect("String write is infallible");
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
    use crate::gene::TriangleGene;

    fn sample_triangle() -> Vec<ShapeGene> {
        vec![ShapeGene::Triangle(TriangleGene {
            x1: 0,
            y1: 0,
            x2: 10,
            y2: 0,
            x3: 5,
            y3: 10,
            oklab: [0.6279, -0.2516, 0.0000],
            alpha: 50,
            z_order: 0,
        })]
    }

    #[test]
    fn build_svg_compact_has_viewbox_and_preserveaspectratio() {
        let svg = build_svg(&sample_triangle(), 100, 200, None, [0.0, 0.0, 0.0], true);
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
        let svg = build_svg(
            &sample_triangle(),
            100,
            100,
            Some(4.0),
            [0.0, 0.0, 0.0],
            true,
        );
        assert!(!svg.contains('\n'), "compact SVG must not contain newlines");
    }

    #[test]
    fn build_svg_compact_includes_blur_filter() {
        let svg = build_svg(
            &sample_triangle(),
            100,
            100,
            Some(8.0),
            [0.0, 0.0, 0.0],
            true,
        );
        assert!(svg.contains("feGaussianBlur"), "missing blur filter: {svg}");
        assert!(
            svg.contains("stdDeviation='8'"),
            "wrong stdDeviation: {svg}"
        );
    }

    #[test]
    fn build_svg_verbose_has_viewbox() {
        let svg = build_svg(&sample_triangle(), 100, 200, None, [0.0, 0.0, 0.0], false);
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
