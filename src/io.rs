use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::annealing::AnnealingState;
use crate::shapes::Shape;

/// Load a PNG and return it as an RGB24 byte buffer plus dimensions.
pub fn load_png(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .into_rgb8();
    let width = img.width();
    let height = img.height();
    Ok((img.into_raw(), width, height))
}

/// Write the current shape set as an SVG file, matching the original format.
pub fn save_svg(path: &Path, shapes: &[Shape], width: u32, height: u32) -> Result<()> {
    let file =
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);

    write!(w, "<?xml version=\"1.0\" standalone=\"no\"?>")?;
    write!(w, "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" ")?;
    write!(w, "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">")?;
    write!(
        w,
        "<svg width=\"100%\" height=\"100%\" style=\"background-color:#000000;\" "
    )?;
    writeln!(w, "version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\">")?;

    writeln!(
        w,
        "<polygon points=\"0,0 {},{} {},{} 0,{}\" \
         style=\"fill:#000000;stroke:#000000;stroke-width:0;fill-opacity:1;\"/>",
        width - 1,
        0,
        width - 1,
        height - 1,
        height - 1
    )?;

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
                writeln!(
                    w,
                    "<polygon points=\"{},{} {},{} {},{}\" \
                     style=\"fill:#{:02x}{:02x}{:02x};stroke:#000000;\
                     stroke-width:0;fill-opacity:{:.2};\"/>",
                    x1,
                    y1,
                    x2,
                    y2,
                    x3,
                    y3,
                    r,
                    g,
                    b,
                    alpha as f32 / 100.0
                )?;
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
                    w,
                    "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" \
                     style=\"fill:#{:02x}{:02x}{:02x};stroke:#000000;\
                     stroke-width:0;fill-opacity:{:.2};\"/>",
                    cx,
                    cy,
                    radius,
                    r,
                    g,
                    b,
                    alpha as f32 / 100.0
                )?;
            }
        }
    }

    writeln!(w, "</svg>")?;
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Checkpoint {
    state: AnnealingState,
    shapes: Vec<Shape>,
}

/// Save annealing state and shapes to a binary checkpoint file.
pub fn save_binary(path: &Path, state: &AnnealingState, shapes: &[Shape]) -> Result<()> {
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
pub fn load_binary(path: &Path) -> Result<Option<(AnnealingState, Vec<Shape>)>> {
    if !path.exists() {
        return Ok(None);
    }
    let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let (checkpoint, _): (Checkpoint, _) =
        bincode::serde::decode_from_slice(&data, bincode::config::standard())
            .context("failed to decode checkpoint")?;
    Ok(Some((checkpoint.state, checkpoint.shapes)))
}
