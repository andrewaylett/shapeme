//! shapeme — approximate images with shapes via simulated annealing.
#![forbid(unsafe_code)]

pub(crate) mod annealing;
pub(crate) mod io;
pub(crate) mod render;
pub(crate) mod shapes;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::render::{Canvas, Texture, TextureCreator};
use sdl2::video::{Window, WindowContext};

use annealing::{AnnealingState, ShapeSet, mutate_shapes};
use io::{build_svg, load_binary, load_png, save_binary, scale_image, svg_to_data_url, write_svg};
use render::{apply_blur, compute_diff, draw_shapes};
use shapes::{Shape, random_shape};

#[derive(Parser, Debug)]
#[command(
    name = "shapeme",
    about = "Approximate images with shapes via simulated annealing"
)]
struct Args {
    /// Input image (PNG, JPEG, or any format supported by the image crate)
    input: PathBuf,
    /// Binary checkpoint file (read on startup, written periodically)
    checkpoint: PathBuf,
    /// Output SVG file (written periodically)
    output_svg: PathBuf,

    #[arg(long, default_value = "1")]
    use_triangles: u8,
    #[arg(long, default_value = "0")]
    use_circles: u8,
    #[arg(long, default_value = "64")]
    max_shapes: usize,
    #[arg(long, default_value = "1")]
    initial_shapes: usize,
    /// Mutation probability per shape per generation (0–1000)
    #[arg(long, default_value = "200")]
    mutation_rate: u32,
    /// Ignore existing checkpoint and start fresh
    #[arg(long)]
    restart: bool,

    /// Skip the SDL window; run until an exit condition is met
    #[arg(long)]
    headless: bool,
    /// Exit after N generations (headless or interactive)
    #[arg(long)]
    max_generations: Option<u64>,
    /// Exit when the best diff falls to or below this percentage
    #[arg(long)]
    target_diff: Option<f32>,
    /// Gaussian blur sigma applied in the SVG output and during diff computation
    #[arg(long)]
    blur_radius: Option<f32>,
    /// Constrain the data URL to at most N bytes (enforced each generation)
    #[arg(long)]
    max_bytes: Option<usize>,
    /// Downsample input so max(width, height) ≤ N; set to 0 to disable
    #[arg(long, default_value = "256")]
    max_dimension: u32,
    /// Print a data URL to stdout on exit
    #[arg(long)]
    data_url: bool,
}

/// SDL2 handles bundled together. Field order determines drop order:
/// texture is destroyed before canvas (renderer), which is correct for SDL2.
struct SdlCtx {
    event_pump: sdl2::EventPump,
    _texture_creator: TextureCreator<WindowContext>,
    texture: Texture,
    canvas: Canvas<Window>,
}

fn init_sdl(width: u32, height: u32) -> Result<SdlCtx> {
    let sdl = sdl2::init().map_err(|e| anyhow::anyhow!("SDL2 init: {e}"))?;
    let video = sdl
        .video()
        .map_err(|e| anyhow::anyhow!("SDL2 video: {e}"))?;
    let window = video
        .window("Shapeme", width, height)
        .opengl()
        .build()
        .context("SDL2 window")?;
    let canvas = window.into_canvas().build().context("SDL2 canvas")?;
    let texture_creator = canvas.texture_creator();
    let texture = texture_creator
        .create_texture_streaming(PixelFormatEnum::RGB24, width, height)
        .context("SDL2 texture")?;
    let event_pump = sdl
        .event_pump()
        .map_err(|e| anyhow::anyhow!("SDL2 events: {e}"))?;
    Ok(SdlCtx {
        event_pump,
        _texture_creator: texture_creator,
        texture,
        canvas,
    })
}

fn finalize(
    args: &Args,
    shapes: &[Shape],
    state: &AnnealingState,
    width: u32,
    height: u32,
) -> Result<()> {
    write_svg(
        &args.output_svg,
        &build_svg(shapes, width, height, args.blur_radius, false),
    )?;
    save_binary(&args.checkpoint, state, shapes)?;
    if args.data_url || args.headless {
        let compact = build_svg(shapes, width, height, args.blur_radius, true);
        println!("{}", svg_to_data_url(&compact));
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let use_triangles = args.use_triangles != 0;
    let use_circles = args.use_circles != 0;
    if !use_triangles && !use_circles {
        anyhow::bail!("at least one of --use-triangles and --use-circles must be non-zero");
    }
    let mutation_rate = args.mutation_rate.min(1000);
    let max_shapes = args.max_shapes.max(args.initial_shapes);

    let (image, width, height) = load_png(&args.input)?;
    let (image, width, height) = if args.max_dimension > 0 {
        scale_image(image, width, height, args.max_dimension)
    } else {
        (image, width, height)
    };
    println!("Image {width}×{height}");

    let mut sdl_ctx: Option<SdlCtx> = if args.headless {
        None
    } else {
        Some(init_sdl(width, height)?)
    };

    let mut rng = SmallRng::from_os_rng();

    let mut fresh = || {
        let s = AnnealingState::new(max_shapes, args.initial_shapes);
        let shapes: Vec<Shape> = (0..args.initial_shapes)
            .map(|_| random_shape(&mut rng, width, height, use_triangles, use_circles))
            .collect();
        (s, shapes)
    };

    let (mut state, mut best_shapes) = if args.restart {
        fresh()
    } else if let Some((s, shapes)) = load_binary(&args.checkpoint)? {
        println!("Loaded {} shapes from checkpoint", shapes.len());
        (s, shapes)
    } else {
        fresh()
    };

    let mut absbest_shapes = best_shapes.clone();

    let mut fb = vec![0u8; (width * height * 3) as usize];
    draw_shapes(&mut fb, width, height, &best_shapes);

    if let Some(ctx) = &mut sdl_ctx {
        let _ = ctx.texture.update(None, &fb, (width * 3) as usize);
        let _ = ctx.canvas.copy(&ctx.texture, None, None);
        ctx.canvas.present();
        std::thread::sleep(std::time::Duration::from_secs(1));
        let _ = ctx.texture.update(None, &image, (width * 3) as usize);
        let _ = ctx.canvas.copy(&ctx.texture, None, None);
        ctx.canvas.present();
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    let mut bestdiff = state.absbestdiff;

    loop {
        state.generation += 1;

        if state.temperature > 0.0 && state.generation % 10 == 0 {
            state.temperature -= 0.00001;
            if state.temperature < 0.0 {
                state.temperature = 0.0;
            }
        }

        if state.generation % 1000 == 0
            && state.max_shapes_incremental < max_shapes
            && best_shapes.len() >= state.max_shapes_incremental
        {
            state.max_shapes_incremental += 1;
        }

        let mut candidate = ShapeSet::new(max_shapes);
        candidate.shapes.clone_from(&best_shapes);
        mutate_shapes(
            &mut rng,
            &mut candidate,
            &state,
            width,
            height,
            mutation_rate,
            use_triangles,
            use_circles,
        );

        let effective = if let Some(max_bytes) = args.max_bytes {
            let mut s = candidate.active().to_vec();
            loop {
                let svg = build_svg(&s, width, height, args.blur_radius, true);
                if svg_to_data_url(&svg).len() <= max_bytes || s.is_empty() {
                    break;
                }
                s.pop();
            }
            s
        } else {
            candidate.active().to_vec()
        };

        fb.fill(0);
        draw_shapes(&mut fb, width, height, &effective);

        let diff = args.blur_radius.map_or_else(
            || compute_diff(&image, &fb),
            |r| {
                let blurred = apply_blur(&fb, width, height, r);
                compute_diff(&image, &blurred)
            },
        );
        let percdiff = diff as f32 / (width * height) as f32 / 442.0 * 100.0;

        let accept = percdiff < bestdiff
            || (state.temperature > 0.0
                && rng.random::<f32>() < state.temperature
                && (percdiff - state.absbestdiff) < 2.0 * state.temperature);

        if accept {
            best_shapes = candidate.shapes;

            if percdiff < bestdiff {
                absbest_shapes.clone_from(&effective);
                state.absbestdiff = percdiff;
            }

            println!(
                "Diff {:.4}% (shapes:{}, max:{}, gen:{}, temp:{:.5})",
                percdiff,
                effective.len(),
                state.max_shapes_incremental,
                state.generation,
                state.temperature
            );

            bestdiff = percdiff;

            if let Some(ctx) = &mut sdl_ctx {
                let _ = ctx.texture.update(None, &fb, (width * 3) as usize);
                let _ = ctx.canvas.copy(&ctx.texture, None, None);
                ctx.canvas.present();
            }
        }

        if let Some(ctx) = &mut sdl_ctx {
            for event in ctx.event_pump.poll_iter() {
                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Q | Keycode::Escape),
                        ..
                    } => {
                        finalize(&args, &absbest_shapes, &state, width, height)?;
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }

        if args.headless {
            let done = args
                .max_generations
                .is_some_and(|n| u64::try_from(state.generation).is_ok_and(|g| g >= n))
                || args.target_diff.is_some_and(|t| state.absbestdiff <= t);
            if done {
                finalize(&args, &absbest_shapes, &state, width, height)?;
                return Ok(());
            }
        }

        if state.generation % 100 == 0 {
            write_svg(
                &args.output_svg,
                &build_svg(&absbest_shapes, width, height, args.blur_radius, false),
            )?;
            save_binary(&args.checkpoint, &state, &absbest_shapes)?;
        }
    }
}
