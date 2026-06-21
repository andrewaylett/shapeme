//! shapeme — approximate images with shapes via simulated annealing.
#![forbid(unsafe_code)]

pub(crate) mod annealing;
pub(crate) mod io;
pub(crate) mod render;
pub(crate) mod shapes;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::render::{Canvas, Texture, TextureCreator};
use sdl2::video::{Window, WindowContext};

use annealing::{AnnealingState, ShapeSet, mutate_shapes};
use io::{
    StoredConfig, build_svg, load_binary, load_png, save_binary, scale_image, svg_to_data_url,
    write_svg,
};
use render::{apply_blur, compute_diff, draw_shapes};
use shapes::{Shape, random_shape};

#[derive(Parser, Debug)]
#[command(
    name = "shapeme",
    about = "Approximate images with shapes via simulated annealing"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Create or overwrite a checkpoint with image and configuration. Does not run the optimiser.
    Setup(SetupArgs),
    /// Load a checkpoint and run the annealing optimiser.
    Process(ProcessArgs),
}

#[derive(ClapArgs, Debug)]
struct SetupArgs {
    /// Binary checkpoint file to create or overwrite
    checkpoint: PathBuf,
    /// Input image (PNG, JPEG, or any format supported by the image crate)
    input: PathBuf,
    /// Output SVG path (stored in checkpoint; used by `process`)
    #[arg(long)]
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
    /// Initial Gaussian blur sigma; will be evolved during processing
    #[arg(long)]
    blur_radius: Option<f32>,
    /// Constrain the data URL to at most N bytes (enforced each generation)
    #[arg(long)]
    max_bytes: Option<usize>,
    /// Downsample input so max(width, height) ≤ N; set to 0 to disable
    #[arg(long, default_value = "256")]
    max_dimension: u32,
}

#[derive(ClapArgs, Debug)]
struct ProcessArgs {
    /// Binary checkpoint file (must exist; created by `setup`)
    checkpoint: PathBuf,
    /// Skip the SDL window; run until an exit condition is met
    #[arg(long)]
    headless: bool,
    /// Exit after N generations
    #[arg(long)]
    max_generations: Option<u64>,
    /// Exit when the best diff falls to or below this percentage
    #[arg(long)]
    target_diff: Option<f32>,
    /// Print a data URL to stdout on exit
    #[arg(long)]
    data_url: bool,
    /// Re-initialise shapes and annealing state, keeping stored config and image
    #[arg(long)]
    restart: bool,
    /// Override the SVG output path stored in the checkpoint
    #[arg(long)]
    output_svg: Option<PathBuf>,
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
    checkpoint: &Path,
    config: &StoredConfig,
    output_svg: &Path,
    shapes: &[Shape],
    blur_radius: Option<f32>,
    state: &AnnealingState,
    print_data_url: bool,
) -> Result<()> {
    write_svg(
        output_svg,
        &build_svg(shapes, config.width, config.height, blur_radius, false),
    )?;
    save_binary(checkpoint, config, state, shapes)?;
    if print_data_url {
        let compact = build_svg(shapes, config.width, config.height, blur_radius, true);
        println!("{}", svg_to_data_url(&compact));
    }
    Ok(())
}

fn setup(args: &SetupArgs) -> Result<()> {
    let use_triangles = args.use_triangles != 0;
    let use_circles = args.use_circles != 0;
    if !use_triangles && !use_circles {
        anyhow::bail!("at least one of --use-triangles and --use-circles must be non-zero");
    }

    let (raw_pixels, raw_width, raw_height) = load_png(&args.input)?;
    let (image, width, height) = if args.max_dimension > 0 {
        scale_image(raw_pixels, raw_width, raw_height, args.max_dimension)
    } else {
        (raw_pixels, raw_width, raw_height)
    };
    println!("Image {width}×{height}");

    let max_shapes = args.max_shapes.max(args.initial_shapes);

    let config = StoredConfig {
        image,
        width,
        height,
        use_triangles,
        use_circles,
        mutation_rate: args.mutation_rate,
        max_bytes: args.max_bytes,
        output_svg: args.output_svg.clone(),
        max_shapes,
        initial_shapes: args.initial_shapes,
        initial_blur_radius: args.blur_radius,
    };

    let mut state = AnnealingState::new(max_shapes, args.initial_shapes);
    state.blur_radius = args.blur_radius;

    save_binary(&args.checkpoint, &config, &state, &[])?;
    println!("Checkpoint written to {}", args.checkpoint.display());
    Ok(())
}

fn process(args: &ProcessArgs) -> Result<()> {
    let checkpoint_path = &args.checkpoint;

    let Some((config, mut state, saved_shapes)) = load_binary(checkpoint_path)? else {
        anyhow::bail!(
            "checkpoint {} not found — run `shapeme setup` to create one",
            checkpoint_path.display()
        );
    };

    let output_svg: &Path = args.output_svg.as_deref().unwrap_or(&config.output_svg);

    let use_triangles = config.use_triangles;
    let use_circles = config.use_circles;
    let mutation_rate = config.mutation_rate.min(1000);
    let max_shapes = config.max_shapes;
    let width = config.width;
    let height = config.height;

    let mut sdl_ctx: Option<SdlCtx> = if args.headless {
        None
    } else {
        Some(init_sdl(width, height)?)
    };

    let mut rng = SmallRng::from_os_rng();

    // Determine starting shapes and blur: fresh start if restarting or no saved shapes yet
    let (init_shapes, init_blur) = if args.restart || saved_shapes.is_empty() {
        let shapes: Vec<Shape> = (0..config.initial_shapes)
            .map(|_| random_shape(&mut rng, width, height, use_triangles, use_circles))
            .collect();
        (shapes, config.initial_blur_radius)
    } else {
        let blur = state.blur_radius;
        (saved_shapes, blur)
    };

    if args.restart {
        state = AnnealingState::new(max_shapes, config.initial_shapes);
    }
    state.blur_radius = init_blur;

    let mut best_shapes = init_shapes;
    let mut best_blur: Option<f32> = init_blur;
    let mut absbest_shapes = best_shapes.clone();
    let mut absbest_blur: Option<f32> = init_blur;

    let mut fb = vec![0u8; (width * height * 3) as usize];
    draw_shapes(&mut fb, width, height, &best_shapes);

    let mut show_original = false;
    let mut last_display_buf: Vec<u8> = best_blur
        .map(|r| apply_blur(&fb, width, height, r))
        .unwrap_or_else(|| fb.clone());

    if let Some(ctx) = &mut sdl_ctx {
        let blurred = best_blur.map(|r| apply_blur(&fb, width, height, r));
        let display = blurred.as_deref().unwrap_or(&fb);
        let _ = ctx.texture.update(None, display, (width * 3) as usize);
        let _ = ctx.canvas.copy(&ctx.texture, None, None);
        ctx.canvas.present();
        std::thread::sleep(std::time::Duration::from_secs(1));
        let _ = ctx
            .texture
            .update(None, &config.image, (width * 3) as usize);
        let _ = ctx.canvas.copy(&ctx.texture, None, None);
        ctx.canvas.present();
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    let mut bestdiff = state.absbestdiff;
    let print_data_url = args.data_url || args.headless;

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
        candidate.blur_radius = best_blur;
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

        let effective = if let Some(max_bytes) = config.max_bytes {
            let mut s = candidate.active().to_vec();
            loop {
                let svg = build_svg(&s, width, height, candidate.blur_radius, true);
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

        // Compute diff against the blurred framebuffer so the fitness function matches SVG output
        let blurred_opt = candidate
            .blur_radius
            .map(|r| apply_blur(&fb, width, height, r));
        let display_buf: &[u8] = blurred_opt.as_deref().unwrap_or(&fb);
        let diff = compute_diff(&config.image, display_buf);
        let percdiff = diff as f32 / (width * height) as f32 / 442.0 * 100.0;

        let accept = percdiff < bestdiff
            || (state.temperature > 0.0
                && rng.random::<f32>() < state.temperature
                && (percdiff - state.absbestdiff) < 2.0 * state.temperature);

        if accept {
            best_shapes = candidate.shapes;
            best_blur = candidate.blur_radius;

            if percdiff < bestdiff {
                absbest_shapes.clone_from(&effective);
                absbest_blur = candidate.blur_radius;
                state.absbestdiff = percdiff;
                state.blur_radius = candidate.blur_radius;
            }

            println!(
                "Diff {:.4}% (shapes:{}, max:{}, gen:{}, temp:{:.5}, blur:{:?})",
                percdiff,
                effective.len(),
                state.max_shapes_incremental,
                state.generation,
                state.temperature,
                candidate.blur_radius,
            );

            bestdiff = percdiff;

            if let Some(ctx) = &mut sdl_ctx {
                last_display_buf.clear();
                last_display_buf.extend_from_slice(display_buf);
                if !show_original {
                    let _ = ctx.texture.update(None, display_buf, (width * 3) as usize);
                    let _ = ctx.canvas.copy(&ctx.texture, None, None);
                    ctx.canvas.present();
                }
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
                        finalize(
                            checkpoint_path,
                            &config,
                            output_svg,
                            &absbest_shapes,
                            absbest_blur,
                            &state,
                            print_data_url,
                        )?;
                        return Ok(());
                    }
                    Event::KeyDown {
                        keycode: Some(Keycode::Space),
                        ..
                    } => {
                        show_original = !show_original;
                        let buf: &[u8] = if show_original {
                            &config.image
                        } else {
                            &last_display_buf
                        };
                        let _ = ctx.texture.update(None, buf, (width * 3) as usize);
                        let _ = ctx.canvas.copy(&ctx.texture, None, None);
                        ctx.canvas.present();
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
                finalize(
                    checkpoint_path,
                    &config,
                    output_svg,
                    &absbest_shapes,
                    absbest_blur,
                    &state,
                    print_data_url,
                )?;
                return Ok(());
            }
        }

        if state.generation % 100 == 0 {
            write_svg(
                output_svg,
                &build_svg(&absbest_shapes, width, height, absbest_blur, false),
            )?;
            save_binary(checkpoint_path, &config, &state, &absbest_shapes)?;
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Setup(setup_args) => setup(&setup_args),
        Command::Process(process_args) => process(&process_args),
    }
}
