//! shapeme — approximate images with shapes via simulated annealing.
#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};

use anyhow::{Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::render::{Canvas, Texture, TextureCreator};
use sdl2::video::{Window, WindowContext};
use tracing::Level;

use libshapeme::annealing::{AnnealingState, ShapeSet, mutate_shapes};
use libshapeme::render::{apply_blur, compute_diff, draw_shapes, scale_image};
use libshapeme::shapes::{Shape, random_shape};
use libshapeme::svg::{build_svg, svg_to_data_url};

#[derive(Parser, Debug)]
#[command(
    name = "shapeme",
    about = "Approximate images with shapes via simulated annealing"
)]
struct Args {
    /// Suppress all output except warnings and errors
    #[arg(long, global = true)]
    quiet: bool,
    /// Enable verbose debug output
    #[arg(long, global = true)]
    verbose: bool,
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
    #[arg(long, default_value = "0")]
    use_polygons: u8,
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
    /// Generations to run per parallel batch per round
    #[arg(long, default_value = "200")]
    batch_size: u32,
    /// Number of parallel batches per round
    #[arg(long, default_value = "10")]
    parallel_batches: usize,
}

/// Configuration embedded in the checkpoint by `setup`. Carries everything `process` needs
/// to reconstruct the run without any additional CLI flags.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct StoredConfig {
    /// Scaled RGB24 pixel data (width × height × 3 bytes).
    image: Vec<u8>,
    width: u32,
    height: u32,
    use_triangles: bool,
    use_circles: bool,
    use_polygons: bool,
    mutation_rate: u32,
    max_bytes: Option<usize>,
    /// Default SVG output path; `process --output-svg` overrides this.
    output_svg: PathBuf,
    max_shapes: usize,
    initial_shapes: usize,
    /// User-supplied starting blur sigma; used to reset blur on `process --restart`.
    initial_blur_radius: Option<f32>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Checkpoint {
    config: StoredConfig,
    state: AnnealingState,
    shapes: Vec<Shape>,
}

/// Load an image (any format supported by the image crate) and return it as an
/// RGB24 byte buffer plus dimensions.
fn load_png(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .into_rgb8();
    let width = img.width();
    let height = img.height();
    Ok((img.into_raw(), width, height))
}

/// Write an SVG string to a file.
fn write_svg(path: &Path, svg: &str) -> Result<()> {
    fs::write(path, svg).with_context(|| format!("failed to write {}", path.display()))
}

/// Save config, annealing state and shapes to a binary checkpoint file.
fn save_binary(
    path: &Path,
    config: &StoredConfig,
    state: &AnnealingState,
    shapes: &[Shape],
) -> Result<()> {
    let checkpoint = Checkpoint {
        config: config.clone(),
        state: state.clone(),
        shapes: shapes.to_vec(),
    };
    let encoded = bincode::serde::encode_to_vec(&checkpoint, bincode::config::standard())
        .context("failed to encode checkpoint")?;
    fs::write(path, encoded).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Load a binary checkpoint. Returns `None` if the file does not exist (fresh start).
///
/// Fails with a clear message on corrupt or incompatible data — run `shapeme setup` to create
/// a new checkpoint.
fn load_binary(path: &Path) -> Result<Option<(StoredConfig, AnnealingState, Vec<Shape>)>> {
    if !path.exists() {
        return Ok(None);
    }
    let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let (checkpoint, _): (Checkpoint, _) =
        bincode::serde::decode_from_slice(&data, bincode::config::standard()).with_context(
            || {
                format!(
                    "failed to decode checkpoint at {} — run `shapeme setup` to create a new one",
                    path.display()
                )
            },
        )?;
    let config = checkpoint.config;
    anyhow::ensure!(
        config.image.len() == (config.width * config.height * 3) as usize,
        "checkpoint image dimensions do not match pixel data — run `shapeme setup` to recreate"
    );
    Ok(Some((config, checkpoint.state, checkpoint.shapes)))
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
    let use_polygons = args.use_polygons != 0;
    if !use_triangles && !use_circles && !use_polygons {
        anyhow::bail!(
            "at least one of --use-triangles, --use-circles, or --use-polygons must be non-zero"
        );
    }

    let (raw_pixels, raw_width, raw_height) = load_png(&args.input)?;
    let (image, width, height) = if args.max_dimension > 0 {
        scale_image(raw_pixels, raw_width, raw_height, args.max_dimension)
    } else {
        (raw_pixels, raw_width, raw_height)
    };
    tracing::info!(width, height, "image loaded");

    let max_shapes = args.max_shapes.max(args.initial_shapes);

    let config = StoredConfig {
        image,
        width,
        height,
        use_triangles,
        use_circles,
        use_polygons,
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
    tracing::info!(path = %args.checkpoint.display(), "checkpoint written");
    Ok(())
}

struct DisplayUpdate {
    shapes: Vec<Shape>,
    blur: Option<f32>,
    diff: f32,
    max_shapes_incremental: usize,
    generation: i64,
    temperature: f32,
}

/// Result of a single parallel annealing batch.
struct BatchResult {
    /// Monotone-best shapes seen during this batch.
    absbest_shapes: Vec<Shape>,
    /// Blur radius corresponding to `absbest_shapes`.
    absbest_blur: Option<f32>,
    /// Full annealing state at batch end; `absbestdiff` is the lowest diff seen.
    state: AnnealingState,
}

#[allow(
    clippy::too_many_arguments,
    reason = "batch parameters, config reference, starting state, and progress channel are all necessary"
)]
fn run_batch(
    config: &StoredConfig,
    start_shapes: &[Shape],
    start_blur: Option<f32>,
    start_state: &AnnealingState,
    batch_size: u32,
    reheated: bool,
    tx: &mpsc::Sender<DisplayUpdate>,
) -> BatchResult {
    let mut rng = SmallRng::from_os_rng();
    let mut fb = vec![0u8; (config.width * config.height * 3) as usize];

    let mut state = start_state.clone();
    if reheated {
        state.temperature = 0.10;
    }

    let mut best_shapes: Vec<Shape> = start_shapes.to_vec();
    let mut best_blur: Option<f32> = start_blur;
    let mut absbest_shapes: Vec<Shape> = start_shapes.to_vec();
    let mut absbest_blur: Option<f32> = start_blur;
    let mut bestdiff = state.absbestdiff;

    let width = config.width;
    let height = config.height;
    let max_shapes = state.max_shapes;
    let mutation_rate = config.mutation_rate.min(1000);
    let use_triangles = config.use_triangles;
    let use_circles = config.use_circles;
    let use_polygons = config.use_polygons;

    for _ in 0..batch_size {
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
            use_polygons,
        );

        let (effective, effective_blur) = if let Some(max_bytes) = config.max_bytes {
            let mut s = candidate.active().to_vec();
            let mut eff_blur = candidate.blur_radius;
            loop {
                let svg = build_svg(&s, width, height, eff_blur, true);
                if svg_to_data_url(&svg).len() <= max_bytes || s.is_empty() {
                    break;
                }
                // Before dropping a shape, check whether removing blur alone brings us within budget.
                // Shapes carry more localised visual information than a global blur filter.
                if eff_blur.is_some() {
                    let svg_no_blur = build_svg(&s, width, height, None, true);
                    if svg_to_data_url(&svg_no_blur).len() <= max_bytes {
                        eff_blur = None;
                        break;
                    }
                }
                s.pop();
            }
            (s, eff_blur)
        } else {
            (candidate.active().to_vec(), candidate.blur_radius)
        };

        fb.fill(0);
        draw_shapes(&mut fb, width, height, &effective);

        // Compute diff against the blurred framebuffer so the fitness function matches SVG output
        let blurred_opt = effective_blur.map(|r| apply_blur(&fb, width, height, r));
        let display_buf: &[u8] = blurred_opt.as_deref().unwrap_or(&fb);
        let diff = compute_diff(&config.image, display_buf);
        let percdiff = diff as f32 / (width * height) as f32 / 442.0 * 100.0;

        let accept = percdiff < bestdiff
            || (state.temperature > 0.0
                && rng.random::<f32>() < state.temperature
                && (percdiff - state.absbestdiff) < 2.0 * state.temperature);

        if accept {
            best_shapes = candidate.shapes;
            best_blur = effective_blur;

            if percdiff < bestdiff {
                absbest_shapes.clone_from(&effective);
                absbest_blur = effective_blur;
                state.absbestdiff = percdiff;
                state.blur_radius = effective_blur;
                let _ = tx.send(DisplayUpdate {
                    shapes: effective.clone(),
                    blur: effective_blur,
                    diff: percdiff,
                    max_shapes_incremental: state.max_shapes_incremental,
                    generation: state.generation,
                    temperature: state.temperature,
                });
            }

            bestdiff = percdiff;
        }
    }

    BatchResult {
        absbest_shapes,
        absbest_blur,
        state,
    }
}

fn process(args: &ProcessArgs) -> Result<()> {
    let checkpoint_path = &args.checkpoint;

    let Some((config, mut state, saved_shapes)) = load_binary(checkpoint_path)? else {
        anyhow::bail!(
            "checkpoint {} not found — run `shapeme setup` to create one",
            checkpoint_path.display()
        );
    };
    let config = Arc::new(config);

    let output_svg: &Path = args.output_svg.as_deref().unwrap_or(&config.output_svg);

    let use_triangles = config.use_triangles;
    let use_circles = config.use_circles;
    let use_polygons = config.use_polygons;
    let width = config.width;
    let height = config.height;

    let mut sdl_ctx: Option<SdlCtx> = if args.headless {
        None
    } else {
        Some(init_sdl(width, height)?)
    };

    // Determine starting shapes and blur: fresh start if restarting or no saved shapes yet
    let (init_shapes, init_blur) = if args.restart || saved_shapes.is_empty() {
        let mut rng = SmallRng::from_os_rng();
        let margin = config.initial_blur_radius.map_or(0, |r| r.ceil() as i16);
        let shapes: Vec<Shape> = (0..config.initial_shapes)
            .map(|_| {
                random_shape(
                    &mut rng,
                    width,
                    height,
                    use_triangles,
                    use_circles,
                    use_polygons,
                    margin,
                )
            })
            .collect();
        (shapes, config.initial_blur_radius)
    } else {
        let blur = state.blur_radius;
        (saved_shapes, blur)
    };

    if args.restart {
        state = AnnealingState::new(config.max_shapes, config.initial_shapes);
    }
    state.blur_radius = init_blur;

    let mut absbest_shapes: Vec<Shape> = init_shapes;
    let mut absbest_blur: Option<f32> = init_blur;

    let mut fb = vec![0u8; (width * height * 3) as usize];
    draw_shapes(&mut fb, width, height, &absbest_shapes);

    let mut show_original = false;
    let mut last_display_buf: Vec<u8> =
        absbest_blur.map_or_else(|| fb.clone(), |r| apply_blur(&fb, width, height, r));

    if let Some(ctx) = &mut sdl_ctx {
        let blurred = absbest_blur.map(|r| apply_blur(&fb, width, height, r));
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

    let print_data_url = args.data_url || args.headless;

    loop {
        let round_state = state.clone();
        let round_shapes = absbest_shapes.clone();
        let round_blur = absbest_blur;
        let n_batches = args.parallel_batches;

        let batch_size = args.batch_size;
        let (tx, rx) = mpsc::channel::<DisplayUpdate>();
        let config_arc = Arc::clone(&config);
        let join_handle = std::thread::spawn(move || {
            (0..n_batches)
                .into_par_iter()
                .map_with(tx, |tx_local, i| {
                    run_batch(
                        &config_arc,
                        &round_shapes,
                        round_blur,
                        &round_state,
                        batch_size,
                        n_batches > 1 && i + 1 == n_batches,
                        tx_local,
                    )
                })
                .collect::<Vec<BatchResult>>()
        });

        let mut should_quit = false;
        let mut round_display_diff = state.absbestdiff;

        'round: loop {
            while let Ok(update) = rx.try_recv() {
                if update.diff < round_display_diff {
                    round_display_diff = update.diff;
                    if let Some(ctx) = &mut sdl_ctx {
                        fb.fill(0);
                        draw_shapes(&mut fb, width, height, &update.shapes);
                        let blurred = update.blur.map(|r| apply_blur(&fb, width, height, r));
                        let display = blurred.as_deref().unwrap_or(&fb);
                        last_display_buf.clear();
                        last_display_buf.extend_from_slice(display);
                        if !show_original {
                            let _ = ctx.texture.update(None, display, (width * 3) as usize);
                            let _ = ctx.canvas.copy(&ctx.texture, None, None);
                            ctx.canvas.present();
                        }
                    }
                    tracing::info!(
                        diff_pct = round_display_diff,
                        shapes = update.shapes.len(),
                        max_shapes = update.max_shapes_incremental,
                        generation = update.generation,
                        temperature = update.temperature,
                        blur = ?update.blur,
                        "improved"
                    );
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
                            should_quit = true;
                            break 'round;
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

            if join_handle.is_finished() {
                while let Ok(update) = rx.try_recv() {
                    if update.diff < round_display_diff {
                        round_display_diff = update.diff;
                        if let Some(ctx) = &mut sdl_ctx {
                            fb.fill(0);
                            draw_shapes(&mut fb, width, height, &update.shapes);
                            let blurred = update.blur.map(|r| apply_blur(&fb, width, height, r));
                            let display = blurred.as_deref().unwrap_or(&fb);
                            last_display_buf.clear();
                            last_display_buf.extend_from_slice(display);
                            if !show_original {
                                let _ = ctx.texture.update(None, display, (width * 3) as usize);
                                let _ = ctx.canvas.copy(&ctx.texture, None, None);
                                ctx.canvas.present();
                            }
                        }
                    }
                }
                break 'round;
            }

            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let results = join_handle.join().expect("worker thread panicked");

        if should_quit {
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

        let winner = results
            .into_iter()
            .min_by(|a, b| {
                a.state
                    .absbestdiff
                    .partial_cmp(&b.state.absbestdiff)
                    .unwrap_or(Ordering::Equal)
            })
            .expect("parallel_batches > 0");

        state = winner.state;
        absbest_shapes = winner.absbest_shapes;
        absbest_blur = winner.absbest_blur;

        tracing::info!(
            diff_pct = state.absbestdiff,
            shapes = absbest_shapes.len(),
            max_shapes = state.max_shapes_incremental,
            generation = state.generation,
            temperature = state.temperature,
            blur = ?absbest_blur,
            "round complete"
        );

        if let Some(ctx) = &mut sdl_ctx {
            fb.fill(0);
            draw_shapes(&mut fb, width, height, &absbest_shapes);
            let blurred = absbest_blur.map(|r| apply_blur(&fb, width, height, r));
            let display = blurred.as_deref().unwrap_or(&fb);
            last_display_buf.clear();
            last_display_buf.extend_from_slice(display);
            if !show_original {
                let _ = ctx.texture.update(None, display, (width * 3) as usize);
                let _ = ctx.canvas.copy(&ctx.texture, None, None);
                ctx.canvas.present();
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

        write_svg(
            output_svg,
            &build_svg(&absbest_shapes, width, height, absbest_blur, false),
        )?;
        save_binary(checkpoint_path, &config, &state, &absbest_shapes)?;
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let level = if args.quiet {
        Level::WARN
    } else if args.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(level).init();

    match args.command {
        Command::Setup(setup_args) => setup(&setup_args),
        Command::Process(process_args) => process(&process_args),
    }
}
