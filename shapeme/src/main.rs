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

use libshapeme::annealing::AnnealingState;
use libshapeme::gene::{MutationConfig, ShapeGene, TRIANGLE_COST};
use libshapeme::genome::{Genome, ShapeGenome};
use libshapeme::render::{apply_blur, draw_shapes, scale_image};
use libshapeme::shapes::{Shape, random_shape};
use libshapeme::svg::{build_svg, build_svg_from_genome, svg_to_data_url};

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
    /// Genomes from top-K batches are recombined each round (set 1 to disable)
    #[arg(long, default_value = "3")]
    top_k: usize,
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

/// Current checkpoint format (V2): stores a full `ShapeGenome`.
#[derive(serde::Serialize, serde::Deserialize)]
struct Checkpoint {
    config: StoredConfig,
    state: AnnealingState,
    genome: ShapeGenome,
}

/// Legacy checkpoint format (V1): flat `Vec<Shape>` from before z-ordering.
///
/// Kept for read-only migration; `save_binary` always writes `Checkpoint`.
#[derive(serde::Serialize, serde::Deserialize)]
struct LegacyCheckpoint {
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

/// Save config, annealing state and genome to a binary checkpoint file.
fn save_binary(
    path: &Path,
    config: &StoredConfig,
    state: &AnnealingState,
    genome: &ShapeGenome,
) -> Result<()> {
    let checkpoint = Checkpoint {
        config: config.clone(),
        state: state.clone(),
        genome: genome.clone(),
    };
    let encoded = bincode::serde::encode_to_vec(&checkpoint, bincode::config::standard())
        .context("failed to encode checkpoint")?;
    fs::write(path, encoded).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Load a binary checkpoint. Returns `None` if the file does not exist (fresh start).
///
/// Attempts the current V2 format first; falls back to the legacy V1 format (flat
/// `Vec<Shape>`) and migrates by assigning `z_order = index`.  Fails with a clear message
/// on corrupt or incompatible data.
fn load_binary(path: &Path) -> Result<Option<(StoredConfig, AnnealingState, ShapeGenome)>> {
    if !path.exists() {
        return Ok(None);
    }
    let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;

    // Try current V2 format first
    if let Ok((checkpoint, _)) =
        bincode::serde::decode_from_slice::<Checkpoint, _>(&data, bincode::config::standard())
    {
        let config = checkpoint.config;
        anyhow::ensure!(
            config.image.len() == (config.width * config.height * 3) as usize,
            "checkpoint image dimensions do not match pixel data — run `shapeme setup` to recreate"
        );
        return Ok(Some((config, checkpoint.state, checkpoint.genome)));
    }

    // Fall back to legacy V1 format, migrating shapes to ShapeGenome
    if let Ok((legacy, _)) =
        bincode::serde::decode_from_slice::<LegacyCheckpoint, _>(&data, bincode::config::standard())
    {
        tracing::info!("migrating legacy V1 checkpoint to V2 (z-order assigned by original draw order)");
        let config = legacy.config;
        anyhow::ensure!(
            config.image.len() == (config.width * config.height * 3) as usize,
            "checkpoint image dimensions do not match pixel data — run `shapeme setup` to recreate"
        );
        let genome = ShapeGenome::from_shapes(legacy.shapes);
        return Ok(Some((config, legacy.state, genome)));
    }

    anyhow::bail!(
        "failed to decode checkpoint at {} — run `shapeme setup` to create a new one",
        path.display()
    );
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
    genome: &ShapeGenome,
    state: &AnnealingState,
    print_data_url: bool,
) -> Result<()> {
    write_svg(output_svg, &build_svg_from_genome(genome, config.width, config.height, false))?;
    save_binary(checkpoint, config, state, genome)?;
    if print_data_url {
        let compact = build_svg_from_genome(genome, config.width, config.height, true);
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

    let mut state = AnnealingState::new(max_shapes * TRIANGLE_COST, args.initial_shapes * TRIANGLE_COST);
    state.blur_radius = args.blur_radius;

    let empty_genome = ShapeGenome::from_shapes(vec![]);
    save_binary(&args.checkpoint, &config, &state, &empty_genome)?;
    tracing::info!(path = %args.checkpoint.display(), "checkpoint written");
    Ok(())
}

struct DisplayUpdate {
    shapes: Vec<Shape>,
    blur: Option<f32>,
    background: (u8, u8, u8),
    diff: f32,
    max_cost_incremental: usize,
    generation: i64,
    temperature: f32,
}

/// Result of a single parallel annealing batch.
struct BatchResult {
    /// All-time best genome seen during this batch.
    best_genome: ShapeGenome,
    /// Full annealing state at batch end; `absbestdiff` is the lowest diff seen.
    state: AnnealingState,
}

/// Trim a genome so that its compact SVG data URL fits within `max_bytes`.
///
/// Shapes are sorted by z-order; the topmost (highest z) shape is removed first.
/// Blur is stripped before shapes, since removing the global filter is cheaper visually.
fn trim_genome_to_budget(genome: ShapeGenome, width: u32, height: u32, max_bytes: usize) -> ShapeGenome {
    let mut shapes = genome.shapes;
    shapes.sort_unstable_by_key(|g| g.z_order);
    let mut blur = genome.blur;
    let background = genome.background;

    loop {
        let flat_shapes: Vec<&Shape> = shapes.iter().map(|g| &g.shape).collect();
        let svg = build_svg(
            &flat_shapes.iter().copied().cloned().collect::<Vec<_>>(),
            width,
            height,
            blur.as_ref().map(|b| b.radius),
            (background.r, background.g, background.b),
            true,
        );
        if svg_to_data_url(&svg).len() <= max_bytes || shapes.is_empty() {
            break;
        }
        // Try dropping blur alone before removing a shape
        if blur.is_some() {
            let svg_no_blur = build_svg(
                &flat_shapes.iter().copied().cloned().collect::<Vec<_>>(),
                width,
                height,
                None,
                (background.r, background.g, background.b),
                true,
            );
            if svg_to_data_url(&svg_no_blur).len() <= max_bytes {
                blur = None;
                break;
            }
        }
        // Remove the topmost (highest z-order) shape — shapes is sorted ascending
        shapes.pop();
    }

    ShapeGenome { shapes, blur, background }
}

#[allow(
    clippy::too_many_arguments,
    reason = "batch parameters, config reference, starting state, and progress channel are all necessary"
)]
fn run_batch(
    config: &StoredConfig,
    start_genome: &ShapeGenome,
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

    let width = config.width;
    let height = config.height;
    let mutation_config = MutationConfig {
        width,
        height,
        margin: start_genome.blur_radius().map_or(0, |r| r.ceil() as i16),
        mutation_rate: config.mutation_rate.min(1000),
        use_triangles: config.use_triangles,
        use_circles: config.use_circles,
        use_polygons: config.use_polygons,
        max_polygon_vertices: config.max_shapes.max(6),
    };

    let mut best_genome = start_genome.clone();
    let mut absbest_genome = start_genome.clone();
    let mut bestdiff = state.absbestdiff;

    for _ in 0..batch_size {
        state.generation += 1;

        if state.temperature > 0.0 && state.generation % 10 == 0 {
            state.temperature -= 0.00001;
            if state.temperature < 0.0 {
                state.temperature = 0.0;
            }
        }

        if state.generation % 1000 == 0
            && state.max_cost_incremental < state.max_cost
            && best_genome.total_cost() + TRIANGLE_COST > state.max_cost_incremental
        {
            state.max_cost_incremental += TRIANGLE_COST;
        }

        let candidate = best_genome.mutate(&mut rng, &state, &mutation_config);

        let effective = if let Some(max_bytes) = config.max_bytes {
            trim_genome_to_budget(candidate, width, height, max_bytes)
        } else {
            candidate
        };

        let percdiff = effective.fitness(&config.image, width, height, &mut fb);

        let accept = percdiff < bestdiff
            || (state.temperature > 0.0
                && rng.random::<f32>() < state.temperature
                && (percdiff - state.absbestdiff) < 2.0 * state.temperature);

        if accept {
            if percdiff < bestdiff {
                absbest_genome = effective.clone();
                state.absbestdiff = percdiff;
                state.blur_radius = effective.blur_radius();
                let sorted_shapes: Vec<Shape> =
                    effective.sorted_shapes().into_iter().cloned().collect();
                let _ = tx.send(DisplayUpdate {
                    shapes: sorted_shapes,
                    blur: effective.blur_radius(),
                    background: effective.background_color(),
                    diff: percdiff,
                    max_cost_incremental: state.max_cost_incremental,
                    generation: state.generation,
                    temperature: state.temperature,
                });
            }
            best_genome = effective;
            bestdiff = percdiff;
        }
    }

    BatchResult {
        best_genome: absbest_genome,
        state,
    }
}

fn process(args: &ProcessArgs) -> Result<()> {
    let checkpoint_path = &args.checkpoint;

    let Some((config, mut state, saved_genome)) = load_binary(checkpoint_path)? else {
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

    // Determine starting genome: fresh start if restarting or no saved shapes yet
    let init_genome = if args.restart || saved_genome.shapes.is_empty() {
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
        let genes: Vec<ShapeGene> = shapes
            .into_iter()
            .enumerate()
            .map(|(i, shape)| ShapeGene {
                shape,
                z_order: i as u16,
            })
            .collect();
        let blur = config.initial_blur_radius.map(|r| libshapeme::gene::BlurGene { radius: r });
        ShapeGenome { shapes: genes, blur, background: libshapeme::gene::BackgroundGene::default() }
    } else {
        saved_genome
    };

    if args.restart {
        state = AnnealingState::new(config.max_shapes * TRIANGLE_COST, config.initial_shapes * TRIANGLE_COST);
    }
    state.blur_radius = init_genome.blur_radius();

    let mut absbest_genome = init_genome;

    let mut fb = vec![0u8; (width * height * 3) as usize];
    let (bg_r, bg_g, bg_b) = absbest_genome.background_color();
    for pixel in fb.chunks_exact_mut(3) {
        pixel[0] = bg_r; pixel[1] = bg_g; pixel[2] = bg_b;
    }
    let absbest_sorted: Vec<Shape> = absbest_genome.sorted_shapes().into_iter().cloned().collect();
    draw_shapes(&mut fb, width, height, &absbest_sorted);

    let mut show_original = false;
    let mut last_display_buf: Vec<u8> = absbest_genome
        .blur_radius()
        .map_or_else(|| fb.clone(), |r| apply_blur(&fb, width, height, r));

    if let Some(ctx) = &mut sdl_ctx {
        let blurred = absbest_genome.blur_radius().map(|r| apply_blur(&fb, width, height, r));
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
    let top_k_n = args.top_k.max(1);

    let mut rng = SmallRng::from_os_rng();

    loop {
        let round_state = state.clone();
        let round_genome = absbest_genome.clone();
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
                        &round_genome,
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
                        let (bg_r, bg_g, bg_b) = update.background;
                        for pixel in fb.chunks_exact_mut(3) {
                            pixel[0] = bg_r; pixel[1] = bg_g; pixel[2] = bg_b;
                        }
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
                        max_cost = update.max_cost_incremental,
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
                            let (bg_r, bg_g, bg_b) = update.background;
                            for pixel in fb.chunks_exact_mut(3) {
                                pixel[0] = bg_r; pixel[1] = bg_g; pixel[2] = bg_b;
                            }
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
                &absbest_genome,
                &state,
                print_data_url,
            )?;
            return Ok(());
        }

        // Sort batches by diff ascending; best is first
        let mut results_sorted = results;
        results_sorted.sort_unstable_by(|a, b| {
            a.state
                .absbestdiff
                .partial_cmp(&b.state.absbestdiff)
                .unwrap_or(Ordering::Equal)
        });

        // Recombination phase: cross the top-K batch winners
        let top_n = top_k_n.min(results_sorted.len());
        let top = &results_sorted[..top_n];
        let mut best_offspring_diff: Option<f32> = None;
        let mut best_offspring: Option<ShapeGenome> = None;

        if top_n >= 2 {
            let mut scratch = vec![0u8; (width * height * 3) as usize];
            for i in 0..top.len() {
                for j in (i + 1)..top.len() {
                    let raw_child = top[i].best_genome.recombine(&top[j].best_genome, &mut rng);
                    // Trim to budget before fitness, same as run_batch does for every candidate
                    let child = if let Some(max_bytes) = config.max_bytes {
                        trim_genome_to_budget(raw_child, width, height, max_bytes)
                    } else {
                        raw_child
                    };
                    let child_diff = child.fitness(&config.image, width, height, &mut scratch);
                    if best_offspring_diff.is_none_or(|d| child_diff < d) {
                        best_offspring_diff = Some(child_diff);
                        best_offspring = Some(child);
                    }
                }
            }
        }

        // Winner is the best batch result, potentially beaten by a recombination offspring
        let batch_winner = results_sorted.into_iter().next().expect("parallel_batches > 0");
        let (winner_genome, winner_state) = if let (Some(offspring), Some(offspring_diff)) =
            (best_offspring, best_offspring_diff)
        {
            if offspring_diff < batch_winner.state.absbestdiff {
                tracing::info!(offspring_diff, batch_diff = batch_winner.state.absbestdiff, "recombination offspring wins round");
                let mut offspring_state = batch_winner.state.clone();
                offspring_state.absbestdiff = offspring_diff;
                offspring_state.blur_radius = offspring.blur_radius();
                (offspring, offspring_state)
            } else {
                (batch_winner.best_genome, batch_winner.state)
            }
        } else {
            (batch_winner.best_genome, batch_winner.state)
        };

        state = winner_state;
        absbest_genome = winner_genome;

        tracing::info!(
            diff_pct = state.absbestdiff,
            shapes = absbest_genome.shapes.len(),
            max_cost = state.max_cost_incremental,
            generation = state.generation,
            temperature = state.temperature,
            blur = ?absbest_genome.blur_radius(),
            "round complete"
        );

        if let Some(ctx) = &mut sdl_ctx {
            let (bg_r, bg_g, bg_b) = absbest_genome.background_color();
            for pixel in fb.chunks_exact_mut(3) {
                pixel[0] = bg_r; pixel[1] = bg_g; pixel[2] = bg_b;
            }
            let sorted: Vec<Shape> = absbest_genome.sorted_shapes().into_iter().cloned().collect();
            draw_shapes(&mut fb, width, height, &sorted);
            let blurred = absbest_genome.blur_radius().map(|r| apply_blur(&fb, width, height, r));
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
                    &absbest_genome,
                    &state,
                    print_data_url,
                )?;
                return Ok(());
            }
        }

        write_svg(
            output_svg,
            &build_svg_from_genome(&absbest_genome, width, height, false),
        )?;
        save_binary(checkpoint_path, &config, &state, &absbest_genome)?;
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
