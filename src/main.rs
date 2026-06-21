mod annealing;
mod io;
mod render;
mod shapes;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;

use annealing::{AnnealingState, ShapeSet, mutate_shapes};
use io::{load_binary, load_png, save_binary, save_svg};
use render::{compute_diff, draw_shapes};
use shapes::random_shape;

#[derive(Parser, Debug)]
#[command(
    name = "shapeme",
    about = "Approximate images with shapes via simulated annealing"
)]
struct Args {
    /// Input PNG image
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
    println!("Image {width}×{height}");

    let sdl = sdl2::init().map_err(|e| anyhow::anyhow!("SDL2 init: {e}"))?;
    let video = sdl
        .video()
        .map_err(|e| anyhow::anyhow!("SDL2 video: {e}"))?;
    let window = video
        .window("Shapeme", width, height)
        .opengl()
        .build()
        .context("SDL2 window")?;
    let mut canvas = window.into_canvas().build().context("SDL2 canvas")?;
    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator
        .create_texture_streaming(PixelFormatEnum::RGB24, width, height)
        .context("SDL2 texture")?;
    let mut event_pump = sdl
        .event_pump()
        .map_err(|e| anyhow::anyhow!("SDL2 events: {e}"))?;

    let mut rng = SmallRng::from_os_rng();

    let (mut state, mut best_shapes) = if !args.restart {
        match load_binary(&args.checkpoint)? {
            Some((s, shapes)) => {
                println!("Loaded {} shapes from checkpoint", shapes.len());
                (s, shapes)
            }
            None => {
                let s = AnnealingState::new(max_shapes, args.initial_shapes);
                let shapes = (0..args.initial_shapes)
                    .map(|_| random_shape(&mut rng, width, height, use_triangles, use_circles))
                    .collect();
                (s, shapes)
            }
        }
    } else {
        let s = AnnealingState::new(max_shapes, args.initial_shapes);
        let shapes = (0..args.initial_shapes)
            .map(|_| random_shape(&mut rng, width, height, use_triangles, use_circles))
            .collect();
        (s, shapes)
    };

    let mut absbest_shapes = best_shapes.clone();

    let show_fb = |texture: &mut sdl2::render::Texture,
                   canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
                   fb: &[u8]| {
        let _ = texture.update(None, fb, (width * 3) as usize);
        let _ = canvas.copy(texture, None, None);
        canvas.present();
    };

    let mut fb = vec![0u8; (width * height * 3) as usize];
    draw_shapes(&mut fb, width, height, &best_shapes);
    show_fb(&mut texture, &mut canvas, &fb);
    std::thread::sleep(std::time::Duration::from_secs(1));
    show_fb(&mut texture, &mut canvas, &image);
    std::thread::sleep(std::time::Duration::from_secs(1));

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
        candidate.shapes = best_shapes.clone();
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

        fb.fill(0);
        draw_shapes(&mut fb, width, height, candidate.active());
        let diff = compute_diff(&image, &fb);
        let percdiff = diff as f32 / (width * height) as f32 / 442.0 * 100.0;

        let accept = percdiff < bestdiff
            || (state.temperature > 0.0
                && rng.random::<f32>() < state.temperature
                && (percdiff - state.absbestdiff) < 2.0 * state.temperature);

        if accept {
            best_shapes = candidate.shapes;

            if percdiff < bestdiff {
                absbest_shapes = best_shapes.clone();
                state.absbestdiff = percdiff;
            }

            println!(
                "Diff {:.4}% (shapes:{}, max:{}, gen:{}, temp:{:.5})",
                percdiff,
                best_shapes.len(),
                state.max_shapes_incremental,
                state.generation,
                state.temperature
            );

            bestdiff = percdiff;
            show_fb(&mut texture, &mut canvas, &fb);
        }

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Q | Keycode::Escape),
                    ..
                } => {
                    save_svg(&args.output_svg, &absbest_shapes, width, height)?;
                    save_binary(&args.checkpoint, &state, &absbest_shapes)?;
                    return Ok(());
                }
                _ => {}
            }
        }

        if state.generation % 100 == 0 {
            save_svg(&args.output_svg, &absbest_shapes, width, height)?;
            save_binary(&args.checkpoint, &state, &absbest_shapes)?;
        }
    }
}
