# shapeme — developer notes

Rust port of [antirez/shapeme](https://github.com/antirez/shapeme), a simulated
annealing image approximation tool. The original C source lives at
`/Users/axa/dev/native/shapeme/shapeme.c` and is the authoritative reference
for algorithm behaviour.

## Workspace layout

The repository is a Cargo workspace with two crates:

- **`libshapeme/`** — core algorithm library. Works on in-memory state only; no file I/O,
  no console output, no SDL2. Uses `tracing` for debug/trace logging.
- **`shapeme/`** — CLI binary. Owns all file I/O, SDL2 UI, CLI argument parsing, and
  the `tracing-subscriber` initialisation.

The dividing line: _file access in the binary, the library works on state_.

## Architecture

| Crate / Module          | Responsibility                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| `libshapeme::shapes`    | `Shape` enum (Triangle/Circle/Polygon), normalisation, mutation, random generation          |
| `libshapeme::gene`      | `Gene` trait, `ShapeGene` (Shape + z_order), `BlurGene`, `MutationConfig`                  |
| `libshapeme::genome`    | `Genome` trait, `ShapeGenome` (fitness/mutate/recombine); replaces the old `ShapeSet`       |
| `libshapeme::render`    | Framebuffer rasterisation, `draw_genes`, `apply_blur`, `compute_diff`, `scale_image`        |
| `libshapeme::annealing` | `AnnealingState` only (mutation logic lives in `ShapeGenome::mutate`)                       |
| `libshapeme::svg`       | `build_svg`, `build_svg_from_genome`, `svg_to_data_url` (no file writes)                    |
| `shapeme::main`         | CLI (clap `setup`/`process`), SDL2 init and event loop, file I/O, checkpoint I/O            |

## Key design decisions

### SDL2 — bundled feature + cmake policy workaround

`sdl2` uses the `bundled` feature to compile SDL2 from source, removing the
system library dependency. CMake 4.x broke compatibility with the old
`cmake_minimum_required` version in the bundled SDL2 source. The workaround is
in `.cargo/config.toml`:

```toml
[env]
CMAKE_POLICY_VERSION_MINIMUM = "3.5"
```

This is harmless on older cmake versions and should be retained until sdl2-sys
ships a version of the bundled SDL2 that specifies a modern cmake minimum.

### Diff percentage constant: 442, not 422

The C code comment says "422" but the computation uses **442** (`width*height*442`).
442 is the correct maximum per-pixel distance: `⌊√(255²×3)⌋ = 441`, rounded up
to 442. The Rust port uses 442 and the comment corrects the original typo.

### Setup/process split

The CLI is split into two subcommands:

- `shapeme setup <checkpoint> <input> --output-svg <path> [options]` — loads and scales the image
  once, stores it with all config in a `StoredConfig` inside the checkpoint, then exits.
- `shapeme process <checkpoint> [options]` — loads everything from the checkpoint and runs the
  annealing loop. No image path or shape flags needed on repeated invocations.

This means all run-to-run flags (triangles, circles, blur, max-shapes, etc.) are frozen at setup
time and do not need to be repeated. `process --restart` re-initialises shapes and annealing state
but keeps the stored config and image.

### Evolved blur

Blur is represented as `Option<BlurGene>` on `ShapeGenome`. Each generation, `ShapeGenome::mutate`
has a ~5% chance to nudge the blur radius up or down by up to 2.0, introduce it from `None`, or
remove it when `BlurGene::mutate` returns a radius below 0.1. The SDL window and the diff
computation both use the blurred framebuffer, matching SVG output.

`AnnealingState::blur_radius` is kept in sync with the current best genome's blur so checkpoints
remain backward-readable without pulling in genome types.

### Binary checkpoint format

Uses `bincode` v2 API (`bincode::serde::encode_to_vec` /
`bincode::serde::decode_from_slice` with `bincode::config::standard()`).

`load_binary` tries the current V2 format (`Checkpoint { config, state, genome: ShapeGenome }`)
first, then falls back to the legacy V1 format (`LegacyCheckpoint { config, state, shapes: Vec<Shape> }`).
On a V1 load, each `Shape` is migrated to `ShapeGene { shape, z_order: index }`, preserving the
original draw order. `save_binary` always writes V2.

Checkpoints from before the `StoredConfig` field was added are intentionally incompatible and fail
with a message directing the user to `shapeme setup`.

### Triangle normalisation

Bubble-sort of (x,y) vertex pairs by y-coordinate, identical to the C
algorithm. Required by the triangle rasteriser, which assumes y1 ≤ y2 ≤ y3.

### Alpha range

Alpha values are stored as integers in 1–100 and divided by 100.0 when
blending: `dst = α·src + (1−α)·dst`. This matches the original.

### `jj pre-commit` known limitation

`jj pre-commit` fails with "Invalid revision range 0000000…" because the
ancestry of this repo reaches the jj root (shown as `zzzzzzzz 00000000`), and
the pre-commit tool does not support the null git SHA as a diff base. Use
`pre-commit run --all-files` instead, which works correctly.

### `bestdiff` initialisation on checkpoint load — intentional deviation

The C code initialises `bestdiff = 100` before calling `loadBinary`. After
loading, `state.absbestdiff` holds the saved value but `bestdiff` is still 100.
This means the first post-load generation unconditionally accepts a mutation and
resets the quality tracking — a bug. The Rust port fixes this by initialising
`bestdiff = state.absbestdiff` after the checkpoint is loaded.

### Annealing acceptance

A candidate is accepted when:

- `percdiff < bestdiff`, **or**
- temperature > 0, `rand < temperature`, and `(percdiff − absbestdiff) < 2·temperature`

The second condition allows occasional uphill moves to escape local minima,
controlled by a temperature that decays by 0.00001 every 10 generations.

### Parallel annealing batches + recombination (Rayon)

The annealing loop runs `--parallel-batches` (default 10) independent trajectories per _round_
via Rayon, each running `--batch-size` (default 200) generations. All batches start from the
current absolute-best `ShapeGenome` and annealing state.

After all batches complete, the top-K batch winners (by `absbestdiff`, `--top-k` default 3) are
crossed over pairwise via `ShapeGenome::recombine`. The best offspring is compared against the best
batch winner; whichever has the lower diff becomes the next round's starting genome. Set `--top-k 1`
to disable recombination.

The last batch in each round is _re-heated_ (temperature reset to 0.10) to enable large
exploratory jumps that cold batches would reject. Re-heating only applies when
`parallel_batches > 1`; a single-batch run is strictly serial.

Generation count advances by `batch_size` per round (not `batch_size × parallel_batches`),
so temperature decay reflects rounds-of-wall-time, not total candidate evaluations. Each
batch allocates its own framebuffer and `SmallRng`; no shared mutable state is needed.

SDL display and checkpoint saves happen on the main thread between rounds.

### Z-ordering

`ShapeGene` carries a `z_order: u16` field. `draw_genes` sorts genes ascending by `z_order` before
rasterising, so layering is independent of `Vec` insertion order. This makes recombination-via-
crossover safe: two parents with different gene orderings produce a child that renders consistently.
`ShapeGenome::mutate` has a ~5% chance to swap two genes' z-orders (reordering), and `ShapeGene::mutate`
has a ~10% chance to nudge its own z_order by ±1000.
