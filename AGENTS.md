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
| `libshapeme::oklab`     | sRGB↔OKlab conversion (`srgb_u8_to_oklab`, `oklab_to_srgb_u8`, bulk variants)              |
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

### OKlab perceptually uniform colour space

All gene colours (`Shape` fields, `BackgroundGene`) are stored as OKlab `[f32; 3]` rather
than sRGB `u8` triples.  OKlab is a perceptually uniform space (Björn Ottosson, 2020):
equal distances correspond to equal perceived differences.

- **Diff metric**: `compute_diff` returns RMSE in OKlab space (result in [0, ~1.0],
  multiplied by 100 for the percentage-diff value used by annealing).  The old sRGB
  Euclidean sum divided by `width*height*442` is removed.
- **Colour mutations**: nudge ±0.02 per OKlab channel — perceptually uniform steps.
- **Recombination**: arithmetic mean in OKlab is the perceptually correct midpoint.
- **Framebuffers**: all internal buffers are `Vec<f32>` (3 floats per pixel, OKlab).
- **Output paths**: SDL textures and SVG output convert OKlab → sRGB u8 at the last step.
- **Image load pipeline** (`setup`): PNG → sRGB u8 → `scale_image` (Lanczos3 in sRGB u8)
  → `image_srgb_to_oklab` → OKlab f32 → `StoredConfig.image`.
  Scaling is done in sRGB u8 **before** the OKlab conversion because `imageops::resize`
  clips negative `f32` values to 0, which would zero the negative `a`/`b` channels of
  cool/blue pixels and produce a warm (sepia) bias in the stored reference.
- **Checkpoint compat**: `StoredConfig.image` changed from `Vec<u8>` to `Vec<f32>`.
  Existing checkpoints are incompatible; re-run `shapeme setup`.
- **Legacy V1 checkpoint migration** (flat `Vec<Shape>`) was removed — those checkpoints
  could not be decoded with the new shape type regardless.

### Diff percentage constant: 442, not 422 (historical, now removed)

Pre-OKlab, the C code comment said "422" but the computation used **442** (`width*height*442`).
With the OKlab RMSE metric this constant is no longer needed; the note is kept for history.

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

Checkpoints created before the cost model was introduced (those with `AnnealingState.max_cost ≤ 512`,
i.e. `max_shapes ≤ 23` when interpreted as a raw shape count) have shape-count values in the
`max_cost` / `max_cost_incremental` fields rather than cost values. These decode without error but
produce wrong behaviour (effectively a very tight budget). Re-run with `process --restart` to
rebuild the annealing state with correct cost values derived from `StoredConfig.max_shapes`.

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

### Polygon cost model and vertex cap

Gene budget is tracked in approximate bincode byte units rather than raw shape count.
`TRIANGLE_COST = 21`, `CIRCLE_COST = 18`, `POLYGON_BASE_COST = 16`, `POLYGON_VERTEX_COST = 4`.
(Costs increased from pre-OKlab values because colours are now 12-byte `[f32; 3]` not 3-byte u8;
bincode v2 uses varint encoding so exact sizes vary with coordinate magnitude — the constants
approximate typical values for a 256×256 image.)
CLI `--max-shapes N` is stored as `max_cost = N × TRIANGLE_COST` in `AnnealingState` so the
user-visible "shapes" concept is preserved while the budget correctly penalises complex polygons.

`max_polygon_vertices` in `MutationConfig` (set to `max_shapes.max(6)`) caps the `split_edge`
mutation, preventing unbounded polygon growth. When the cap is reached, `split_edge` falls back
to a small-nudge mutation instead.

Polygon vertices are sorted by angle from the centroid on every `normalize()` call, eliminating
self-intersecting edges. Clamping runs first so the centroid is computed from the final
in-bounds positions.

### Z-ordering

`ShapeGene` carries a `z_order: u16` field. `draw_genes` sorts genes ascending by `z_order` before
rasterising, so layering is independent of `Vec` insertion order. This makes recombination-via-
crossover safe: two parents with different gene orderings produce a child that renders consistently.
`ShapeGenome::mutate` has a ~5% chance to swap two genes' z-orders (reordering), and `ShapeGene::mutate`
has a ~10% chance to nudge its own z_order by ±1000.
