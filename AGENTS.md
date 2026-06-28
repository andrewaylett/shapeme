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

| Crate / Module          | Responsibility                                                                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `libshapeme::shapes`    | Utility helpers: `random_oklab_color`, `nudge_oklab`, `rand_between`, `clamp_coord`, shape-kind selection                           |
| `libshapeme::gene`      | `Gene` trait; `TriangleGene`, `CircleGene`, `PolygonGene` structs; `ShapeGene` enum; `BlurGene`, `BackgroundGene`, `MutationConfig` |
| `libshapeme::genome`    | `Genome` trait, `ShapeGenome` (fitness/mutate/recombine); replaces the old `ShapeSet`                                               |
| `libshapeme::render`    | Framebuffer rasterisation, `draw_genes`, `apply_blur`, `compute_diff`, `scale_image`                                                |
| `libshapeme::annealing` | `AnnealingState` only (mutation logic lives in `ShapeGenome::mutate`)                                                               |
| `libshapeme::oklab`     | sRGB↔OKlab conversion (`srgb_u8_to_oklab`, `oklab_to_srgb_u8`, bulk variants)                                                      |
| `libshapeme::svg`       | `build_svg(&[ShapeGene], ...)`, `build_svg_from_genome`, `svg_to_data_url` (no file writes)                                         |
| `shapeme::main`         | CLI (clap `setup`/`process`), SDL2 init and event loop, file I/O, checkpoint I/O                                                    |

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

### Typed gene structs

The `Shape` enum is gone. Each shape type is its own struct (`TriangleGene`, `CircleGene`,
`PolygonGene`), each owning its fields directly and implementing `Gene` independently.
`ShapeGene` is now an enum wrapping these three types. Mutation code contains no
shape-type match — each struct's `Gene::mutate` knows exactly which fields it has.

`PolygonGene` owns the polygon-specific operations: `split()` (divide into two
colour-diverged halves) and `angle_crossover()` (recombine two polygons at a random
dividing angle). These are inherent methods on `PolygonGene`, not free functions.

`ShapeGene` helpers:

- `z_order() -> u16` / `set_z_order(u16)` — z-order access without destructuring
- `cost() -> usize` — budget accounting
- `random(rng, config) -> Self` — random gene of any enabled type
- `random_full(rng, config, z_order) -> Self` — random gene at a specific z-order (pub, used by main.rs init)

`shapes.rs` is now a small utility module: `random_oklab_color`, `nudge_oklab`,
`rand_between`, `clamp_coord`, `ShapeKind`, `select_shape_type`. No types, no `Shape`.

**Checkpoint incompatibility**: the `ShapeGene` enum serialisation is different from the
old `ShapeGene { shape: Shape, z_order }` struct. Old checkpoints fail to decode and
display the existing "re-run `shapeme setup`" message.

### OKlab colour sampling

Colours are sampled directly in OKlab space (`random_oklab_color`): L ∈ [0,1],
a/b ∈ [−0.4, 0.4]. This produces a perceptually uniform colour distribution, unlike
the old approach of sampling random sRGB bytes and converting.

Nudges are per-channel with different scales: L ±0.01 (tight, narrower range),
a/b ±0.05 (wider, chromatic channels). Previously all channels used ±0.02.
`BackgroundGene::mutate` uses the same `random_oklab_color` and `nudge_oklab` helpers.

### OKlab perceptually uniform colour space

All gene colours (`TriangleGene`, `CircleGene`, `PolygonGene`, `BackgroundGene`) are
stored as OKlab `[f32; 3]` rather than sRGB `u8` triples. OKlab is a perceptually
uniform space (Björn Ottosson, 2020): equal distances correspond to equal perceived
differences.

- **Diff metric**: `compute_diff` returns RMSE in OKlab space (result in [0, ~1.0],
  multiplied by 100 for the percentage-diff value used by annealing). The old sRGB
  Euclidean sum divided by `width*height*442` is removed.
- **Colour mutations**: nudge per-channel (L ±0.01, a/b ±0.05) — perceptually uniform steps.
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

`load_binary` tries to decode `Checkpoint { config, state, genome: ShapeGenome }`.
Any failure (wrong type, old struct layout, old `Shape`-based layout) produces a clear
"re-run `shapeme setup`" message. `save_binary` always writes the current format.

The `ShapeGene` type changed from a struct (`{ shape: Shape, z_order: u16 }`) to an enum
(`Triangle(TriangleGene) | Circle(CircleGene) | Polygon(PolygonGene)`), making all
previously written checkpoints incompatible at the bincode level.

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

`max_polygon_vertices` in `MutationConfig` caps the `split_edge` mutation, preventing unbounded
polygon growth. When the cap is reached, `split_edge` falls back to a small-nudge mutation instead.
The cap is computed dynamically inside `ShapeGenome::mutate` as `(4 × genome.shapes.len()).max(6)`,
tying per-shape complexity to genome density. The static `max_shapes * 4` value stored in
`MutationConfig` is a fallback used only when genes are mutated outside that path (e.g. initial
genome construction).

Polygon vertices are sorted by angle from the centroid on every `normalize()` call, eliminating
self-intersecting edges. Clamping runs first so the centroid is computed from the final
in-bounds positions.

### Z-ordering

`ShapeGene` carries a `z_order: u16` field. `draw_genes` sorts genes ascending by `z_order` before
rasterising, so layering is independent of `Vec` insertion order. This makes recombination-via-
crossover safe: two parents with different gene orderings produce a child that renders consistently.
`ShapeGenome::mutate` has a ~5% chance to swap two genes' z-orders (reordering), and `ShapeGene::mutate`
has a ~10% chance to nudge its own z_order by ±1000.

### Grid genome (`--start grid`)

`GridGenome` in `libshapeme/src/grid.rs` is a shared-vertex quad mesh covering the entire canvas.
It is a separate genome type from `ShapeGenome` because:

- Adjacent cells share control points; moving one point reshapes multiple cells simultaneously.
- There is no cost budget or incremental ramp — the grid size is fixed at setup time.
- Recombination swaps full point or colour arrays (no z-ordering, no polygon splitting).

`GridGenome` stores `(cols+1) × (rows+1)` control points and `cols × rows` OKlab colours.
Each cell is rendered as two triangles (TL-TR-BR, TL-BR-BL) via `draw_triangle` (made `pub(crate)`)
to guarantee gap-free coverage even when points are deformed by mutation.

**Point constraints (enforced in `normalize`)**: corners are **pinned** to `±margin`, edge points
have one axis pinned and one clamped, interior points are clamped on both axes. "Pinned" means
assignment to the exact boundary — not clamping — so that when blur (and thus `margin`) grows,
edge points move outward and the canvas remains fully covered.

**Blur and margin coupling**: `blur_radius = min(cell_w, cell_h) / 2` at setup. If blur evolves,
`normalize` re-pins edge points to the updated margin so the canvas stays covered. This is the
same margin semantics used by `ShapeGene` coordinates.

**No cost budget**: `AnnealingState::new(usize::MAX / 2, usize::MAX / 2)` is used so the
incremental cost ramp never triggers. `GridGenome::total_cost()` returns the cell count (for
display logging), not a byte budget.

**Checkpoint format**: `CheckpointGenome` is now an enum (`Shape(ShapeGenome)` | `Grid(GridGenome)`),
stored in `Checkpoint.genome`. `StoredConfig.use_grid: bool` signals which variant `--restart`
should reconstruct. Old checkpoints decode with the "re-run `shapeme setup`" message as before.

**`Genome` trait extension**: `render_to_fb`, `build_svg_output`, `blur_radius`, `total_cost`
(default 0), and `trim_to_budget` (default identity) were added to the trait. `ShapeGenome`
provides concrete impls; `GridGenome` overrides all except `trim_to_budget`. The generic
`run_process_loop` and `run_batch` functions use these trait methods so both genome types share
the same annealing loop.

### Combined blur/sharp fitness function

When the genome has an active blur radius `r`, the fitness function computes a 50/50 blend:

```
blurred_candidate = apply_blur(render(genome), r)
blurred_target    = apply_blur(sharp_target, r)

diff_sharp   = compute_diff(blurred_candidate, sharp_target)
diff_blurred = compute_diff(blurred_candidate, blurred_target)

fitness = (diff_sharp + diff_blurred) / 2.0 × 100
```

When the genome has no blur, the formula reduces to `compute_diff(render(genome), sharp_target) × 100` unchanged.

**Why**: `diff_blurred` rewards low-frequency fidelity (blur removes detail both sides can't see),
while `diff_sharp` keeps a high-frequency penalty so the annealer cannot cheaply inflate blur to
collapse the signal. The previous `--blur-target-factor` flag had the same motivation but exposed
a "blur inflation" dynamic: blurring both sides equally always reduces the reported diff regardless
of genome quality, incentivising ever-larger blur radii. The 50/50 blend avoids that by keeping
`diff_sharp` in the sum, which increases when blur erases useful structure.

**Implementation** (`libshapeme/src/genome.rs`, `libshapeme/src/grid.rs`): `Genome::fitness` takes
an optional `blurred_target: Option<&[f32]>` parameter alongside the sharp `target`. When
`Some`, both diffs are computed and averaged; when `None`, only the sharp diff is computed.

**Callers** (`shapeme/src/main.rs`): `run_batch` and the recombination loop in `run_process_loop`
each maintain a `blurred_target_cache: Option<(f32, Vec<f32>)>` holding the target blurred at the
current candidate's blur radius. The cache is only recomputed when blur radius changes (≈5% of
mutations). The SDL display sites are unaffected — spacebar always toggles the sharp reference.
