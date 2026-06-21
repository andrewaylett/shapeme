# shapeme — developer notes

Rust port of [antirez/shapeme](https://github.com/antirez/shapeme), a simulated
annealing image approximation tool. The original C source lives at
`/Users/axa/dev/native/shapeme/shapeme.c` and is the authoritative reference
for algorithm behaviour.

## Architecture

| Module         | Responsibility                                                                     |
| -------------- | ---------------------------------------------------------------------------------- |
| `main.rs`      | CLI (clap), SDL2 init and event loop, annealing main loop                          |
| `shapes.rs`    | `Shape` enum (Triangle/Circle), normalisation, mutation, random generation         |
| `render.rs`    | Framebuffer drawing (`draw_hline`, `draw_triangle`, `draw_circle`), `compute_diff` |
| `annealing.rs` | `AnnealingState`, `ShapeSet`, `mutate_shapes` (add/remove/swap/mutate)             |
| `io.rs`        | PNG load (`image` crate), SVG save, binary checkpoint (`bincode` v2 + serde)       |

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

### Binary checkpoint format

Uses `bincode` v2 API (`bincode::serde::encode_to_vec` /
`bincode::serde::decode_from_slice` with `bincode::config::standard()`).
This is intentionally **incompatible** with the raw C struct format.

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
