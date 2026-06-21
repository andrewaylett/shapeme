# shapeme

Approximates a PNG image using a set of triangles and/or circles, evolved via
simulated annealing. Progress is shown in a live SDL2 window; results are saved
as SVG and binary checkpoints.

This is a Rust port of the original C implementation by
[Salvatore Sanfilippo](https://github.com/antirez/shapeme).

## Build

Requires Rust 1.85+ (edition 2024). SDL2 is compiled from source via the
`bundled` feature — you need cmake and a C compiler, but no system SDL2 library.

```sh
cargo build --release
```

## Usage

```
shapeme <input.png> <checkpoint.bin> <output.svg> [options]

Options:
  --use-triangles <0|1>     Use triangles (default: 1)
  --use-circles <0|1>       Use circles (default: 0)
  --max-shapes <n>          Maximum number of shapes (default: 64)
  --initial-shapes <n>      Starting number of shapes (default: 1)
  --mutation-rate <0-1000>  Mutation probability per shape (default: 200)
  --restart                 Ignore existing checkpoint and start fresh
```

Press `q` or `Escape` to stop and save.

The binary checkpoint format is not compatible with the original C version.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE). Original work copyright Salvatore
Sanfilippo; Rust port copyright Andrew Aylett.
