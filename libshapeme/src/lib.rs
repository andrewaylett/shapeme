//! Core algorithm library for the shapeme image approximator.
//!
//! Provides shapes, rendering, annealing, and SVG-generation primitives.
//! Works on in-memory state only; all file I/O and UI live in the `shapeme` binary.
#![forbid(unsafe_code)]

/// Simulated annealing state and mutation logic.
pub mod annealing;
/// Framebuffer rasterisation, blur, and diff computation.
pub mod render;
/// Shape types and mutation/generation functions.
pub mod shapes;
/// SVG generation and data URL encoding.
pub mod svg;
