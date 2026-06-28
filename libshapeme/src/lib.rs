//! Core algorithm library for the shapeme image approximator.
//!
//! Provides shapes, rendering, annealing, and SVG-generation primitives.
//! Works on in-memory state only; all file I/O and UI live in the `shapeme` binary.
#![forbid(unsafe_code)]

/// Simulated annealing state.
pub mod annealing;
/// Gene-level traits and types (`ShapeGene`, `BlurGene`, `Gene` trait).
pub mod gene;
/// Genome-level traits and types (`ShapeGenome`, `Genome` trait).
pub mod genome;
/// Grid genome: shared-vertex quad-mesh covering the canvas.
pub mod grid;
/// OKlab perceptually uniform colour space conversions.
pub mod oklab;
/// Framebuffer rasterisation, blur, and diff computation.
pub mod render;
/// Shape types and mutation/generation functions.
pub mod shapes;
/// SVG generation and data URL encoding.
pub mod svg;
