use rand::Rng;

pub(crate) const MINALPHA: u8 = 1;
pub(crate) const MAXALPHA: u8 = 100;

pub(crate) fn rand_between(rng: &mut impl Rng, min: i32, max: i32) -> i32 {
    rng.random_range(min..=max)
}

pub(crate) fn clamp_coord(v: i16, min: i16, max: i16) -> i16 {
    v.clamp(min, max)
}

/// Sample a colour uniformly in `OKlab` space: L ∈ [0,1], a/b ∈ [−0.4, 0.4].
///
/// Sampling directly in `OKlab` produces a perceptually uniform colour distribution,
/// unlike sampling in sRGB (which is biased by its non-linear gamut shape).
pub(crate) fn random_oklab_color(rng: &mut impl Rng) -> ([f32; 3], u8) {
    let l = rng.random::<f32>();
    let a = (rng.random::<f32>() * 2.0 - 1.0) * 0.4;
    let b_ch = (rng.random::<f32>() * 2.0 - 1.0) * 0.4;
    let alpha = rand_between(rng, MINALPHA as i32, MAXALPHA as i32) as u8;
    ([l, a, b_ch], alpha)
}

/// Per-channel `OKlab` nudge: L ±0.01 (tight), a/b ±0.05 (wide).
///
/// L gets a tighter scale because it has a narrower perceptually meaningful range
/// (0→1) and should drift less aggressively than the chromatic a/b channels.
pub(crate) fn nudge_oklab(rng: &mut impl Rng, lab: [f32; 3]) -> [f32; 3] {
    let dl = (rng.random::<f32>() * 2.0 - 1.0) * 0.01;
    let da = (rng.random::<f32>() * 2.0 - 1.0) * 0.05;
    let db = (rng.random::<f32>() * 2.0 - 1.0) * 0.05;
    [(lab[0] + dl).clamp(0.0, 1.0), lab[1] + da, lab[2] + db]
}

pub(crate) enum ShapeKind {
    Triangle,
    Circle,
    Polygon,
}

pub(crate) fn select_shape_type(
    rng: &mut impl Rng,
    use_triangles: bool,
    use_circles: bool,
    use_polygons: bool,
) -> ShapeKind {
    let mut kinds = Vec::with_capacity(3);
    if use_triangles {
        kinds.push(ShapeKind::Triangle);
    }
    if use_circles {
        kinds.push(ShapeKind::Circle);
    }
    if use_polygons {
        kinds.push(ShapeKind::Polygon);
    }
    debug_assert!(!kinds.is_empty(), "at least one shape type must be enabled");
    let idx = rng.random_range(0..kinds.len());
    kinds.remove(idx)
}
