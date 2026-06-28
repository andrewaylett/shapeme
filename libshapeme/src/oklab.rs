//! Conversion between sRGB u8 and OKlab f32 colour spaces.
//!
//! OKlab (Björn Ottosson, 2020) is a perceptually uniform colour space built
//! on a cube-root-compressed LMS cone space.  Equal distances in OKlab
//! correspond to equal perceptual differences, making it suitable as a
//! diff metric and for arithmetic colour interpolation.
//!
//! Forward pipeline: sRGB u8 → linear f32 (piecewise gamma) → M1 (linear LMS)
//! → cube-root (L'M'S') → M2 → [L, a, b].
//!
//! Inverse: [L, a, b] → M2⁻¹ → cube → M1⁻¹ → linear RGB → clamp → gamma → u8.

/// Expand one sRGB channel value (0–255) to linear f32 [0, 1].
#[inline]
fn srgb_to_linear(c: u8) -> f32 {
    let f = c as f32 / 255.0;
    if f <= 0.04045 {
        f / 12.92
    } else {
        ((f + 0.055) / 1.055).powf(2.4)
    }
}

/// Compress a linear f32 [0, 1] channel to a gamma-encoded f32 [0, 1].
#[inline]
fn linear_to_srgb_f32(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert sRGB u8 (r, g, b) to `OKlab` [L, a, b].
///
/// Matrix constants from Björn Ottosson's 2020 reference implementation.
#[must_use]
#[allow(
    clippy::excessive_precision,
    reason = "these are exact reference matrix constants from Björn Ottosson's 2020 OKlab spec"
)]
pub fn srgb_u8_to_oklab(r: u8, g: u8, b: u8) -> [f32; 3] {
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);

    // M1: linear sRGB → LMS
    let l = 0.412_221_47 * rl + 0.536_332_54 * gl + 0.051_445_99 * bl;
    let m = 0.211_903_50 * rl + 0.680_699_55 * gl + 0.107_396_96 * bl;
    let s = 0.088_302_46 * rl + 0.281_718_84 * gl + 0.629_978_70 * bl;

    // Cube-root (L'M'S')
    let lp = l.cbrt();
    let mp = m.cbrt();
    let sp = s.cbrt();

    // M2: L'M'S' → OKlab
    [
        0.210_454_26 * lp + 0.793_617_78 * mp - 0.004_072_05 * sp,
        1.977_998_50 * lp - 2.428_592_21 * mp + 0.450_593_71 * sp,
        0.025_904_04 * lp + 0.782_771_77 * mp - 0.808_675_77 * sp,
    ]
}

/// Convert `OKlab` [L, a, b] to sRGB u8 (r, g, b).
///
/// Out-of-gamut linear values are clamped to [0, 1] before gamma encoding.
#[must_use]
#[allow(
    clippy::excessive_precision,
    reason = "these are exact reference matrix constants from Björn Ottosson's 2020 OKlab spec"
)]
pub fn oklab_to_srgb_u8(lab: [f32; 3]) -> [u8; 3] {
    let [l, a, b] = lab;

    // M2⁻¹: OKlab → L'M'S'
    let lp = l + 0.396_337_78 * a + 0.215_803_76 * b;
    let mp = l - 0.105_561_35 * a - 0.063_854_17 * b;
    let sp = l - 0.089_484_18 * a - 1.291_485_55 * b;

    // Cube (LMS)
    let lc = lp * lp * lp;
    let mc = mp * mp * mp;
    let sc = sp * sp * sp;

    // M1⁻¹: LMS → linear sRGB
    let rl = 4.076_741_66 * lc - 3.307_711_59 * mc + 0.230_969_93 * sc;
    let gl = -1.268_438_00 * lc + 2.609_757_40 * mc - 0.341_319_40 * sc;
    let bll = -0.004_196_09 * lc - 0.703_418_61 * mc + 1.707_614_70 * sc;

    // Clamp to [0, 1] then gamma compress and convert to u8
    let to_u8 = |c: f32| -> u8 {
        let clamped = c.clamp(0.0, 1.0);
        (linear_to_srgb_f32(clamped) * 255.0).round() as u8
    };

    [to_u8(rl), to_u8(gl), to_u8(bll)]
}

/// Convert a packed RGB24 byte buffer to an `OKlab` f32 buffer (3 floats per pixel).
#[must_use]
pub fn image_srgb_to_oklab(src: &[u8]) -> Vec<f32> {
    let n = src.len() / 3;
    let mut dst = Vec::with_capacity(n * 3);
    for i in 0..n {
        let base = i * 3;
        let lab = srgb_u8_to_oklab(src[base], src[base + 1], src[base + 2]);
        dst.extend_from_slice(&lab);
    }
    dst
}

/// Convert an `OKlab` f32 buffer (3 floats per pixel) to a packed RGB24 byte buffer.
#[must_use]
pub fn image_oklab_to_srgb(src: &[f32]) -> Vec<u8> {
    let n = src.len() / 3;
    let mut dst = Vec::with_capacity(n * 3);
    for i in 0..n {
        let base = i * 3;
        let rgb = oklab_to_srgb_u8([src[base], src[base + 1], src[base + 2]]);
        dst.extend_from_slice(&rgb);
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        let lab = srgb_u8_to_oklab(r, g, b);
        let [ro, go, bo] = oklab_to_srgb_u8(lab);
        (ro, go, bo)
    }

    fn within_1(a: u8, b: u8) -> bool {
        a.abs_diff(b) <= 1
    }

    #[test]
    fn black_round_trips() {
        let (r, g, b) = round_trip(0, 0, 0);
        assert!(
            within_1(r, 0) && within_1(g, 0) && within_1(b, 0),
            "black failed: ({r},{g},{b})"
        );
    }

    #[test]
    fn white_round_trips() {
        let (r, g, b) = round_trip(255, 255, 255);
        assert!(
            within_1(r, 255) && within_1(g, 255) && within_1(b, 255),
            "white failed: ({r},{g},{b})"
        );
    }

    #[test]
    fn primary_colours_round_trip() {
        for (r, g, b) in [(255, 0, 0), (0, 255, 0), (0, 0, 255)] {
            let (ro, go, bo) = round_trip(r, g, b);
            assert!(
                within_1(ro, r) && within_1(go, g) && within_1(bo, b),
                "primary ({r},{g},{b}) → ({ro},{go},{bo})"
            );
        }
    }

    #[test]
    fn black_oklab_is_zero() {
        let lab = srgb_u8_to_oklab(0, 0, 0);
        assert!(
            lab[0].abs() < 1e-4,
            "L for black should be ~0, got {}",
            lab[0]
        );
        assert!(
            lab[1].abs() < 1e-4,
            "a for black should be ~0, got {}",
            lab[1]
        );
        assert!(
            lab[2].abs() < 1e-4,
            "b for black should be ~0, got {}",
            lab[2]
        );
    }

    #[test]
    fn white_oklab_l_is_one() {
        let lab = srgb_u8_to_oklab(255, 255, 255);
        assert!(
            (lab[0] - 1.0).abs() < 1e-4,
            "L for white should be ~1, got {}",
            lab[0]
        );
        assert!(
            lab[1].abs() < 1e-4,
            "a for white should be ~0, got {}",
            lab[1]
        );
        assert!(
            lab[2].abs() < 1e-4,
            "b for white should be ~0, got {}",
            lab[2]
        );
    }

    #[test]
    fn sweep_round_trip_within_1_lsb() {
        // Sample a range of colours; each must round-trip within 1 u8 per channel.
        let step = 32u8;
        let mut worst = 0u8;
        for r in (0u8..=255).step_by(step as usize) {
            for g in (0u8..=255).step_by(step as usize) {
                for b in (0u8..=255).step_by(step as usize) {
                    let (ro, go, bo) = round_trip(r, g, b);
                    let diff = r.abs_diff(ro).max(g.abs_diff(go)).max(b.abs_diff(bo));
                    worst = worst.max(diff);
                    assert!(
                        diff <= 1,
                        "round-trip error > 1 for ({r},{g},{b}): got ({ro},{go},{bo})"
                    );
                }
            }
        }
        // Inform the reader of maximum observed error
        let _ = worst;
    }

    #[test]
    fn bulk_conversion_round_trips() {
        let src: Vec<u8> = (0u8..=255).step_by(16).flat_map(|v| [v, v, v]).collect();
        let lab = image_srgb_to_oklab(&src);
        let back = image_oklab_to_srgb(&lab);
        assert_eq!(src.len(), back.len());
        for (a, b) in src.iter().zip(back.iter()) {
            assert!(a.abs_diff(*b) <= 1, "bulk round-trip error: {a} → {b}");
        }
    }
}
