//! Film emulation: **Halation** and **Film Look** (definitions in
//! `filmcraft_project::effect::film`).
//!
//! Halation isolates bright regions per channel, spreads each channel by its own distance (red the
//! furthest, as light reflected from the film base re-exposes the red layer), shapes the glow in
//! HSP space and adds it back, held off where the picture is already bright. It runs in linear
//! light on the premultiplied layer.
//!
//! Film Look grades in DaVinci Wide Gamut / DaVinci Intermediate (mid grey 0.336, the pivot of
//! every stage): pre-saturation → core look (tetrahedral colour cube, seven hue regions, a
//! luma-preserving split tone, lift/gamma/gain) → contrast (soft shoulder/toe, highlight
//! compression, fade) → saturation-vs-saturation and level-vs-saturation curves → subtractive
//! saturation and richness → bleach bypass → split tone → final saturation, with optional
//! halation, bloom, grain and vignette. Each stage is mixed by Color Blend / Effects Blend, so 0
//! leaves the picture unchanged.

use std::f32::consts::FRAC_1_SQRT_2;

use filmcraft_project::EffectInstance;
use rayon::prelude::*;

use super::{bv, chv, cv, fv};
use crate::effects::{FxCtx, hash3, noise_frame_seed};
use crate::image::Image;

/// DaVinci Intermediate code value of 18 % grey: the pivot of the Film Look stages.
const PIVOT: f32 = 0.336;
const LUMA709: [f32; 3] = [0.2126, 0.7152, 0.0722];
const HSP: [f32; 3] = [0.299, 0.587, 0.114];
/// Spread distances are defined on a UHD frame; other sizes scale by the diagonal.
const REF_DIAGONAL: f32 = 4405.815;
/// Working-image pixels per unit of Halation Spread on the reference frame.
const SPREAD_PX: f32 = 216.0;
/// Glow strength and highlight-reduction gains applied to the slider values.
const STRENGTH_GAIN: f32 = 5.0;
const REDUCE_GAIN: f32 = 4.0;

// ------------------------------------------------------------------ small helpers

#[inline]
fn sat(x: f32) -> f32 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}
#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}
#[inline]
fn map3(a: [f32; 3], f: impl Fn(f32) -> f32) -> [f32; 3] {
    [f(a[0]), f(a[1]), f(a[2])]
}
#[inline]
fn finite_or(v: f32, d: f32) -> f32 {
    if v.is_finite() { v } else { d }
}
/// A float parameter, clamped to its range (project files and control params are untrusted).
fn param(e: &EffectInstance, id: &str, cx: &FxCtx, lo: f32, hi: f32, def: f32) -> f32 {
    finite_or(fv(e, id, cx), def).clamp(lo, hi)
}
/// Working pixels for a distance defined on the UHD reference frame.
fn frame_scale(img: &Image) -> f32 {
    let d = ((img.w as f32).powi(2) + (img.h as f32).powi(2)).sqrt();
    if d.is_finite() && d > 0.0 { d / REF_DIAGONAL } else { 1.0 }
}

/// Gaussian-blur one plane with separate x/y sigmas (via a temporary image).
fn blur_xy(plane: &[f32], w: usize, h: usize, sx: f32, sy: f32) -> Vec<f32> {
    let sx = finite_or(sx, 0.0).max(0.0);
    let sy = finite_or(sy, 0.0).max(0.0);
    if sx <= 0.3 && sy <= 0.3 {
        return plane.to_vec();
    }
    let mut im = Image { w, h, px: plane.iter().flat_map(|&v| [v, 0.0, 0.0, 0.0]).collect() };
    crate::effects::gaussian(&mut im, sx, sy, true);
    im.px.as_chunks::<4>().0.iter().map(|p| p[0]).collect()
}

/// Planes of a per-pixel colour buffer.
fn planes(c: &[[f32; 3]]) -> [Vec<f32>; 3] {
    [c.iter().map(|p| p[0]).collect(), c.iter().map(|p| p[1]).collect(), c.iter().map(|p| p[2]).collect()]
}
fn blur3(c: &[[f32; 3]], w: usize, h: usize, sigma: [f32; 3], aspect: f32) -> Vec<[f32; 3]> {
    let [r, g, b] = planes(c);
    let r = blur_xy(&r, w, h, sigma[0] * aspect, sigma[0]);
    let g = blur_xy(&g, w, h, sigma[1] * aspect, sigma[1]);
    let b = blur_xy(&b, w, h, sigma[2] * aspect, sigma[2]);
    r.into_iter().zip(g).zip(b).map(|((r, g), b)| [r, g, b]).collect()
}

/// Cubic Bernstein curve with control values `p` (0 … 1 input).
#[inline]
fn bezier(x: f32, p: [f32; 4]) -> f32 {
    let u = 1.0 - x;
    u * u * u * p[0] + 3.0 * x * u * u * p[1] + 3.0 * x * x * u * p[2] + x * x * x * p[3]
}

/// Weights of the pure hue `h` (0 … 1) per channel, the HSV hexcone.
#[inline]
fn hue_rgb(h: f32) -> [f32; 3] {
    [sat((h * 6.0 - 3.0).abs() - 1.0), sat(2.0 - (h * 6.0 - 2.0).abs()), sat(2.0 - (h * 6.0 - 4.0).abs())]
}
/// A tint colour: hue `h`, saturation `s`, gain `g`.
#[inline]
fn tint(h: f32, s: f32, g: f32) -> [f32; 3] {
    map3(hue_rgb(h), |w| (1.0 + s * (w - 1.0)) * g)
}

/// HSV with hue 0 … 1, saturation (max − min) / max and value max.
fn rgb_to_hsv(c: [f32; 3]) -> (f32, f32, f32) {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    let d = mx - mn;
    let h = if mx == c[0] {
        (c[1] - c[2]) / (6.0 * d + 1e-10)
    } else if mx == c[1] {
        (c[2] - c[0]) / (6.0 * d + 1e-10) + 1.0 / 3.0
    } else {
        (c[0] - c[1]) / (6.0 * d + 1e-10) + 2.0 / 3.0
    };
    let h = (if h < 0.0 { h + 1.0 } else { h }).abs();
    (h, d / (mx + 1e-10), mx)
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    map3(hue_rgb(h), |w| v * (1.0 + s * (w - 1.0)))
}

// ------------------------------------------------------------------ colour spaces

/// Rec.709 → DaVinci Wide Gamut (both D65), from the published primaries.
const REC709_TO_DWG: [[f32; 3]; 3] =
    [[0.562_767_45, 0.323_516_58, 0.113_715_95], [0.077_754_63, 0.749_577_34, 0.172_668_02], [0.064_669_2, 0.191_998_69, 0.743_332_1]];
const DWG_TO_REC709: [[f32; 3]; 3] =
    [[1.898_614_9, -0.792_176_2, -0.106_438_72], [-0.168_948_78, 1.488_975_8, -0.320_027], [-0.121_539_17, -0.315_675_85, 1.437_215_1]];

#[inline]
fn mat(m: &[[f32; 3]; 3], c: [f32; 3]) -> [f32; 3] {
    [dot(m[0], c), dot(m[1], c), dot(m[2], c)]
}
/// DaVinci Intermediate encoding (linear → log).
#[inline]
pub(crate) fn di_encode(x: f32) -> f32 {
    if x <= 0.002_624_09 { x * 10.444_268 } else { ((x + 0.0075).log2() + 7.0) * 0.073_292_48 }
}
#[inline]
pub(crate) fn di_decode(y: f32) -> f32 {
    if y <= 0.027_406_68 { y / 10.444_268 } else { (y / 0.073_292_48 - 7.0).exp2() - 0.0075 }
}
fn to_working(lin709: [f32; 3]) -> [f32; 3] {
    map3(mat(&REC709_TO_DWG, lin709), di_encode)
}
fn from_working(di: [f32; 3]) -> [f32; 3] {
    mat(&DWG_TO_REC709, map3(di, di_decode))
}

// ------------------------------------------------------------------ Halation

/// Soft key of one channel between `lo` and `hi`: smoothstep for a narrow range, linear for a wide one.
pub(crate) fn isolate(x: f32, lo: f32, hi: f32) -> f32 {
    let lo = lo.clamp(0.0, 0.99);
    let hi = (lo + 0.01).max(hi).min(1.0);
    let range = hi - lo;
    let t = sat((x - lo) / range);
    let s = t * t * (3.0 - 2.0 * t);
    let r3 = range * range * range;
    t * r3 + s * (1.0 - r3)
}

/// HSP perceived brightness, HSV hue (0 … 6) and saturation of a (possibly negative) colour.
fn rgb_to_hsp(c: [f32; 3]) -> (f32, f32, f32) {
    let p = dot(HSP, [c[0] * c[0], c[1] * c[1], c[2] * c[2]]).sqrt();
    let p = if dot(HSP, c) < 0.0 { -p } else { p };
    let (mn, mx) = (c[0].min(c[1]).min(c[2]), c[0].max(c[1]).max(c[2]));
    let d = mx - mn;
    if d <= f32::EPSILON {
        return (p, 0.0, 0.0);
    }
    let m = mx.abs().max(f32::EPSILON).copysign(mx);
    let h = if (m - c[0]).abs() < 1e-6 {
        (c[1] - c[2]) / d
    } else if (m - c[1]).abs() < 1e-6 {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    };
    (p, if h < 0.0 { h + 6.0 } else { h }, sat(1.0 - mn / m))
}

/// Inverse of [`rgb_to_hsp`] (Finley's six-sector HSP → RGB).
fn hsp_to_rgb(p: f32, h: f32, s: f32) -> [f32; 3] {
    let neg = p < 0.0;
    let pp = p.abs();
    let h = if neg { ((h / 6.0 + 0.5).rem_euclid(1.0)) * 6.0 } else { h.rem_euclid(6.0) };
    let mom = 1.0 - s;
    // (largest, middle, smallest channel, middle-channel position)
    let (hi, mid, lo, hh) = match h {
        h if h < 1.0 => (0, 1, 2, h),
        h if h < 2.0 => (1, 0, 2, 2.0 - h),
        h if h < 3.0 => (1, 2, 0, h - 2.0),
        h if h < 4.0 => (2, 1, 0, 4.0 - h),
        h if h < 5.0 => (2, 0, 1, h - 4.0),
        h => (0, 2, 1, 6.0 - h),
    };
    let mut out = [0.0f32; 3];
    if mom > 0.0 {
        let inv = 1.0 / mom;
        let part = 1.0 + hh * (inv - 1.0);
        let l = pp / (HSP[hi] * inv * inv + HSP[mid] * part * part + HSP[lo]).sqrt();
        let top = l * inv;
        out[hi] = top;
        out[mid] = l + hh * (top - l);
        out[lo] = l;
    } else {
        let top = (pp * pp / (HSP[hi] + HSP[mid] * hh * hh)).sqrt();
        out[hi] = top;
        out[mid] = top * hh;
    }
    let out = map3(out, |v| finite_or(v, 0.0));
    if neg { map3(out, |v| -v) } else { out }
}

/// Gamma on HSP brightness and a saturation multiplier; hue is kept.
pub(crate) fn glow_gamma_sat(c: [f32; 3], gamma: f32, saturation: f32) -> [f32; 3] {
    let (p, h, s) = rgb_to_hsp(c);
    let p = p.abs().powf(gamma).copysign(p);
    hsp_to_rgb(p, h, sat(s * saturation))
}

pub(super) fn halation(img: &mut Image, e: &EffectInstance, cx: &FxCtx) {
    let (w, h) = (img.w, img.h);
    let threshold = param(e, "threshold", cx, 0.0, 1.0, 0.2);
    let normalization = param(e, "normalization", cx, 0.0, 1.0, 1.0);
    let film_sat = param(e, "film_sat_level", cx, 0.1, 100.0, 1.0);
    let strength = param(e, "strength", cx, 0.0, 10.0, 0.5);
    let gamma = param(e, "gamma", cx, 0.1, 10.0, 1.35);
    let saturation = param(e, "saturation", cx, 0.0, 5.0, 1.0);
    let spread = param(e, "spread", cx, 0.0, 10.0, 0.333);
    let rel = if bv(e, "fine_tune") {
        [param(e, "spread_r", cx, 0.0, 2.0, 1.6), param(e, "spread_g", cx, 0.0, 2.0, 1.0), param(e, "spread_b", cx, 0.0, 2.0, 0.75)]
    } else {
        [1.6, 1.0, 0.75]
    };
    let sec_strength = param(e, "sec_strength", cx, 0.0, 5.0, 0.0);
    let sec_gamma = param(e, "sec_gamma", cx, 0.1, 10.0, 1.35);
    let sec_spread = param(e, "sec_spread", cx, 0.0, 5.0, 0.6);
    let filter = cv(e, "sec_filter", cx);
    let reduce = param(e, "reduce_highlights", cx, 0.0, 2.0, 0.5);
    let aspect = param(e, "aspect_ratio", cx, 0.1, 10.0, 1.0);

    // 1. bright regions, per channel (premultiplied linear: transparent areas give nothing)
    let src: Vec<[f32; 3]> = img.px.as_chunks::<4>().0.iter().map(|p| [p[0], p[1], p[2]]).collect();
    let iso: Vec<[f32; 3]> = src.par_iter().map(|&c| map3(c, |v| isolate(v / film_sat, threshold, normalization))).collect();
    if bv(e, "view_isolated") {
        write_rgb(img, &iso, false);
        return;
    }
    // 2. primary glow: each dye layer spreads by its own distance, then gamma/saturation
    let unit = SPREAD_PX * frame_scale(img);
    let sigma = map3(rel, |r| spread * r * unit);
    let primary: Vec<[f32; 3]> = blur3(&iso, w, h, sigma, aspect).into_par_iter().map(|c| glow_gamma_sat(c, gamma, saturation)).collect();
    // 3. secondary glow: one wider spread, its own gamma, coloured by the filter
    let secondary = (sec_strength > 0.0).then(|| {
        let s = sec_spread * unit;
        blur3(&iso, w, h, [s; 3], aspect)
            .into_par_iter()
            .map(|c| {
                let g = glow_gamma_sat(c, sec_gamma, 1.0);
                [g[0] * filter[0], g[1] * filter[1], g[2] * filter[2]]
            })
            .collect::<Vec<_>>()
    });
    // 4. the halo, optionally with grain
    let grain = bv(e, "grain_on").then(|| {
        let gs = param(e, "grain_strength", cx, 0.0, 5.0, 0.25);
        let size = param(e, "grain_size", cx, 0.0, 1.0, 0.5);
        let seed = noise_frame_seed(cx);
        let n: Vec<f32> = (0..w * h).map(|i| (hash3(i % w.max(1), i / w.max(1), seed) - 0.5) * 3.46).collect();
        let n = blur_xy(&n, w, h, (0.5 + 2.0 * size) * cx.px_scale.max(0.25), (0.5 + 2.0 * size) * cx.px_scale.max(0.25));
        (gs, n)
    });
    let (sk, sec_k) = (strength * STRENGTH_GAIN, sec_strength * STRENGTH_GAIN);
    let alone = bv(e, "view_glow_alone");
    let reduce_k = reduce * REDUCE_GAIN;
    let out: Vec<[f32; 3]> = (0..src.len())
        .into_par_iter()
        .map(|i| {
            let (Some(&c), Some(&p)) = (src.get(i), primary.get(i)) else { return [0.0; 3] };
            let sc = secondary.as_ref().and_then(|s| s.get(i)).copied().unwrap_or([0.0; 3]);
            let mut halo = map3([p[0] * sk + sc[0] * sec_k, p[1] * sk + sc[1] * sec_k, p[2] * sk + sc[2] * sec_k], |v| finite_or(v, 0.0).max(0.0));
            if let Some((gs, n)) = &grain {
                let g = 1.0 + gs * n.get(i).copied().unwrap_or(0.0);
                halo = map3(halo, |v| (v * g).max(0.0));
            }
            // 5. hold the halo off where the picture is already bright
            let k = 1.0 - sat(dot(LUMA709, c) * reduce_k).sqrt();
            let halo = map3(halo, |v| v * k);
            if alone { halo } else { [c[0] + halo[0], c[1] + halo[1], c[2] + halo[2]] }
        })
        .collect();
    write_rgb(img, &out, true);
}

/// Replace the premultiplied colour; `glow_on_alpha` keeps the glow inside the layer's alpha.
fn write_rgb(img: &mut Image, c: &[[f32; 3]], glow_on_alpha: bool) {
    img.px.par_chunks_mut(4).zip(c.par_iter()).for_each(|(p, c)| {
        let a = if glow_on_alpha { p[3].clamp(0.0, 1.0) } else { 1.0 };
        let src = [p[0], p[1], p[2]];
        for (dst, (&v, &s)) in p.iter_mut().zip(c.iter().zip(&src)) {
            // the source part is already premultiplied; only the added glow is scaled by alpha
            *dst = finite_or(s + (v - s) * a, 0.0).max(0.0);
        }
    });
}

// ------------------------------------------------------------------ Film Look: core looks

struct CoreLook {
    pre_sat: f32,
    /// Corners of the colour cube: red, green, blue, cyan, magenta, yellow.
    cube: [[f32; 3]; 6],
    /// Hue regions: centre °, width °, rotation °, saturation ×, brightness (stops).
    regions: [[f32; 5]; 7],
    split_top: [f32; 3],
    split_bottom: [f32; 3],
    lift: [f32; 3],
    gamma: [f32; 3],
    gain: [f32; 3],
    /// Skin region (region 1) rotation, saturation, brightness at Skin Bias −1, 0, +1.
    skin: [[f32; 3]; 3],
    sat_vs_sat: ([f32; 4], f32),
    level_vs_sat: ([f32; 4], f32),
}

const CYAN: [f32; 3] = [0.0, 1.0, 1.0];
const MAGENTA: [f32; 3] = [1.0, 0.0, 1.0];
const NEUTRAL: [f32; 3] = [1.0, 1.0, 1.0];

static LOOKS: [CoreLook; 5] = [
    // Cinematic
    CoreLook {
        pre_sat: 0.7,
        cube: [[1.0, 0.03, 0.0], [0.0, 1.3, 0.3], [-0.15, 0.0, 1.0], CYAN, [1.2, 0.0, 1.0], [1.0, 0.8, 0.2]],
        regions: [
            [0.0, 60.0, 2.0, 1.1, -0.25],
            [20.0, 20.0, 5.0, 1.0, -0.75],
            [60.0, 60.0, 15.0, 2.0, -1.0],
            [120.0, 60.0, -30.0, 1.0, -2.0],
            [200.0, 60.0, -2.0, 1.0, 0.0],
            [240.0, 60.0, 5.0, 1.2, -0.42],
            [300.0, 60.0, 0.0, 0.9, -0.59],
        ],
        split_top: [0.225, 0.4, 1.0],
        split_bottom: [0.515, 0.6, 0.37],
        lift: [0.0; 3],
        gamma: NEUTRAL,
        gain: NEUTRAL,
        skin: [[10.0, 5.0, 0.0], [1.15, 1.0, 0.85], [-4.0, -0.75, 0.75]],
        sat_vs_sat: ([1.2, 1.2, 0.5, 0.0], 0.9),
        level_vs_sat: ([1.6, 1.2, 0.9, 0.4], 0.85),
    },
    // Rochester
    CoreLook {
        pre_sat: 0.85,
        cube: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.25], [0.0, 0.25, 1.2], CYAN, MAGENTA, [1.2, 1.0, -0.1]],
        regions: [
            [0.0, 60.0, 8.0, 1.3, -0.4],
            [20.0, 60.0, -1.0, 0.88, 0.75],
            [60.0, 60.0, -0.5, 1.6, 0.0],
            [120.0, 60.0, 0.0, 1.12, 0.0],
            [200.0, 60.0, 0.0, 1.0, 0.0],
            [240.0, 60.0, -15.0, 1.23, 0.0],
            [300.0, 60.0, 0.0, 0.9, -0.2],
        ],
        split_top: [0.225, 0.4, 1.0],
        split_bottom: [0.515, 1.0, 0.37],
        lift: [0.005, 0.005, 0.0],
        gamma: [0.98, 0.975, 1.01],
        gain: [1.0, 1.0, 0.99],
        skin: [[4.0, -1.0, -6.0], [1.03, 0.88, 0.73], [-1.25, 0.75, 0.75]],
        sat_vs_sat: ([1.2, 1.2, 0.5, 0.3], 0.8),
        level_vs_sat: ([1.6, 1.2, 0.5, 0.3], 0.85),
    },
    // Akasaka
    CoreLook {
        pre_sat: 0.85,
        cube: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.25], [0.0, 0.25, 1.2], CYAN, MAGENTA, [1.2, 1.0, -0.1]],
        regions: [
            [0.0, 60.0, 8.0, 1.3, -0.4],
            [20.0, 60.0, -2.0, 0.75, 0.75],
            [60.0, 60.0, 1.5, 1.5, -0.42],
            [120.0, 60.0, 0.0, 1.12, 0.0],
            [200.0, 60.0, 0.0, 1.0, 0.0],
            [240.0, 60.0, -0.2, 1.23, -0.586],
            [300.0, 60.0, 0.0, 0.9, -0.2],
        ],
        split_top: [0.225, 0.4, 1.0],
        split_bottom: [0.515, 1.0, 0.37],
        lift: [0.0; 3],
        gamma: [0.986, 0.991, 1.0],
        gain: [1.0, 0.997, 1.008],
        skin: [[3.0, -2.0, -7.0], [0.9, 0.75, 0.6], [-1.25, 0.75, 0.75]],
        sat_vs_sat: ([1.2, 1.2, 0.8, 0.3], 0.8),
        level_vs_sat: ([1.6, 1.2, 0.7, 0.3], 0.85),
    },
    // Elated
    CoreLook {
        pre_sat: 0.8,
        cube: [[1.14, 0.09, 0.07], [-0.1, 1.1, 0.1], [0.0, 0.15, 1.0], CYAN, MAGENTA, [1.18, 0.88, -0.45]],
        regions: [
            [0.0, 60.0, 12.5, 1.2, -0.4],
            [20.0, 40.0, -12.5, 0.75, 0.4],
            [60.0, 60.0, 10.0, 1.1, -0.5],
            [120.0, 60.0, 30.0, 1.0, 0.0],
            [200.0, 60.0, -10.0, 0.75, 0.0],
            [240.0, 60.0, 0.0, 1.25, 0.0],
            [300.0, 60.0, 0.0, 0.8, 0.0],
        ],
        split_top: [0.225, 0.4, 1.0],
        split_bottom: [0.515, 1.0, 0.37],
        lift: [0.005; 3],
        gamma: NEUTRAL,
        gain: NEUTRAL,
        skin: [[-7.5, -12.5, -17.5], [0.9, 0.75, 0.6], [-1.6, 0.4, 0.4]],
        sat_vs_sat: ([1.2, 1.0, 0.4, 0.0], 0.9),
        level_vs_sat: ([1.6, 1.5, 0.4, 0.0], 0.85),
    },
    // Vintage
    CoreLook {
        pre_sat: 0.85,
        cube: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.15], [0.0, 0.1, 1.2], CYAN, MAGENTA, [1.25, 1.0, 0.33]],
        regions: [
            [0.0, 60.0, 8.0, 1.0, -0.42],
            [20.0, 60.0, 0.0, 1.3, 0.75],
            [60.0, 60.0, -3.0, 1.4, -1.25],
            [120.0, 60.0, 0.0, 1.12, 0.0],
            [200.0, 60.0, 0.0, 1.0, 0.0],
            [240.0, 60.0, -14.9, 1.23, 0.0],
            [300.0, 60.0, 0.0, 0.9, -0.2],
        ],
        split_top: [0.225, 0.4, 1.0],
        split_bottom: [0.515, 1.0, 0.37],
        lift: [0.005, 0.019, 0.02],
        gamma: [1.05, 0.99, 1.05],
        gain: [0.88, 0.95, 0.88],
        skin: [[5.0, 0.0, -5.0], [1.45, 1.3, 1.15], [-1.25, 0.75, 0.75]],
        sat_vs_sat: ([1.2, 1.2, 0.7, 0.3], 0.8),
        level_vs_sat: ([1.6, 1.2, 0.45, 0.0], 0.85),
    },
];
const SPLIT_BLEND: f32 = 0.5;

// ------------------------------------------------------------------ Film Look: stages

/// Saturation about the channel mean.
fn saturate(c: [f32; 3], s: f32) -> [f32; 3] {
    let m = (c[0] + c[1] + c[2]) / 3.0;
    map3(c, |v| m + s * (v - m))
}

/// Tetrahedral interpolation through the six cube corners (black and white fixed).
pub(crate) fn colour_cube(c: [f32; 3], k: &[[f32; 3]; 6]) -> [f32; 3] {
    let [r, g, b] = map3(c, f32::abs);
    let [cr, cg, cb, cc, cm, cy] = *k;
    let lin = |a: f32, p: [f32; 3], bb: f32, q: [f32; 3], cc_: f32, s: [f32; 3]| -> [f32; 3] {
        [a * p[0] + bb * q[0] + cc_ * s[0], a * p[1] + bb * q[1] + cc_ * s[1], a * p[2] + bb * q[2] + cc_ * s[2]]
    };
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let w = NEUTRAL;
    if r > g {
        if g > b {
            lin(r, cr, g, sub(cy, cr), b, sub(w, cy))
        } else if r > b {
            lin(r, cr, b, sub(cm, cr), g, sub(w, cm))
        } else {
            lin(b, cb, r, sub(cm, cb), g, sub(w, cm))
        }
    } else if b > g {
        lin(b, cb, g, sub(cc, cb), r, sub(w, cc))
    } else if b > r {
        lin(g, cg, b, sub(cc, cg), r, sub(w, cc))
    } else {
        lin(g, cg, r, sub(cy, cg), b, sub(w, cy))
    }
}

const TO_OPP: [[f32; 3]; 3] = [[0.816_497, -0.408_248, -0.408_248], [0.0, FRAC_1_SQRT_2, -FRAC_1_SQRT_2], [0.577_35, 0.577_35, 0.577_35]];

/// Seven hue regions in a spherical colour space (hue angle, angle from neutral, length).
pub(crate) fn hue_regions(c: [f32; 3], regions: &[[f32; 5]; 7]) -> [f32; 3] {
    let sign = map3(c, |v| if v < 0.0 { -1.0 } else { 1.0 });
    let v = mat(&TO_OPP, map3(c, sat));
    let mut hue = v[1].atan2(v[0]).to_degrees().rem_euclid(360.0);
    let s0 = (v[0].hypot(v[1])).atan2(v[2]) * 1.046_77;
    let l0 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() * 0.577_35;
    let w_sat = sat((s0 - 0.05) / 0.45);
    let w_skin = sat(s0) * sat((l0 - 0.1) / 0.5);
    let (mut s, mut l) = (sat(s0), sat(l0));
    for (i, rg) in regions.iter().enumerate() {
        let [centre, width, rot, smul, bright] = *rg;
        let d = (hue - centre + 180.0).rem_euclid(360.0) - 180.0;
        let t = sat(d.abs() / width.max(1e-6));
        let wgt = 1.0 - t * t * (3.0 - 2.0 * t);
        let wl = if i == 1 { w_skin } else { w_sat };
        let l2 = l + (l * bright.exp2() - l) * wl;
        l = sat(l + (l2 - l) * wgt);
        hue = (hue + rot * wgt).rem_euclid(360.0);
        s = sat(s + (s * smul - s) * wgt);
    }
    let a = s * 0.955_317;
    let hr = hue.to_radians();
    let len = l * 1.732_050_8;
    let o = [hr.cos() * a.sin() * len, hr.sin() * a.sin() * len, a.cos() * len];
    // TO_OPP is orthonormal: its transpose is the inverse
    let rgb = [
        TO_OPP[0][0] * o[0] + TO_OPP[1][0] * o[1] + TO_OPP[2][0] * o[2],
        TO_OPP[0][1] * o[0] + TO_OPP[1][1] * o[1] + TO_OPP[2][1] * o[2],
        TO_OPP[0][2] * o[0] + TO_OPP[1][2] * o[1] + TO_OPP[2][2] * o[2],
    ];
    [rgb[0] * sign[0], rgb[1] * sign[1], rgb[2] * sign[2]]
}

/// The core's split tone: per-channel curves below/above the mid point, then only the chroma is
/// kept (BT.709 YCbCr with the original luma).
fn core_split(c: [f32; 3], top: [f32; 3], bottom: [f32; 3]) -> [f32; 3] {
    let (t, bt) = (tint(top[0], top[1], top[2]), tint(bottom[0], bottom[1], bottom[2]));
    let mid = PIVOT;
    let mut o = [0.0f32; 3];
    for ((dst, &x), (&tt, &bb)) in o.iter_mut().zip(&c).zip(t.iter().zip(&bt)) {
        *dst = if x <= mid {
            let u = (x / mid).max(0.0);
            mid * bezier(u, [0.0, bb, bb, 1.0])
        } else {
            let u = (x - mid) / (1.0 - mid);
            mid + (1.0 - mid) * bezier(u, [0.0, tt, tt, 1.0])
        };
    }
    let y = dot(LUMA709, map3(c, sat));
    let o = map3(o, sat);
    let cb = dot([-0.1146, -0.3854, 0.5], o);
    let cr = dot([0.5, -0.4542, -0.0458], o);
    let rec = map3([y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb], sat);
    mix(c, rec, SPLIT_BLEND * 0.25)
}

fn lift_gamma_gain(c: [f32; 3], lift: [f32; 3], gamma: [f32; 3], gain: [f32; 3]) -> [f32; 3] {
    let mut o = c;
    for (((v, l), g), k) in o.iter_mut().zip(lift).zip(gamma).zip(gain) {
        let x = *v + l * (1.0 - *v);
        let inv = 1.0 / g.max(1e-3);
        let x = if (inv - 1.0).abs() < 1e-6 { x } else { x.abs().powf(inv).copysign(x) };
        *v = x * k;
    }
    o
}

/// Soft shoulder above `t`: input `w` lands on `x`, sharpness `k`.
pub(crate) fn knee(v: f32, w: f32, x: f32, t: f32, k: f32) -> f32 {
    if v.is_nan() || v <= t {
        return v;
    }
    let a = w.max(x + 0.03) - t;
    let b = x.max(t + 0.03) - t;
    let n = k + 0.09;
    let s = a / ((a / b).powf(n) - 1.0).powf(1.0 / n);
    if !(s.is_finite() && s > 0.0) {
        return v;
    }
    let d = v - t;
    t + d / (1.0 + (d / s).powf(n)).powf(1.0 / n)
}

/// Contrast about the pivot with a soft shoulder and toe, then highlight compression and fade.
pub(crate) fn contrast_fade(c: f32, contrast: f32, highlights: f32, fade: f32) -> f32 {
    let o = if contrast < 1.0 {
        PIVOT + (c - PIVOT) * contrast
    } else {
        let x = knee(PIVOT + 2.0 * (c - PIVOT), 5.0, 1.035, PIVOT, 1.5);
        let x = 1.0 - knee(1.0 - x, 5.0, 1.085, 0.666, 1.5);
        c + (sat(x) - c) * sat(contrast - 1.0)
    };
    let o = o + (knee(o, 5.0, 0.64, 0.5, 1.5) - o) * (1.0 - highlights);
    let z = 1.0 - o;
    1.0 - (z + (knee(z, 5.0, 0.98, 0.666, 1.5) - z) * fade)
}

/// Saturation scaled by a curve of saturation (`by_level = false`) or of the brightest channel.
fn curve_saturation(c: [f32; 3], by_level: bool, curve: [f32; 4], amount: f32) -> [f32; 3] {
    let (_, s, v) = rgb_to_hsv(map3(c, sat));
    let a = bezier(if by_level { v } else { s }, curve) * amount;
    map3(saturate(c, a), sat)
}

/// Subtractive saturation (pull the other channels toward the brightest) and richness (more
/// saturation in the mid range).
fn subtractive_richness(c: [f32; 3], subtractive: f32, richness: f32) -> [f32; 3] {
    let mx = c[0].max(c[1]).max(c[2]);
    let c2 = map3(c, |v| sat(mx + (v - mx) * subtractive));
    let (h, s, v) = rgb_to_hsv(c2);
    let s2 = sat(s + (s * richness - s) * bezier(s, [0.0, 0.0, 3.0, 0.0]));
    hsv_to_rgb(h, s2, v)
}

fn bleach_bypass(c: [f32; 3], amount: f32) -> [f32; 3] {
    let y = dot(LUMA709, c);
    let y2 = PIVOT + (y - PIVOT) * 1.4;
    mix(c, [y2; 3], amount)
}

/// Tint pushed about the pivot; hue and saturation from the result, value from the original.
fn split_tone(c: [f32; 3], amount: f32, pivot: f32, hue: f32) -> [f32; 3] {
    let t = tint(hue, 0.7, 0.7);
    let c2 = [pivot + 2.0 * (c[0] - pivot) * t[0], pivot + 2.0 * (c[1] - pivot) * t[1], pivot + 2.0 * (c[2] - pivot) * t[2]];
    let (_, _, v0) = rgb_to_hsv(map3(c, sat));
    let (h, s, _) = rgb_to_hsv(map3(c2, sat));
    mix(c, map3(hsv_to_rgb(h, s, v0), sat), amount)
}

// Film Look's own halation (threshold mask → blur → dye-weighted mix)
fn threshold_mask(c: [f32; 3]) -> [f32; 3] {
    let y = sat(dot(LUMA709, map3(c, sat)));
    map3(c, |v| v * y)
}
pub(crate) fn halation_mix(src: [f32; 3], spill: [f32; 3], amount: f32, hue: f32, saturation: f32) -> [f32; 3] {
    let w = [1.0, 1.0 + (hue - 1.0) * saturation, 1.0 - saturation];
    let mut o = src;
    for ((v, &sp), &wk) in o.iter_mut().zip(&spill).zip(&w) {
        let m = (*v + wk * sp) / (1.0 + wk).max(1e-3);
        *v += (m - *v) * sat(amount);
    }
    o
}

struct LookParams {
    look: &'static CoreLook,
    skin: [f32; 3],
    color_blend: f32,
    look_blend: f32,
    contrast: f32,
    highlights: f32,
    fade: f32,
    subtractive: f32,
    richness: f32,
    bleach: f32,
    split: Option<(f32, f32, f32)>,
}

/// The colour stages of Film Look on one DaVinci Intermediate colour.
fn grade(c: [f32; 3], p: &LookParams) -> [f32; 3] {
    let k = p.look;
    let cb = p.color_blend;
    let x = mix(c, saturate(c, k.pre_sat), cb);
    // core look
    let mut regions = k.regions;
    if let Some(r) = regions.get_mut(1) {
        r[2] = p.skin[0];
        r[3] = p.skin[1];
        r[4] = p.skin[2];
    }
    let core = map3(colour_cube(x, &k.cube), sat);
    let core = hue_regions(core, &regions);
    let core = core_split(core, k.split_top, k.split_bottom);
    let core = lift_gamma_gain(core, k.lift, k.gamma, k.gain);
    let x = mix(x, core, p.look_blend * cb);
    // contrast, curves, saturation
    let x = mix(x, map3(x, |v| contrast_fade(v, p.contrast, p.highlights, p.fade)), cb);
    let x = mix(x, curve_saturation(x, false, k.sat_vs_sat.0, k.sat_vs_sat.1), cb);
    let x = mix(x, curve_saturation(x, true, k.level_vs_sat.0, k.level_vs_sat.1), cb);
    let x = mix(x, subtractive_richness(x, p.subtractive, p.richness), cb);
    let x = bleach_bypass(x, p.bleach * 0.6 * cb);
    let x = match p.split {
        Some((amount, pivot, hue)) => split_tone(x, amount * 0.6 * cb, pivot, hue),
        None => x,
    };
    mix(x, saturate(x, 0.85), cb)
}

/// Skin region values for a bias in −1 … 1 (piecewise linear through −1, 0, +1).
fn skin_at(skin: &[[f32; 3]; 3], bias: f32) -> [f32; 3] {
    let b = bias.clamp(-1.0, 1.0);
    let at = |v: [f32; 3]| if b < 0.0 { v[1] + (v[0] - v[1]) * -b } else { v[1] + (v[2] - v[1]) * b };
    [at(skin[0]), at(skin[1]), at(skin[2])]
}

pub(super) fn film_look(img: &mut Image, e: &EffectInstance, cx: &FxCtx) {
    let (w, h) = (img.w, img.h);
    let look_idx = chv(e, "core_look") as usize;
    let Some(look) = LOOKS.get(look_idx).or(LOOKS.first()) else { return };
    let color_blend = param(e, "color_blend", cx, 0.0, 1.0, 1.0);
    let effects_blend = param(e, "effects_blend", cx, 0.0, 1.0, 1.0);
    let lp = LookParams {
        look,
        skin: skin_at(&look.skin, param(e, "skin_bias", cx, -1.0, 1.0, 0.0)),
        color_blend,
        look_blend: param(e, "film_look_blend", cx, 0.0, 1.0, 1.0),
        contrast: param(e, "contrast", cx, 0.1, 2.0, 1.25),
        highlights: param(e, "highlights", cx, 0.0, 1.0, 0.35),
        fade: param(e, "fade", cx, 0.0, 1.0, 0.285),
        subtractive: {
            let s = param(e, "subtractive_sat", cx, 0.0, 3.0, 1.2);
            if s > 1.0 { 1.0 + (s - 1.0) * 0.5 } else { s }
        },
        richness: param(e, "richness", cx, 0.0, 3.0, 1.0),
        bleach: param(e, "bleach_bypass", cx, 0.0, 1.0, 0.0),
        split: bv(e, "split_on").then(|| {
            (param(e, "split_amount", cx, 0.0, 1.0, 0.0), param(e, "split_pivot", cx, 0.0, 1.0, 0.3), param(e, "split_angle", cx, 0.0, 360.0, 20.0) / 360.0)
        }),
    };
    let gain = param(e, "exposure", cx, -5.0, 5.0, 0.0).exp2();
    let scale = frame_scale(img);

    // straight linear colour (transparent pixels stay untouched)
    let alpha: Vec<f32> = img.px.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
    let mut lin: Vec<[f32; 3]> = img.px.as_chunks::<4>().0.iter().map(|p| map3(Image::unpremul(*p), |v| v * gain)).collect();

    if bv(e, "bloom_on") {
        let amount = param(e, "bloom_amount", cx, 0.0, 1.0, 0.25) * effects_blend;
        let sigma = param(e, "bloom_radius", cx, 0.0, 100.0, 10.0) / 25.0 * 50.0 * scale;
        let b = blur3(&lin, w, h, [sigma; 3], 1.0);
        lin.par_iter_mut().zip(b.par_iter()).for_each(|(c, b)| *c = mix(*c, *b, amount));
    }
    let mut di: Vec<[f32; 3]> = lin.par_iter().map(|&c| to_working(c)).collect();

    if bv(e, "halation_on") {
        let amount = param(e, "halation_amount", cx, 0.0, 1.0, 0.25) * effects_blend;
        let sigma = param(e, "halation_radius", cx, 0.0, 10.0, 4.0) * 20.0 * scale;
        let hue = param(e, "halation_hue", cx, 0.0, 1.0, 0.5);
        let hs = param(e, "halation_sat", cx, 0.0, 1.1, 1.0);
        let mask: Vec<[f32; 3]> = di.par_iter().map(|&c| threshold_mask(c)).collect();
        let spill = blur3(&mask, w, h, [sigma; 3], 1.0);
        let only_edges = bv(e, "halation_highlights_only");
        di.par_iter_mut().zip(spill.par_iter().zip(mask.par_iter())).for_each(|(c, (s, m))| {
            let o = halation_mix(*c, [c[0] + s[0], c[1] + s[1], c[2] + s[2]], amount, hue, hs);
            *c = if only_edges { mix(o, *c, 1.0 - sat(1.0 - sat(dot(LUMA709, *m)))) } else { o };
        });
    }
    if bv(e, "grain_on") {
        let amount = param(e, "grain_amount", cx, 0.0, 1.0, 0.125) * effects_blend * 0.1;
        let gsat = param(e, "grain_saturation", cx, 0.0, 1.0, 0.3);
        let seed = noise_frame_seed(cx);
        di.par_iter_mut().enumerate().for_each(|(i, c)| {
            let (x, y) = (i % w.max(1), i / w.max(1));
            let n = [0u64, 1, 2].map(|k| (hash3(x, y, seed.wrapping_add(k.wrapping_mul(0x9E37_79B9))) - 0.5) * 3.46);
            let mono = (n[0] + n[1] + n[2]) / 3.0;
            let l = sat(dot(LUMA709, *c));
            let wgt = amount * (1.0 - (2.0 * l - 1.0).abs());
            for (v, nk) in c.iter_mut().zip(n) {
                *v += (mono + (nk - mono) * gsat) * wgt;
            }
        });
    }
    if color_blend > 0.0 {
        di.par_iter_mut().for_each(|c| *c = grade(*c, &lp));
    }
    let mut out: Vec<[f32; 3]> = di.par_iter().map(|&c| map3(from_working(c), |v| finite_or(v, 0.0).max(0.0))).collect();

    if bv(e, "vignette_on") {
        let amount = param(e, "vignette_amount", cx, 0.0, 1.0, 0.25) * effects_blend;
        let size = param(e, "vignette_size", cx, 0.0, 1.0, 0.25).max(1e-3);
        let m = w.max(h).max(1) as f32;
        out.par_iter_mut().enumerate().for_each(|(i, c)| {
            let (x, y) = ((i % w.max(1)) as f32 + 0.5, (i / w.max(1)) as f32 + 0.5);
            let r = (((x - w as f32 / 2.0) * 2.0 / m).powi(2) + ((y - h as f32 / 2.0) * 2.0 / m).powi(2)).sqrt();
            let k = 1.0 - amount * sat((r - (1.0 - size)) / size).powi(2);
            *c = map3(*c, |v| v * k);
        });
    }
    // back to premultiplied (fully transparent pixels keep their original values)
    img.px.par_chunks_mut(4).zip(out.par_iter().zip(alpha.par_iter())).for_each(|(p, (c, &a))| {
        if a > 1e-6 {
            p[0] = c[0] * a;
            p[1] = c[1] * a;
            p[2] = c[2] * a;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close3(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn intermediate_grey_is_the_pivot_and_round_trips() {
        assert!((di_encode(0.18) - PIVOT).abs() < 1e-3);
        for x in [0.0, 0.001, 0.01, 0.18, 1.0, 10.0] {
            assert!((di_decode(di_encode(x)) - x).abs() < 1e-5 * x.max(1.0), "{x}");
        }
        let c = [0.3, 0.6, 0.1];
        assert!(close3(from_working(to_working(c)), c, 1e-4));
    }

    #[test]
    fn hsp_round_trips_and_neutral_glow_is_identity() {
        for c in [[0.2, 0.5, 0.9], [1.0, 0.0, 0.0], [0.3, 0.3, 0.3], [0.9, 0.7, 0.1], [0.0, 0.0, 0.0]] {
            assert!(close3(glow_gamma_sat(c, 1.0, 1.0), c, 1e-4), "{c:?}");
        }
        let g = glow_gamma_sat([0.8, 0.2, 0.1], 1.0, 0.0);
        assert!((g[0] - g[1]).abs() < 1e-4 && (g[1] - g[2]).abs() < 1e-4, "{g:?}");
    }

    #[test]
    fn isolate_is_a_soft_key() {
        assert_eq!(isolate(0.1, 0.2, 1.0), 0.0);
        assert!((isolate(1.0, 0.2, 1.0) - 1.0).abs() < 1e-6);
        assert!(isolate(0.6, 0.2, 1.0) > isolate(0.4, 0.2, 1.0));
    }

    #[test]
    fn neutral_cube_and_regions_are_identity() {
        let cube = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], CYAN, MAGENTA, [1.0, 1.0, 0.0]];
        let regions = [[0.0, 60.0, 0.0, 1.0, 0.0]; 7];
        for c in [[0.2, 0.5, 0.9], [0.9, 0.4, 0.1], [0.5, 0.5, 0.5], [0.1, 0.8, 0.3]] {
            assert!(close3(colour_cube(c, &cube), c, 1e-6), "cube {c:?}");
            assert!(close3(hue_regions(c, &regions), c, 1e-4), "regions {c:?}");
        }
    }

    #[test]
    fn contrast_curve_pivots_on_mid_grey() {
        for contrast in [0.5, 1.0, 1.25, 2.0] {
            let y = contrast_fade(PIVOT, contrast, 1.0, 0.0);
            assert!((y - PIVOT).abs() < 0.01, "{contrast}: {y}");
        }
        // more contrast: darker below the pivot, brighter above
        assert!(contrast_fade(0.2, 1.6, 1.0, 0.0) < contrast_fade(0.2, 1.0, 1.0, 0.0));
        assert!(contrast_fade(0.5, 1.6, 1.0, 0.0) > contrast_fade(0.5, 1.0, 1.0, 0.0));
        // fade lifts black
        assert!(contrast_fade(0.0, 1.0, 1.0, 1.0) > contrast_fade(0.0, 1.0, 1.0, 0.0));
    }

    #[test]
    fn skin_bias_interpolates_the_skin_region() {
        let k = &LOOKS[0].skin;
        assert_eq!(skin_at(k, 0.0), [k[0][1], k[1][1], k[2][1]]);
        assert_eq!(skin_at(k, -1.0), [k[0][0], k[1][0], k[2][0]]);
        assert_eq!(skin_at(k, 1.0), [k[0][2], k[1][2], k[2][2]]);
        // the skin region of every look is its Skin Bias mid point
        for l in &LOOKS {
            assert_eq!([l.regions[1][2], l.regions[1][3], l.regions[1][4]], skin_at(&l.skin, 0.0));
        }
    }
}
