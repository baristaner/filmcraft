//! Film emulation effects: **Halation** (light spill from the film base) and **Film Look**
//! (a creative film look with core looks, colour settings and film effects).
//!
//! They live in their own top-level Effects-panel folder ([`FILM`]) because the "Video Effects"
//! folders mirror Premiere's catalogue. Pixel implementations: `filmcraft_render::vfx::film`.

use super::*;

/// Effects-panel folder of the film emulation effects.
pub const FILM: &[&str] = &["Film Emulation"];

/// Core looks of Film Look, in parameter order.
pub const FILM_LOOKS: &[&str] = &["Cinematic", "Rochester", "Akasaka", "Elated", "Vintage"];

fn r(id: &'static str, label: &'static str, def: f64, hard: (f64, f64), soft: (f64, f64)) -> ParamDef {
    fs(id, label, def, hard, soft, "", 3)
}

fn film(id: &'static str, name: &'static str, params: Vec<ParamDef>) -> EffectDef {
    EffectDef { id, name, kind: EffectKind::Video, category: FILM, params, intrinsic: false, accelerated: false, float32: true, yuv: false }
}

fn halation() -> EffectDef {
    const ISO: &str = "Isolation";
    const DYE: &str = "Dye Layer Reflections";
    const SEC: &str = "Secondary Glow";
    const GRAIN: &str = "Basic Grain";
    const GLOBAL: &str = "Global Adjustments";
    film(
        "halation",
        "Halation",
        vec![
            grp(r("threshold", "Threshold", 0.2, (0.0, 1.0), (0.0, 1.0)), ISO),
            grp(r("normalization", "Normalization", 1.0, (0.0, 1.0), (0.0, 1.0)), ISO),
            grp(r("film_sat_level", "Film Saturation Level", 1.0, (0.1, 100.0), (1.0, 10.0)), ISO),
            grp(b("view_isolated", "View Isolated Regions", false), ISO),
            grp(r("strength", "Strength", 0.5, (0.0, 10.0), (0.0, 1.0)), DYE),
            grp(r("gamma", "Gamma", 1.35, (0.1, 10.0), (0.2, 3.0)), DYE),
            grp(r("saturation", "Saturation", 1.0, (0.0, 5.0), (0.5, 2.0)), DYE),
            grp(r("spread", "Spread", 0.333, (0.0, 10.0), (0.0, 1.0)), DYE),
            grp(b("fine_tune", "Fine Tune Relative Spread", false), DYE),
            grp(r("spread_r", "Relative Spread Red", 1.6, (0.0, 2.0), (0.0, 2.0)), DYE),
            grp(r("spread_g", "Relative Spread Green", 1.0, (0.0, 2.0), (0.0, 2.0)), DYE),
            grp(r("spread_b", "Relative Spread Blue", 0.75, (0.0, 2.0), (0.0, 2.0)), DYE),
            grp(r("sec_strength", "Strength", 0.0, (0.0, 5.0), (0.0, 1.0)), SEC),
            grp(r("sec_gamma", "Gamma", 1.35, (0.1, 10.0), (0.2, 3.0)), SEC),
            grp(r("sec_spread", "Spread", 0.6, (0.0, 5.0), (0.0, 1.0)), SEC),
            grp(col("sec_filter", "Filter", [1.0, 1.0, 1.0, 1.0]), SEC),
            grp(b("grain_on", "Append Grain Internally", false), GRAIN),
            grp(r("grain_strength", "Strength", 0.25, (0.0, 5.0), (0.0, 1.0)), GRAIN),
            grp(r("grain_size", "Size", 0.5, (0.0, 1.0), (0.0, 1.0)), GRAIN),
            grp(b("view_glow_alone", "View Glow Alone", false), GLOBAL),
            grp(r("reduce_highlights", "Reduce Highlights", 0.5, (0.0, 2.0), (0.0, 1.0)), GLOBAL),
            grp(r("aspect_ratio", "Aspect Ratio", 1.0, (0.1, 10.0), (0.5, 2.0)), GLOBAL),
        ],
    )
}

fn film_look() -> EffectDef {
    const LOOK: &str = "Film Look";
    const COLOR: &str = "Color Settings";
    const SPLIT: &str = "Split Tone";
    const VIG: &str = "Vignette";
    const HAL: &str = "Halation";
    const BLOOM: &str = "Bloom";
    const GRAIN: &str = "Grain";
    film(
        "film_look",
        "Film Look",
        vec![
            r("color_blend", "Color Blend", 1.0, (0.0, 1.0), (0.0, 1.0)),
            r("effects_blend", "Effects Blend", 1.0, (0.0, 1.0), (0.0, 1.0)),
            grp(r("film_look_blend", "Film Look Blend", 1.0, (0.0, 1.0), (0.0, 1.0)), LOOK),
            grp(ch("core_look", "Core Look", FILM_LOOKS, 0), LOOK),
            grp(r("skin_bias", "Skin Bias", 0.0, (-1.0, 1.0), (-1.0, 1.0)), LOOK),
            grp(r("exposure", "Exposure", 0.0, (-5.0, 5.0), (-5.0, 5.0)), COLOR),
            grp(r("contrast", "Contrast", 1.25, (0.1, 2.0), (0.1, 2.0)), COLOR),
            grp(r("highlights", "Highlights", 0.35, (0.0, 1.0), (0.0, 1.0)), COLOR),
            grp(r("fade", "Fade", 0.285, (0.0, 1.0), (0.0, 1.0)), COLOR),
            grp(r("subtractive_sat", "Subtractive Sat", 1.2, (0.0, 3.0), (0.0, 3.0)), COLOR),
            grp(r("richness", "Richness", 1.0, (0.0, 3.0), (0.0, 3.0)), COLOR),
            grp(r("bleach_bypass", "Bleach Bypass", 0.0, (0.0, 1.0), (0.0, 1.0)), COLOR),
            grp(b("split_on", "Enable Split Tone", false), SPLIT),
            grp(r("split_amount", "Amount", 0.0, (0.0, 1.0), (0.0, 1.0)), SPLIT),
            grp(fs("split_angle", "Hue Angle", 20.0, (0.0, 360.0), (0.0, 360.0), "°", 0), SPLIT),
            grp(r("split_pivot", "Pivot", 0.3, (0.0, 1.0), (0.0, 1.0)), SPLIT),
            grp(b("vignette_on", "Enable Vignette", false), VIG),
            grp(r("vignette_amount", "Amount", 0.25, (0.0, 1.0), (0.0, 1.0)), VIG),
            grp(r("vignette_size", "Size", 0.25, (0.0, 1.0), (0.0, 1.0)), VIG),
            grp(b("halation_on", "Enable Halation", false), HAL),
            grp(b("halation_highlights_only", "Highlights Only", false), HAL),
            grp(r("halation_amount", "Amount", 0.25, (0.0, 1.0), (0.0, 1.0)), HAL),
            grp(r("halation_radius", "Radius", 4.0, (0.0, 10.0), (0.0, 10.0)), HAL),
            grp(r("halation_sat", "Saturation", 1.0, (0.0, 1.1), (0.0, 1.0)), HAL),
            grp(r("halation_hue", "Hue", 0.5, (0.0, 1.0), (0.0, 1.0)), HAL),
            grp(b("bloom_on", "Enable Bloom", false), BLOOM),
            grp(r("bloom_amount", "Amount", 0.25, (0.0, 1.0), (0.0, 1.0)), BLOOM),
            grp(r("bloom_radius", "Radius", 10.0, (0.0, 100.0), (0.0, 100.0)), BLOOM),
            grp(b("grain_on", "Enable Grain", false), GRAIN),
            grp(r("grain_amount", "Amount", 0.125, (0.0, 1.0), (0.0, 1.0)), GRAIN),
            grp(r("grain_saturation", "Saturation", 0.3, (0.0, 1.0), (0.0, 1.0)), GRAIN),
        ],
    )
}

pub(super) fn defs() -> Vec<EffectDef> {
    vec![halation(), film_look()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn film_effects_are_listed_in_their_own_folder() {
        for id in ["halation", "film_look"] {
            let d = find_effect(id).unwrap_or_else(|| panic!("{id}"));
            assert_eq!(d.kind, EffectKind::Video);
            assert_eq!(d.category, FILM);
            assert!(crate::vtransition::EFFECT_TOP_FOLDERS.contains(&FILM[0]));
            for p in &d.params {
                if let ParamKind::Float { min, max, soft_min, soft_max, .. } = p.kind {
                    let v = p.default.as_f64().unwrap();
                    assert!(v >= min && v <= max, "{id}.{} default {v}", p.id);
                    assert!(soft_min >= min && soft_max <= max, "{id}.{} soft range", p.id);
                }
            }
        }
        assert_eq!(find_effect_by_name("Film Look").unwrap().id, "film_look");
    }
}
