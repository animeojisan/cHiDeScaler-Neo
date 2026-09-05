//! mpv user-shader (.glsl hook) parser: passes, directives, RPN size exprs.
//!
//! Supported grammar (fragment-shader subset):
//!   //!HOOK (multiple) / BIND (multiple) / SAVE / WIDTH / HEIGHT / OFFSET
//!   / COMPONENTS / WHEN / DESC — a directive after body lines starts a new pass.
//!   WIDTH/HEIGHT/WHEN are RPN over NAME.w/NAME.h, literals and + - * / > < >= <= =

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum PassOffset {
    #[default]
    None,
    Pixels(f32, f32),
    Align,
}

#[derive(Default, Debug, Clone)]
pub struct Pass {
    pub hooks: Vec<String>,
    pub binds: Vec<String>,
    pub save: Option<String>,
    pub width: Option<String>,
    pub height: Option<String>,
    pub offset: PassOffset,
    pub compute: Option<ComputeSpec>,
    pub components: u8,
    pub when: Option<String>,
    pub desc: String,
    pub body: Vec<String>,
}

impl Pass {
    pub fn code(&self) -> String {
        self.body.join("\n")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeSpec {
    pub block_w: u32,
    pub block_h: u32,
    pub threads_w: u32,
    pub threads_h: u32,
}

/// GLSL type a //!PARAM is declared as. mpv/libplacebo syntax:
/// `//!TYPE [DYNAMIC|CONSTANT|ENUM] [float|int|uint]` or `//!TYPE DEFINE`.
/// Declaring everything as `uniform float` breaks shaders that assign to int
/// (`int r = RAD;` → "implicit cast from float to int", seen with
/// kgradfun_RT.glsl); DEFINE params are preprocessor text, not uniforms at all
/// (loop bounds / array sizes need them constant).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParamTy {
    #[default]
    Float,
    Int,
    Uint,
    Define,
    /// //!TYPE CONSTANT …: mpv injects the value as a compile-time constant
    /// (crt-royale initializes global `const` structs from these — a uniform
    /// there is a GLSL error "non constant expression in initialization")
    ConstFloat,
    ConstInt,
    ConstUint,
}

impl ParamTy {
    fn parse(v: &str) -> Self {
        let low = v.to_ascii_lowercase();
        let has = |w: &str| low.split_whitespace().any(|t| t == w);
        if has("define") {
            ParamTy::Define
        } else if has("constant") {
            if has("uint") {
                ParamTy::ConstUint
            } else if has("int") {
                ParamTy::ConstInt
            } else {
                ParamTy::ConstFloat
            }
        } else if has("uint") {
            ParamTy::Uint
        } else if has("int") {
            ParamTy::Int
        } else {
            ParamTy::Float
        }
    }
}

/// //!PARAM tunable (float/int/uint/define), user-editable at runtime.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub desc: String,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub value: f32,
    pub ty: ParamTy,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TexBorder {
    #[default]
    Clamp,
    Repeat,
    Mirror,
}

/// //!TEXTURE block: an embedded LUT/data texture (hex-encoded texel data),
/// bindable by name like any saved pass texture. Used heavily by libretro
/// ports (crt-royale etc. embed their phosphor/LUT PNGs this way).
#[derive(Clone, Debug, Default)]
pub struct ShaderTexture {
    pub name: String,
    pub w: i32,
    pub h: i32,
    pub d: i32,
    pub comps: u8,
    /// texel storage: false = unorm8, true = float16 (32f input is converted)
    pub f16: bool,
    pub filter_linear: bool,
    pub border: TexBorder,
    /// //!STORAGE: a writable image2D (imageStore/imageLoad), zero-initialized,
    /// persistent across frames (temporal shaders keep history in these)
    pub storage: bool,
    /// decoded raw texel bytes, tightly packed (empty allowed for STORAGE)
    pub data: Vec<u8>,
}

/// A //!TEXTURE block being collected during parsing.
#[derive(Default)]
struct PendingTexture {
    name: String,
    dims: Vec<i32>,
    format: String,
    filter_linear: bool,
    border: TexBorder,
    storage: bool,
    hex: String,
}

impl PendingTexture {
    /// (comps, f16 storage, source bytes per component, float32 source)
    fn format_info(fmt: &str) -> Option<(u8, bool, usize, bool)> {
        let fmt = fmt.to_ascii_lowercase();
        let comps: u8 = match fmt.trim_end_matches(|c: char| !c.is_ascii_alphabetic()) {
            f if f.starts_with("rgba") => 4,
            f if f.starts_with("rgb") => 3,
            f if f.starts_with("rg") => 2,
            f if f.starts_with('r') => 1,
            _ => return None,
        };
        if fmt.ends_with("32f") || fmt.ends_with("16f") {
            // mpv/libplacebo's shader format deliberately separates the
            // file representation from the GPU representation: both 32f
            // and 16f payloads are written as little-endian f32 values.
            // The driver converts 16f to half precision when uploading.
            Some((comps, true, 4, true))
        } else if fmt.ends_with("16hf") {
            // Some compact shader packs use an explicit half-float
            // extension. Unlike standard mpv 16f, these bytes are already
            // IEEE-754 binary16 and must not be interpreted as f32.
            Some((comps, true, 2, false))
        } else if fmt.ends_with('8') || fmt.chars().all(|c| c.is_ascii_alphabetic()) {
            // rgba8 / rgba (default unorm8)
            Some((comps, false, 1, false))
        } else {
            // 16-bit unorm etc: store as f16 via normalization
            None
        }
    }

    fn finalize(self) -> Option<ShaderTexture> {
        if self.name.is_empty() || self.dims.is_empty() {
            return None;
        }
        if self.dims.len() > 3 {
            log::warn!(
                "//!TEXTURE {}: textures above 3D are not supported",
                self.name
            );
            return None;
        }
        let w = self.dims[0];
        let h = *self.dims.get(1).unwrap_or(&1);
        let d = *self.dims.get(2).unwrap_or(&1);
        let fmt = if self.format.is_empty() {
            "rgba8".to_string()
        } else {
            self.format.clone()
        };
        let Some((comps, f16, src_bytes, is_f32)) = Self::format_info(&fmt) else {
            log::warn!("//!TEXTURE {}: unsupported FORMAT {fmt}", self.name);
            return None;
        };
        if self.storage && self.hex.trim().is_empty() {
            // STORAGE scratch texture: zero-initialized, no embedded data
            let store_bytes = if f16 { 2 } else { 1 };
            return Some(ShaderTexture {
                name: self.name,
                w,
                h,
                d,
                comps,
                f16,
                filter_linear: self.filter_linear,
                border: self.border,
                storage: true,
                data: vec![
                    0u8;
                    (w as usize) * (h as usize) * (d as usize) * comps as usize * store_bytes
                ],
            });
        }
        let raw = decode_hex(&self.hex)?;
        let expected = (w as usize) * (h as usize) * (d as usize) * comps as usize * src_bytes;
        if raw.len() < expected {
            log::warn!(
                "//!TEXTURE {}: data too short ({} < {expected} bytes)",
                self.name,
                raw.len()
            );
            return None;
        }
        let data = if is_f32 {
            // convert little-endian f32 texels to f16 storage
            raw[..expected]
                .chunks_exact(4)
                .flat_map(|c| {
                    let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    half::f16::from_f32(v).to_le_bytes()
                })
                .collect()
        } else {
            raw[..expected].to_vec()
        };
        Some(ShaderTexture {
            name: self.name,
            w,
            h,
            d,
            comps,
            f16,
            filter_linear: self.filter_linear,
            border: self.border,
            storage: self.storage,
            data,
        })
    }
}

fn decode_hex(hex: &str) -> Option<Vec<u8>> {
    let clean: Vec<u8> = hex.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if clean.len() % 2 != 0 {
        return None;
    }
    let nib = |b: u8| -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    };
    clean
        .chunks_exact(2)
        .map(|c| Some((nib(c[0])? << 4) | nib(c[1])?))
        .collect()
}

#[derive(Clone)]
pub struct UserShader {
    pub path: String,
    /// Content identity used by the GL program cache. Computing this once at
    /// load time avoids hashing very large generated shaders every frame.
    pub source_hash: u64,
    pub passes: Vec<Pass>,
    pub is_rgb: bool,
    /// Requires an actual CHROMA plane. LUMA-only shaders (FSRCNNX/RAVU-lite)
    /// do not need full YUV emulation: they can preserve the original RGB
    /// chroma and substitute only the processed luma.
    pub uses_chroma: bool,
    pub is_compute: bool,
    /// hooks only OUTPUT/SCALED etc: runs on the final display-size image
    /// after scaling, not on the source planes
    pub is_post: bool,
    /// Neo extension: rerun this stage and the remaining chain at the physical
    /// display refresh cadence. Ordinary shaders never set this flag.
    pub display_hz: bool,
    pub params: Vec<Param>,
    /// //!TEXTURE embedded data textures (LUTs), bindable by name
    pub textures: Vec<ShaderTexture>,
}

impl UserShader {
    pub fn load(path: &str) -> std::io::Result<Self> {
        let src = std::fs::read_to_string(path)?;
        Ok(Self::parse(path, &src))
    }

    pub fn parse(path: &str, src: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        src.hash(&mut hasher);
        let source_hash = hasher.finish();
        let (passes, params, textures) = parse_passes(src);
        let passes = expand_multi_hook_passes(passes);
        let display_hz = src
            .lines()
            .any(|line| line.trim().eq_ignore_ascii_case("//!DISPLAY_HZ"));
        let mut hooks_rgb = false;
        let mut uses_chroma = false;
        let mut is_compute = false;
        let mut any_hook = false;
        let mut all_post = true;
        for p in &passes {
            for h in &p.hooks {
                any_hook = true;
                if is_color_hook(h) {
                    hooks_rgb = true;
                }
                if is_chroma_hook(h) {
                    uses_chroma = true;
                }
                if !is_post_hook(h) {
                    all_post = false;
                }
            }
            for b in &p.binds {
                if is_chroma_hook(&b.to_uppercase()) {
                    uses_chroma = true;
                }
            }
        }
        if src.contains("//!COMPUTE") {
            is_compute = true;
        }
        Self {
            path: path.to_string(),
            source_hash,
            passes,
            is_rgb: hooks_rgb,
            uses_chroma,
            is_compute,
            is_post: any_hook && all_post,
            display_hz,
            params,
            textures,
        }
    }

    pub fn name(&self) -> String {
        std::path::Path::new(&self.path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.path.clone())
    }
}

fn expand_multi_hook_passes(passes: Vec<Pass>) -> Vec<Pass> {
    let mut out = Vec::with_capacity(passes.len());
    for p in passes {
        if p.save.is_none() && p.hooks.len() > 1 {
            for hook in &p.hooks {
                let mut one = p.clone();
                one.hooks = vec![hook.clone()];
                out.push(one);
            }
        } else {
            out.push(p);
        }
    }
    out
}

fn parse_passes(src: &str) -> (Vec<Pass>, Vec<Param>, Vec<ShaderTexture>) {
    let mut passes: Vec<Pass> = Vec::new();
    let mut cur: Option<usize> = None;
    let mut params: Vec<Param> = Vec::new();
    let mut textures: Vec<ShaderTexture> = Vec::new();
    // //!PARAM block being collected: the first non-directive line is its
    // default value (it must NOT become shader body)
    let mut pending_param: Option<Param> = None;
    // //!TEXTURE block being collected: its non-directive lines are hex texel
    // data (they must NOT become shader body — before this existed, embedded
    // LUT shaders like crt-royale compiled megabytes of hex as GLSL)
    let mut pending_texture: Option<PendingTexture> = None;
    for line in src.lines() {
        let s = line.trim();
        if let Some(rest) = s.strip_prefix("//!") {
            let (pkey, pval) = match rest.trim().split_once(' ') {
                Some((k, v)) => (k.to_uppercase(), v.trim().to_string()),
                None => (rest.trim().to_uppercase(), String::new()),
            };
            if pkey == "TEXTURE" {
                if let Some(p) = pending_param.take() {
                    params.push(p);
                }
                if let Some(t) = pending_texture.take().and_then(PendingTexture::finalize) {
                    textures.push(t);
                }
                pending_texture = Some(PendingTexture {
                    name: pval,
                    ..Default::default()
                });
                continue;
            }
            if let Some(t) = pending_texture.as_mut() {
                match pkey.as_str() {
                    "SIZE" => {
                        t.dims = pval
                            .split_whitespace()
                            .filter_map(|v| v.parse().ok())
                            .collect();
                        continue;
                    }
                    "FORMAT" => {
                        t.format = pval;
                        continue;
                    }
                    "FILTER" => {
                        t.filter_linear = pval.eq_ignore_ascii_case("LINEAR");
                        continue;
                    }
                    "BORDER" => {
                        t.border = match pval.to_ascii_uppercase().as_str() {
                            "REPEAT" => TexBorder::Repeat,
                            "MIRROR" => TexBorder::Mirror,
                            _ => TexBorder::Clamp,
                        };
                        continue;
                    }
                    "STORAGE" => {
                        t.storage = true;
                        continue;
                    }
                    _ => {
                        // a different directive ends the texture block
                        if let Some(t) = pending_texture.take().and_then(PendingTexture::finalize) {
                            textures.push(t);
                        }
                    }
                }
            }
            if pkey == "PARAM" {
                if let Some(p) = pending_param.take() {
                    params.push(p);
                }
                pending_param = Some(Param {
                    name: pval,
                    desc: String::new(),
                    min: f32::NEG_INFINITY,
                    max: f32::INFINITY,
                    default: 0.0,
                    value: 0.0,
                    ty: ParamTy::Float,
                });
                continue;
            }
            if let Some(p) = pending_param.as_mut() {
                match pkey.as_str() {
                    "DESC" => {
                        p.desc = pval;
                        continue;
                    }
                    "TYPE" => {
                        p.ty = ParamTy::parse(&pval);
                        continue;
                    }
                    "MINIMUM" => {
                        p.min = pval.parse().unwrap_or(f32::NEG_INFINITY);
                        continue;
                    }
                    "MAXIMUM" => {
                        p.max = pval.parse().unwrap_or(f32::INFINITY);
                        continue;
                    }
                    _ => {
                        // a different directive ends the param block
                        params.push(pending_param.take().unwrap());
                    }
                }
            }
            let need_new = match cur {
                None => true,
                Some(i) => !passes[i].body.is_empty(),
            };
            if need_new {
                passes.push(Pass {
                    components: 4,
                    ..Default::default()
                });
                cur = Some(passes.len() - 1);
            }
            let p = &mut passes[cur.unwrap()];
            let (key, val) = match rest.trim().split_once(' ') {
                Some((k, v)) => (k.to_uppercase(), v.trim().to_string()),
                None => (rest.trim().to_uppercase(), String::new()),
            };
            match key.as_str() {
                "HOOK" => p.hooks.push(val.to_uppercase()),
                "BIND" => p.binds.push(val),
                "SAVE" => p.save = Some(val),
                "WIDTH" => p.width = Some(val),
                "HEIGHT" => p.height = Some(val),
                "OFFSET" => p.offset = parse_offset(&val),
                "COMPUTE" => p.compute = parse_compute(&val),
                "COMPONENTS" => {
                    p.components = val
                        .split_whitespace()
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(4)
                }
                "WHEN" => p.when = Some(val),
                "DESC" => p.desc = val,
                _ => {}
            }
        } else if let Some(t) = pending_texture.as_mut() {
            // hex texel data (may span multiple lines)
            t.hex.push_str(s);
        } else if let Some(mut p) = pending_param.take() {
            // first non-directive line after //!PARAM = default value
            if let Ok(v) = s.parse::<f32>() {
                p.default = v;
                p.value = v;
                params.push(p);
            } else {
                params.push(p); // malformed default: keep 0
            }
        } else if s.starts_with("//") {
            if let Some(i) = cur {
                if !passes[i].body.is_empty() {
                    passes[i].body.push(line.to_string());
                }
            }
        } else if let Some(i) = cur {
            passes[i].body.push(line.to_string());
        }
    }
    if let Some(p) = pending_param.take() {
        params.push(p);
    }
    if let Some(t) = pending_texture.take().and_then(PendingTexture::finalize) {
        textures.push(t);
    }
    (
        passes
            .into_iter()
            .filter(|p| p.body.iter().any(|l| !l.trim().is_empty()))
            .collect(),
        params,
        textures,
    )
}

fn parse_offset(val: &str) -> PassOffset {
    if val.trim().eq_ignore_ascii_case("ALIGN") {
        return PassOffset::Align;
    }
    let mut it = val.split_whitespace().filter_map(|v| v.parse::<f32>().ok());
    match (it.next(), it.next()) {
        (Some(x), Some(y)) => PassOffset::Pixels(x, y),
        _ => PassOffset::None,
    }
}

fn parse_compute(val: &str) -> Option<ComputeSpec> {
    let nums: Vec<u32> = val
        .split_whitespace()
        .filter_map(|v| v.parse::<u32>().ok())
        .collect();
    // mpv: //!COMPUTE bw bh [tw th] — threads default to the block size
    if nums.len() < 2 {
        return None;
    }
    Some(ComputeSpec {
        block_w: nums[0].max(1),
        block_h: nums[1].max(1),
        threads_w: nums.get(2).copied().unwrap_or(nums[0]).max(1),
        threads_h: nums.get(3).copied().unwrap_or(nums[1]).max(1),
    })
}

fn is_color_hook(h: &str) -> bool {
    matches!(
        h,
        "MAIN" | "RGB" | "NATIVE" | "MAINPRESUB" | "OUTPUT" | "SCALED" | "PREKERNEL" | "POSTKERNEL"
    )
}

fn is_chroma_hook(h: &str) -> bool {
    h == "CHROMA"
}

fn is_post_hook(h: &str) -> bool {
    matches!(h, "OUTPUT" | "SCALED" | "PREKERNEL" | "POSTKERNEL")
}

pub type Sizes = HashMap<String, (f64, f64)>;
pub type Params = HashMap<String, f64>;

/// Evaluate an mpv RPN size/condition expression.
/// Compute only the requested operator; and
/// division by zero yields 0.0 (a `0/0` at ratio ~1.2 used to crash mpv-style
/// WHEN chains).
pub fn eval_rpn(expr: &str, sizes: &Sizes) -> Option<f64> {
    eval_rpn_p(expr, sizes, &Params::new())
}

/// RPN with //!PARAM name lookups (bare identifiers).
pub fn eval_rpn_p(expr: &str, sizes: &Sizes, params: &Params) -> Option<f64> {
    let expr = expr.trim();
    if expr.is_empty() {
        return None;
    }
    let mut st: Vec<f64> = Vec::new();
    for tok in expr.split_whitespace() {
        match tok {
            "!" => {
                let a = st.pop()?;
                st.push((a == 0.0) as i32 as f64);
            }
            "+" | "-" | "*" | "/" | ">" | "<" | ">=" | "<=" | "=" => {
                let b = st.pop()?;
                let a = st.pop()?;
                let r = match tok {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => {
                        if b != 0.0 {
                            a / b
                        } else {
                            0.0
                        }
                    }
                    ">" => (a > b) as i32 as f64,
                    "<" => (a < b) as i32 as f64,
                    ">=" => (a >= b) as i32 as f64,
                    "<=" => (a <= b) as i32 as f64,
                    _ => (a == b) as i32 as f64,
                };
                st.push(r);
            }
            _ => {
                if let Some((name, attr)) = split_size_ref(tok) {
                    let (w, h) = *sizes.get(name)?;
                    st.push(if attr == "w" || attr == "width" { w } else { h });
                } else if let Some(v) = params.get(tok) {
                    st.push(*v);
                } else {
                    st.push(tok.parse().ok()?);
                }
            }
        }
    }
    st.last().copied()
}

fn split_size_ref(tok: &str) -> Option<(&str, &str)> {
    let (name, attr) = tok.rsplit_once('.')?;
    if name.is_empty()
        || !name
            .chars()
            .next()
            .map(|c| c.is_ascii_alphabetic() || c == '_')
            .unwrap_or(false)
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return None;
    }
    matches!(attr, "w" | "h" | "width" | "height").then_some((name, attr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpn_basic() {
        let mut s = Sizes::new();
        s.insert("MAIN".into(), (640.0, 360.0));
        s.insert("OUTPUT".into(), (1280.0, 720.0));
        assert_eq!(eval_rpn("MAIN.w 2 *", &s), Some(1280.0));
        // Anime4K-style WHEN: OUTPUT.w MAIN.w / 1.200 > OUTPUT.h MAIN.h / 1.200 > *
        let e = "OUTPUT.w MAIN.w / 1.200 > OUTPUT.h MAIN.h / 1.200 > *";
        assert_eq!(eval_rpn(e, &s), Some(1.0));
        // ratio 1.0 -> both comparisons false -> 0*0, must not crash
        s.insert("OUTPUT".into(), (640.0, 360.0));
        assert_eq!(eval_rpn(e, &s), Some(0.0));
        assert_eq!(eval_rpn("MAIN.w OUTPUT.w = !", &s), Some(0.0));
        s.insert("OUTPUT".into(), (1280.0, 720.0));
        assert_eq!(eval_rpn("MAIN.w OUTPUT.w = !", &s), Some(1.0));
    }

    #[test]
    fn parse_params() {
        let src = "//!PARAM CB\n//!TYPE float\n//!MINIMUM 0.0\n//!MAXIMUM 5.0\n2.0\n\n//!HOOK SCALED\n//!BIND HOOKED\n//!WHEN CB 0 >\nvec4 hook(){return vec4(1.0);}\n";
        let sh = UserShader::parse("p.glsl", src);
        assert_eq!(sh.params.len(), 1);
        assert_eq!(sh.params[0].name, "CB");
        assert!((sh.params[0].default - 2.0).abs() < 1e-6);
        assert!(sh.is_post);
        assert_eq!(sh.passes.len(), 1);
        let mut pm = Params::new();
        pm.insert("CB".into(), 2.0);
        assert_eq!(eval_rpn_p("CB 0 >", &Sizes::new(), &pm), Some(1.0));
    }

    #[test]
    fn texture_float_formats_follow_mpv_payload_encoding() {
        assert_eq!(
            PendingTexture::format_info("rgba16f"),
            Some((4, true, 4, true))
        );
        assert_eq!(
            PendingTexture::format_info("rg16f"),
            Some((2, true, 4, true))
        );
        assert_eq!(
            PendingTexture::format_info("rgba16hf"),
            Some((4, true, 2, false))
        );
        assert_eq!(
            PendingTexture::format_info("r32f"),
            Some((1, true, 4, true))
        );
    }

    #[test]
    fn parse_embedded_3d_half_float_lut() {
        let src =
            "//!TEXTURE LUT\n//!SIZE 1 1 2\n//!FORMAT rgba16hf\n00000000000000000000000000000000\n";
        let shader = UserShader::parse("lut.glsl", src);
        assert_eq!(shader.textures.len(), 1);
        let lut = &shader.textures[0];
        assert_eq!((lut.w, lut.h, lut.d, lut.comps), (1, 1, 2, 4));
        assert_eq!(lut.data.len(), 16);
    }

    #[test]
    fn param_types_parse_like_mpv() {
        // kgradfun_RT.glsl regression: //!TYPE int must become `uniform int`
        // (declaring it float broke `int r = RAD;` with an implicit-cast error)
        let src = "//!PARAM STR\n//!TYPE float\n1.2\n\n//!PARAM RAD\n//!TYPE int\n//!MINIMUM 4\n//!MAXIMUM 32\n16\n\n//!PARAM TAPS\n//!TYPE DEFINE\n3\n\n//!PARAM MODE\n//!TYPE CONSTANT uint\n2\n\n//!HOOK LUMA\n//!BIND HOOKED\nvec4 hook(){int r = RAD; return HOOKED_tex(HOOKED_pos);}\n";
        let sh = UserShader::parse("t.glsl", src);
        assert_eq!(sh.params.len(), 4);
        assert_eq!(sh.params[0].ty, ParamTy::Float);
        assert_eq!(sh.params[1].ty, ParamTy::Int);
        assert!((sh.params[1].default - 16.0).abs() < 1e-6);
        assert_eq!(sh.params[2].ty, ParamTy::Define);
        assert_eq!(sh.params[3].ty, ParamTy::ConstUint);
    }

    #[test]
    fn parse_param_desc_does_not_create_shader_pass() {
        let src = "//!PARAM bright\n//!DESC Brightness\n//!TYPE CONSTANT float\n//!MINIMUM 0\n//!MAXIMUM 2\n1.2\n\n//!HOOK MAIN\n//!BIND HOOKED\nvec4 hook(){return HOOKED_tex(HOOKED_pos) * bright;}\n";
        let sh = UserShader::parse("crt.glsl", src);
        assert_eq!(sh.params.len(), 1);
        assert_eq!(sh.params[0].name, "bright");
        assert_eq!(sh.params[0].desc, "Brightness");
        assert!((sh.params[0].default - 1.2).abs() < 1e-6);
        assert_eq!(sh.passes.len(), 1);
        assert_eq!(sh.passes[0].desc, "");
        assert!(sh.is_rgb);
    }

    #[test]
    fn parse_display_hz_extension_is_opt_in() {
        let ordinary = UserShader::parse(
            "ordinary.glsl",
            "//!HOOK RGB\n//!BIND HOOKED\nvec4 hook(){return HOOKED_tex(HOOKED_pos);}",
        );
        let refresh = UserShader::parse(
            "refresh.glsl",
            "//!DISPLAY_HZ\n//!HOOK RGB\n//!BIND HOOKED\nvec4 hook(){return HOOKED_tex(HOOKED_pos);}",
        );
        assert!(!ordinary.display_hz);
        assert!(refresh.display_hz);
    }

    #[test]
    fn parse_two_passes() {
        let src = "//!DESC pass one\n//!HOOK MAIN\n//!BIND HOOKED\nvec4 hook(){return vec4(1.0);}\n//!DESC pass two\n//!HOOK MAIN\n//!BIND HOOKED\n//!SAVE X\n//!COMPONENTS 2\nvec4 hook(){return vec4(0.5);}\n";
        let sh = UserShader::parse("test.glsl", src);
        assert_eq!(sh.passes.len(), 2);
        assert!(sh.is_rgb);
        assert_eq!(sh.passes[1].save.as_deref(), Some("X"));
        assert_eq!(sh.passes[1].components, 2);
    }

    #[test]
    fn parse_texture_block_with_hex_data() {
        // crt-royale style: embedded LUT. The hex data must become texture
        // bytes, NOT shader body, and the block must not create a pass.
        let src = "//!TEXTURE LUT1\n//!SIZE 2 2\n//!BORDER REPEAT\n//!FILTER LINEAR\n//!FORMAT rgba8\n00112233445566778899aabbccddeeff\n\n//!HOOK MAIN\n//!BIND HOOKED\n//!BIND LUT1\nvec4 hook(){return LUT1_tex(HOOKED_pos);}\n";
        let sh = UserShader::parse("t.glsl", src);
        assert_eq!(sh.passes.len(), 1, "TEXTURE block must not become a pass");
        assert_eq!(sh.textures.len(), 1);
        let t = &sh.textures[0];
        assert_eq!(t.name, "LUT1");
        assert_eq!((t.w, t.h, t.comps, t.f16), (2, 2, 4, false));
        assert!(t.filter_linear);
        assert_eq!(t.border, TexBorder::Repeat);
        assert_eq!(t.data.len(), 16);
        assert_eq!(t.data[0], 0x00);
        assert_eq!(t.data[15], 0xff);
        // pass body must contain only the hook fn
        assert!(sh.passes[0].code().contains("LUT1_tex"));
        assert!(!sh.passes[0].code().contains("00112233"));
    }

    #[test]
    fn parse_texture_f32_converts_to_f16() {
        // 1x1 r32f texel = 1.0f LE
        let src = "//!TEXTURE W\n//!SIZE 1 1\n//!FORMAT r32f\n0000803f\n\n//!HOOK MAIN\n//!BIND HOOKED\nvec4 hook(){return HOOKED_tex(HOOKED_pos);}\n";
        let sh = UserShader::parse("t.glsl", src);
        assert_eq!(sh.textures.len(), 1);
        let t = &sh.textures[0];
        assert!(t.f16);
        assert_eq!(t.comps, 1);
        assert_eq!(t.data, half::f16::from_f32(1.0).to_le_bytes().to_vec());
    }

    #[test]
    fn parse_offset_directive_and_align() {
        let numeric = UserShader::parse(
            "offset.glsl",
            "//!HOOK MAIN\n//!BIND HOOKED\n//!OFFSET -0.5 1.25\nvec4 hook(){return HOOKED_tex(HOOKED_pos);}\n",
        );
        assert_eq!(numeric.passes[0].offset, PassOffset::Pixels(-0.5, 1.25));

        let align = UserShader::parse(
            "align.glsl",
            "//!HOOK MAIN\n//!BIND HOOKED\n//!OFFSET ALIGN\nvec4 hook(){return HOOKED_tex(HOOKED_pos);}\n",
        );
        assert_eq!(align.passes[0].offset, PassOffset::Align);
    }

    #[test]
    fn luma_only_shader_does_not_require_chroma_emulation() {
        let luma_only = UserShader::parse(
            "fsrcnnx.glsl",
            "//!HOOK LUMA\n//!BIND LUMA\nvec4 hook(){return vec4(LUMA_tex(LUMA_pos));}\n",
        );
        assert!(!luma_only.is_rgb);
        assert!(!luma_only.uses_chroma);

        let chroma = UserShader::parse(
            "chroma.glsl",
            "//!HOOK CHROMA\n//!BIND CHROMA\nvec4 hook(){return CHROMA_tex(CHROMA_pos);}\n",
        );
        assert!(chroma.uses_chroma);
    }

    #[test]
    fn parse_compute_directive() {
        let src =
            "//!HOOK LUMA\n//!COMPUTE 16 8 8 8\n//!BIND LUMA\nvec4 hook(){return vec4(0.0);}\n";
        let sh = UserShader::parse("c.glsl", src);
        assert_eq!(
            sh.passes[0].compute,
            Some(ComputeSpec {
                block_w: 16,
                block_h: 8,
                threads_w: 8,
                threads_h: 8,
            })
        );
        assert!(sh.is_compute);
    }
}
