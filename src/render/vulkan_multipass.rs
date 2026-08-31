//! v634 backend-neutral mpv multi-pass planner + experimental Vulkan executor.
//!
//! Neo's existing mpv parser (`UserShader`) remains authoritative. The first
//! half of this module lowers its BIND/SAVE/WIDTH/HEIGHT/WHEN graph into Vulkan
//! compute SPIR-V. The second half can execute the admitted Anime4K CNN family
//! on the explicitly selected Vulkan GPU behind an opt-in test flag.
//! GPU=Auto and unsupported shaders remain on the established OpenGL path.
//!
//! v634 scope:
//! - ordinary RGB and LUMA fragment-style multi-pass shaders
//! - //!BIND / //!SAVE
//! - //!WIDTH / //!HEIGHT / //!WHEN
//! - output COMPONENTS 1/2/4
//! - current //!PARAM values compiled as typed constants
//! - named intermediate resources and HOOKED aliases
//! - per-pass sampled-image descriptors + storage-image output
//! - Naga GLSL -> SPIR-V validation for every active pass
//!
//! Still rejected by the independent multi-pass production path:
//! - //!TEXTURE / //!STORAGE embedded resources
//! - CHROMA plane emulation
//!
//! The important architectural change is that compatibility is no longer
//! defined as "exactly one pass". Neo's existing parser owns mpv semantics;
//! this module consumes the parsed pass graph and lowers it for Vulkan.

use super::gl::{GlContext, GpuTex};
use super::mpv::{
    ParamTy, Params, Pass, PassOffset, ShaderTexture, Sizes, TexBorder, UserShader, eval_rpn_p,
};
use anyhow::{Context, Result, anyhow};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct PlannedImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub components: u8,
    pub offset_x: f32,
    pub offset_y: f32,
}

#[derive(Clone, Debug)]
pub struct PlannedBind {
    /// Name exposed to GLSL (`MAIN`, `HOOKED`, `conv2d_tf`, ...).
    pub shader_name: String,
    /// Concrete resource name in the pass graph.
    pub resource_name: String,
    pub width: u32,
    pub height: u32,
    pub components: u8,
    pub depth: u32,
    pub storage: bool,
}

#[derive(Clone, Debug)]
pub struct PlannedPass {
    pub source_index: usize,
    pub desc: String,
    pub hooked: String,
    pub save: String,
    pub width: u32,
    pub height: u32,
    pub components: u8,
    pub dispatch_block_w: u32,
    pub dispatch_block_h: u32,
    pub binds: Vec<PlannedBind>,
    pub spirv: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct MultiPassPlan {
    pub shader_name: String,
    pub passes: Vec<PlannedPass>,
    pub final_resource: String,
    pub final_width: u32,
    pub final_height: u32,
    pub final_components: u8,
    pub total_spirv_words: usize,
    pub textures: Vec<ShaderTexture>,
    pub final_luma_resource: Option<String>,
    pub final_chroma_resource: Option<String>,
}

#[derive(Clone, Debug)]
pub struct MultiPassCompatibility {
    pub compatible: bool,
    pub reason: String,
    pub active_passes: usize,
    pub total_passes: usize,
    pub spirv_words: usize,
    pub final_width: u32,
    pub final_height: u32,
}

fn canonical_tex_name(name: &str) -> &str {
    match name {
        "NATIVE" | "MAINPRESUB" => "MAIN",
        "SCALED" | "PREKERNEL" | "POSTKERNEL" => "OUTPUT",
        _ => name,
    }
}

fn hooked_texture_key(p: &Pass, images: &HashMap<String, PlannedImage>) -> String {
    for h in &p.hooks {
        let key = canonical_tex_name(h);
        if images.contains_key(key) {
            return key.to_string();
        }
        if images.contains_key(h) {
            return h.clone();
        }
    }
    p.hooks
        .first()
        .map(|hook| canonical_tex_name(hook).to_string())
        .unwrap_or_else(|| "MAIN".to_string())
}

fn raw_hook_name(p: &Pass, hooked: &str) -> String {
    p.hooks
        .iter()
        .find(|h| h.as_str() == hooked || canonical_tex_name(h) == hooked)
        .cloned()
        .unwrap_or_else(|| hooked.to_string())
}

fn glsl_float_literal(value: f32) -> String {
    if !value.is_finite() {
        return "0.0".to_string();
    }
    let mut text = format!("{value:?}");
    if !text.contains('.') && !text.contains('e') && !text.contains('E') {
        text.push_str(".0");
    }
    text
}

fn parameter_prelude(shader: &UserShader) -> String {
    let mut out = String::new();
    // mpv exposes these even when the shader did not declare them as PARAMs.
    if !shader.params.iter().any(|p| p.name == "random") {
        out.push_str("const float random = 0.0;\n");
    }
    if !shader.params.iter().any(|p| p.name == "frame") {
        out.push_str("const int frame = 0;\n");
    }
    for param in &shader.params {
        let name = &param.name;
        let value = param.value;
        match param.ty {
            ParamTy::Define => {
                if value.fract() == 0.0 && value.abs() < 1e9 {
                    out.push_str(&format!("#define {name} {}\n", value as i64));
                } else {
                    out.push_str(&format!("#define {name} {}\n", glsl_float_literal(value)));
                }
            }
            ParamTy::Int | ParamTy::ConstInt => {
                out.push_str(&format!("const int {name} = {};\n", value.round() as i64));
            }
            ParamTy::Uint | ParamTy::ConstUint => {
                out.push_str(&format!(
                    "const uint {name} = {}u;\n",
                    value.round().max(0.0) as u64
                ));
            }
            ParamTy::Float | ParamTy::ConstFloat => {
                out.push_str(&format!(
                    "const float {name} = {};\n",
                    glsl_float_literal(value)
                ));
            }
        }
    }
    out
}

fn output_layout(components: u8) -> Result<&'static str> {
    match components {
        1 => Ok("r16f"),
        2 => Ok("rg16f"),
        3 | 4 => Ok("rgba16f"),
        other => Err(anyhow!(
            "unsupported COMPONENTS={other}; Vulkan path supports 1/2/3/4"
        )),
    }
}

fn sample_expr(name: &str, components: u8, expr: &str) -> String {
    let _ = components;
    format!("textureLod({name}_tx, ({expr}), 0.0)")
}

fn sample_off_expr(name: &str, components: u8, pos: &str, off: &str) -> String {
    let uv = format!("({pos} + vec2({off}) * {name}_pt)");
    sample_expr(name, components, &uv)
}

fn append_bind_prelude(out: &mut String, binding: u32, bind: &PlannedBind, out_w: u32, out_h: u32) {
    let n = &bind.shader_name;
    if bind.storage {
        let layout = match (bind.components, bind.depth) {
            (_, d) if d > 1 => "rgba16f",
            (1, _) => "r16f",
            (2, _) => "rg16f",
            _ => "rgba16f",
        };
        out.push_str(&format!(
            "layout({layout}, set = 0, binding = {binding}) uniform image2D {n};\n\
const vec2 {n}_size = vec2({}.0, {}.0);\n\
const vec2 {n}_pt = vec2(1.0 / {}.0, 1.0 / {}.0);\n\
#define {n}_map(id) ((vec2(id) + vec2(0.5)) / vec2({out_w}.0, {out_h}.0))\n\
#define {n}_pos {n}_map(neo_invocation_p)\n\
vec4 neo_{n}_tex(vec2 uv) {{ return imageLoad({n}, clamp(ivec2(uv * {n}_size), ivec2(0), ivec2({n}_size) - ivec2(1))); }}\n\
#define {n}_tex(p) neo_{n}_tex(p)\n\
#define {n}_texOff(o) neo_{n}_tex({n}_pos + vec2(o) * {n}_pt)\n\
#define {n}_raw {n}\n\
#define {n}_mul 1.0\n\
#define {n}_rot mat2(1.0, 0.0, 0.0, 1.0)\n\
#define {n}_off vec2(0.0)\n",
            bind.width,
            bind.height,
            bind.width.max(1),
            bind.height.max(1),
        ));
        return;
    }
    let sampler_type = if bind.depth > 1 { "3D" } else { "2D" };
    let sampler_binding = binding + 1;
    out.push_str(&format!(
        "layout(set = 0, binding = {binding}) uniform texture{sampler_type} {n}_texture;\n\
layout(set = 0, binding = {sampler_binding}) uniform sampler {n}_sampler;\n\
#define {n}_tx sampler{sampler_type}({n}_texture, {n}_sampler)\n"
    ));
    if bind.depth > 1 {
        out.push_str(&format!(
            "const vec2 {n}_size = vec2({}.0, {}.0);\n\
const vec2 {n}_pt = vec2(1.0 / {}.0, 1.0 / {}.0);\n\
#define {n}_map(id) ((vec2(id) + vec2(0.5)) / vec2({out_w}.0, {out_h}.0))\n\
#define {n}_pos {n}_map(neo_invocation_p)\n\
#define {n}_tex(p) textureLod({n}_tx, (p), 0.0)\n\
#define {n}_texOff(o) textureLodOffset({n}_tx, vec3({n}_pos, 0.5), 0.0, ivec3((o), 0))\n\
#define {n}_off vec2(0.0)\n\
#ifndef {n}_raw\n#define {n}_raw {n}_tx\n#endif\n\
#ifndef {n}_mul\n#define {n}_mul 1.0\n#endif\n\
#ifndef {n}\n#define {n} {n}_tx\n#endif\n",
            bind.width,
            bind.height,
            bind.width.max(1),
            bind.height.max(1),
        ));
        return;
    }
    out.push_str(&format!(
        "\
const vec2 {n}_size = vec2({}.0, {}.0);\n\
const vec2 {n}_pt = vec2(1.0 / {}.0, 1.0 / {}.0);\n\
#define {n}_map(id) ((vec2(id) + vec2(0.5)) / vec2({out_w}.0, {out_h}.0))\n\
#define {n}_pos {n}_map(neo_invocation_p)\n\
#ifndef {n}_raw\n#define {n}_raw {n}_tx\n#endif\n\
#ifndef {n}_mul\n#define {n}_mul 1.0\n#endif\n\
#ifndef {n}_rot\n#define {n}_rot mat2(1.0, 0.0, 0.0, 1.0)\n#endif\n\
#define {n}_tex(p) ({})\n\
#define {n}_texOff(o) ({})\n\
vec4 neo_{n}_gather_off(vec2 uv, ivec2 off, int c) {{\n\
    ivec2 q = ivec2(floor(uv * {n}_size - vec2(0.5))) + off;\n\
    vec4 a = texelFetch({n}_tx, q + ivec2(0,1), 0);\n\
    vec4 b = texelFetch({n}_tx, q + ivec2(1,1), 0);\n\
    vec4 d = texelFetch({n}_tx, q + ivec2(1,0), 0);\n\
    vec4 e = texelFetch({n}_tx, q + ivec2(0,0), 0);\n\
    vec4 mask = c == 0 ? vec4(1,0,0,0) : (c == 1 ? vec4(0,1,0,0) : (c == 2 ? vec4(0,0,1,0) : vec4(0,0,0,1)));\n\
    return vec4(dot(a,mask), dot(b,mask), dot(d,mask), dot(e,mask));\n\
}}\n\
vec4 neo_{n}_gather(vec2 uv, int c) {{ return neo_{n}_gather_off(uv, ivec2(0), c); }}\n\
#define {n}_gather(p,c) neo_{n}_gather((p), (c))\n\
#define {n}_off vec2(0.0)\n\
#ifndef {n}\n#define {n} {n}_tx\n#endif\n",
        bind.width,
        bind.height,
        bind.width.max(1),
        bind.height.max(1),
        sample_expr(n, bind.components, "p"),
        sample_off_expr(n, bind.components, &format!("{n}_pos"), "o"),
    ));
}

fn append_hook_aliases(out: &mut String, raw_hook: &str, binds: &[PlannedBind]) {
    let has = |name: &str| binds.iter().any(|b| b.shader_name == name);
    let target = if has(raw_hook) {
        raw_hook.to_string()
    } else if has("HOOKED") {
        "HOOKED".to_string()
    } else if let Some(first) = binds.first() {
        first.shader_name.clone()
    } else {
        return;
    };

    let aliases: Vec<&str> = match raw_hook {
        "MAIN" => vec!["MAIN"],
        "RGB" => vec!["RGB", "MAIN"],
        "NATIVE" => vec!["NATIVE", "MAIN"],
        "MAINPRESUB" => vec!["MAINPRESUB", "MAIN"],
        "OUTPUT" | "SCALED" | "PREKERNEL" | "POSTKERNEL" => vec!["OUTPUT"],
        other => vec![other],
    };
    for alias in &aliases {
        if has(alias) || *alias == target {
            continue;
        }
        out.push_str(&format!(
            "#ifndef {alias}_raw\n#define {alias}_raw {target}_raw\n#endif\n\
#ifndef {alias}_size\n#define {alias}_size {target}_size\n#endif\n\
#ifndef {alias}_pt\n#define {alias}_pt {target}_pt\n#endif\n\
#ifndef {alias}_pos\n#define {alias}_pos {target}_pos\n#endif\n\
#ifndef {alias}_map\n#define {alias}_map(id) {target}_map(id)\n#endif\n\
#ifndef {alias}_tex\n#define {alias}_tex(p) {target}_tex(p)\n#endif\n\
#ifndef {alias}_texOff\n#define {alias}_texOff(o) {target}_texOff(o)\n#endif\n\
#ifndef {alias}_gather\n#define {alias}_gather(p,c) {target}_gather(p,c)\n#endif\n\
#ifndef {alias}_off\n#define {alias}_off {target}_off\n#endif\n"
        ));
    }
}

fn multipass_compute_source(
    shader: &UserShader,
    pass: &Pass,
    raw_hook: &str,
    binds: &[PlannedBind],
    out_w: u32,
    out_h: u32,
    out_components: u8,
    tex_offset: (f32, f32),
) -> Result<String> {
    let layout = output_layout(out_components)?;
    // v664: fragment-style mpv passes previously used only 8x8 (64)
    // invocations per workgroup.  On modern desktop GPUs that leaves too much
    // dispatch/scheduler overhead for long Anime4K/FSRCNNX chains.  16x8 keeps
    // the group at a conservative 128 threads (warp/wave friendly on NVIDIA,
    // AMD and Intel) while explicit //!COMPUTE shaders retain their declared
    // workgroup exactly.
    let (local_w, local_h) = pass
        .compute
        .map(|spec| (spec.threads_w, spec.threads_h))
        .unwrap_or((16, 8));
    let mut prelude = format!(
        "#version 450\n\
layout(local_size_x = {local_w}, local_size_y = {local_h}, local_size_z = 1) in;\n\
layout({layout}, set = 0, binding = 0) uniform image2D out_image;\n\
const vec2 out_size = vec2({out_w}.0, {out_h}.0);\n\
const vec2 target_size = out_size;\n"
    );
    prelude.push_str(&parameter_prelude(shader));
    let mut next_binding = 1u32;
    for bind in binds {
        append_bind_prelude(&mut prelude, next_binding, bind, out_w, out_h);
        next_binding += if bind.storage { 1 } else { 2 };
    }
    append_hook_aliases(&mut prelude, raw_hook, binds);
    // Compute shaders have no implicit derivatives. mpv user shaders commonly
    // call texture() directly, so map the two-argument form to the base mip.
    prelude.push_str(
        "#define texture(tex, coord) textureLod(tex, coord, 0.0)\n\
#define textureOffset(tex, coord, off) textureLodOffset(tex, coord, 0.0, off)\n",
    );

    let bind_components = binds
        .iter()
        .map(|bind| (bind.shader_name.clone(), bind.components, false))
        .collect::<Vec<_>>();
    let mut body = super::glsl_engine::compat_shader_code(pass, &bind_components);
    body = body
        .lines()
        .filter(|line| !line.trim_start().starts_with("#error"))
        .collect::<Vec<_>>()
        .join("\n");
    body = body.replace(
        "gl_FragCoord.xy",
        "(vec2(gl_GlobalInvocationID.xy) + vec2(0.5))",
    );
    // Vulkan compute barriers already combine shared-memory visibility and
    // workgroup execution ordering. Naga does not expose the legacy GLSL
    // groupMemoryBarrier spelling used by RAVU, so use the equivalent barrier.
    body = body.replace("groupMemoryBarrier();", "barrier();");
    // A few desktop GLSL drivers accept scalar swizzles. Preserve the scalar
    // loop bounds explicitly for Naga instead of broad source rewriting.
    if body.contains("float start") && body.contains("float end") {
        body = body.replace("start.x", "start").replace("end.x", "end");
    }
    // KrigBilateral uses one sqr macro for both scalars and vectors. Route it
    // through overloads because dot(float,float) is invalid in strict GLSL.
    if body.contains("#define sqr(x)") && body.contains("dot(x,x)") {
        prelude.push_str(
            "float neo_sqr(float x) { return x*x; }\n\
float neo_sqr(vec2 x) { return dot(x,x); }\n\
float neo_sqr(vec3 x) { return dot(x,x); }\n\
float neo_sqr(vec4 x) { return dot(x,x); }\n",
        );
        body = body.replace("dot(x,x)", "neo_sqr(x)");
    }
    // Naga currently reverses nested local-array indexing during validation.
    // Flatten the 4x2 chroma gather cache while retaining the original row-
    // major mapping used by the CfL family.
    if body.contains("vec4 chroma_quads[4][2];") {
        body = body.replace("vec4 chroma_quads[4][2];", "vec4 chroma_quads[8];");
        body = body
            .replace("chroma_quads[i][0]", "chroma_quads[i*2]")
            .replace("chroma_quads[i][1]", "chroma_quads[i*2+1]");
        for row in 0..4 {
            body = body
                .replace(
                    &format!("chroma_quads[{row}][0]"),
                    &format!("chroma_quads[{}]", row * 2),
                )
                .replace(
                    &format!("chroma_quads[{row}][1]"),
                    &format!("chroma_quads[{}]", row * 2 + 1),
                );
        }
    }
    if body.contains("vec4 getMedian(") {
        body = body.replace(
            "\t}\n}\n\nvec4 hook()",
            "\t}\n\treturn v[0];\n}\n\nvec4 hook()",
        );
    }
    if body.contains("float mosaic_pix = 16.0;") {
        if let Some(close) = body.rfind('}') {
            body.insert_str(close, "\n    return color;\n");
        }
    }
    body = body.replace(
        " case 2: return p * vec2(-1, 1); \n\t }\n}",
        " case 2: return p * vec2(-1, 1); \n\t }\n\t return p;\n}",
    );
    body = body.replace(
        "case 2: return p * vec2(-1, 1);",
        "case 2: return p * vec2(-1, 1); default: return p;",
    );
    body = body.replace(
        "case 0: return val_swizz(GET_(off));",
        "case 0: return val_swizz(GET_(off)); default: return val_swizz(GET_(off));",
    );
    for bind in binds.iter().filter(|bind| !bind.storage && bind.depth == 1) {
        let n = &bind.shader_name;
        body = body.replace(
            &format!("textureGather({n}_raw,"),
            &format!("neo_{n}_gather("),
        );
        body = body.replace(
            &format!("textureGather({n}_tx,"),
            &format!("neo_{n}_gather("),
        );
        let gather_offsets = format!("textureGatherOffsets({n}_raw,");
        if body.contains(&gather_offsets) {
            prelude.push_str(&format!(
                "vec4 neo_{n}_gather_offsets(vec2 uv, ivec2 offs[4]) {{\n\
    ivec2 q = ivec2(floor(uv * {n}_size - vec2(0.5)));\n\
    return vec4(texelFetch({n}_tx,q+offs[0]+ivec2(0,1),0).x,\n\
                texelFetch({n}_tx,q+offs[1]+ivec2(1,1),0).x,\n\
                texelFetch({n}_tx,q+offs[2]+ivec2(1,0),0).x,\n\
                texelFetch({n}_tx,q+offs[3]+ivec2(0,0),0).x);\n}}\n"
            ));
            body = body.replace(&gather_offsets, &format!("neo_{n}_gather_offsets("));
        }
    }
    // GLSL writes multidimensional array constructors in type order, while
    // Naga's GLSL frontend currently expects the dimensions in declaration
    // indexing order. ACNet-family kernels use this exact 9x2 construct.
    if body.contains("const vec4 r[9][2] = vec4[9][2](") {
        body = body.replace(
            "const vec4 r[9][2] = vec4[9][2](",
            "const vec4 r[2][9] = vec4[2][9](",
        );
    }
    body = body.replace("vec4 g[3][8];", "vec4 g[8][3];");
    for channels in [8, 16] {
        body = body.replace(
            &format!("shared V4 inp[{channels}][isize.y][isize.x];"),
            &format!("shared V4 inp[isize.x][isize.y][{channels}];"),
        );
    }
    body = body.replace(
        "shared V4 inp[isize.y][isize.x];",
        "shared V4 inp[isize.x][isize.y];",
    );
    body = body.replace("GET_GUIDE(r).x", "GET_GUIDE(r)");
    body = body.replace("poi2.x - GET_GUIDE(r)", "poi2 - GET_GUIDE(r)");
    if body.contains("#define val_guide float") {
        body = body.replace("weight.x", "weight");
    }
    body = body.replace(
        "for (int i=2; float(i)<kernel_size; i++)",
        "for (int i=2; i<int(kernel_size); i++)",
    );
    body = body.replace(
        "for (int i=2; float(i)<KERNELSIZE; i++)",
        "for (int i=2; i<int(KERNELSIZE); i++)",
    );
    // Compute stages have no implicit fragment derivatives. Adaptive-sharpen
    // already owns a 5x5 neighborhood (`c`), so derive the same local edge
    // energy explicitly from that neighborhood instead of using fwidth().
    body = body.replace(
        "length(fwidth(val))",
        "(0.5*length((val)-c[12]) + 0.25*length(c[11]-c[13]) + 0.25*length(c[7]-c[17]))",
    );
    if body.contains("SMAALumaEdgeDetectionPS")
        || body.contains("SMAABlendingWeightCalculationPS")
        || body.contains("SMAANeighborhoodBlendingPS")
    {
        for sampler_name in ["colorTex", "edgesTex", "areaTex", "searchTex", "blendTex"] {
            body = body
                .replace(&format!(", sampler2D {sampler_name}"), "")
                .replace(&format!("sampler2D {sampler_name}, "), "");
        }
        body = body
            .replace("colorTex", "HOOKED_tx")
            .replace("edgesTex", "EDGES_TEX_tx")
            .replace("areaTex", "AREA_TEX_tx")
            .replace("searchTex", "SEARCH_TEX_tx")
            .replace("blendTex", "BLEND_TEX_tx")
            .replace(
                "SMAALumaEdgeDetectionPS(HOOKED_pos, HOOKED_raw)",
                "SMAALumaEdgeDetectionPS(HOOKED_pos)",
            )
            .replace(
                "SMAAColorEdgeDetectionPS(HOOKED_pos, HOOKED_raw)",
                "SMAAColorEdgeDetectionPS(HOOKED_pos)",
            )
            .replace("SMAASearchDiag1(EDGES_TEX_tx,", "SMAASearchDiag1(")
            .replace("SMAASearchDiag2(EDGES_TEX_tx,", "SMAASearchDiag2(")
            .replace("SMAAAreaDiag(AREA_TEX_tx,", "SMAAAreaDiag(")
            .replace(
                "SMAACalculateDiagWeights(EDGES_TEX_tx, AREA_TEX_tx,",
                "SMAACalculateDiagWeights(",
            )
            .replace("SMAASearchLength(SEARCH_TEX_tx,", "SMAASearchLength(")
            .replace(
                "SMAASearchXLeft(EDGES_TEX_tx, SEARCH_TEX_tx,",
                "SMAASearchXLeft(",
            )
            .replace(
                "SMAASearchXRight(EDGES_TEX_tx, SEARCH_TEX_tx,",
                "SMAASearchXRight(",
            )
            .replace(
                "SMAASearchYUp(EDGES_TEX_tx, SEARCH_TEX_tx,",
                "SMAASearchYUp(",
            )
            .replace(
                "SMAASearchYDown(EDGES_TEX_tx, SEARCH_TEX_tx,",
                "SMAASearchYDown(",
            )
            .replace("SMAAArea(AREA_TEX_tx,", "SMAAArea(")
            .replace(
                "SMAADetectHorizontalCornerPattern(EDGES_TEX_tx,",
                "SMAADetectHorizontalCornerPattern(",
            )
            .replace(
                "SMAADetectVerticalCornerPattern(EDGES_TEX_tx,",
                "SMAADetectVerticalCornerPattern(",
            )
            .replace(
                "SMAABlendingWeightCalculationPS(HOOKED_pos, EDGES_TEX_raw, AREA_TEX, SEARCH_TEX,",
                "SMAABlendingWeightCalculationPS(HOOKED_pos,",
            )
            .replace(
                "SMAANeighborhoodBlendingPS(HOOKED_pos, HOOKED_raw, BLEND_TEX_raw)",
                "SMAANeighborhoodBlendingPS(HOOKED_pos)",
            );
    }
    if body.contains("textureGatherOffset(HOOKED_raw") {
        body = body.replace(
            "textureGatherOffset(HOOKED_raw, p, gatherOffsets[i],",
            "neo_HOOKED_gather_off(p, gatherOffsets[i],",
        );
    }
    for divisor in [2, 3, 4] {
        body = body.replace(
            &format!("opos % {divisor}"),
            &format!("opos % ivec2({divisor})"),
        );
    }
    for name in ["coord", "hr_coord"] {
        body = body.replace(&format!("{name} % 2"), &format!("{name} % ivec2(2)"));
    }
    if pass.compute.is_some() {
        body = body
            .replace(
                "const ivec2 wg_size = ivec2(gl_WorkGroupSize);",
                &format!("const ivec2 wg_size = ivec2({local_w}, {local_h});"),
            )
            .replace(
                "const ivec2 isize = ivec2(gl_WorkGroupSize)",
                &format!("const ivec2 isize = ivec2({local_w}, {local_h})"),
            )
            .replace(
                "const uvec2 local_xy = gl_LocalInvocationID.xy;",
                "uvec2 local_xy = gl_LocalInvocationID.xy;",
            )
            .replace(
                "const ivec2 base = ivec2(gl_WorkGroupID) * wg_size;",
                "ivec2 base = ivec2(gl_WorkGroupID) * wg_size;",
            );
        if body.contains("#define V4 f16vec4") {
            let mut rewritten = String::with_capacity(body.len() + 320);
            rewritten.push_str(
                "f16vec4 neo_f16_max(f16vec4 a, f16vec4 b) { return f16vec4(max(vec4(a), vec4(b))); }\n\
f16vec4 neo_f16_min(f16vec4 a, f16vec4 b) { return f16vec4(min(vec4(a), vec4(b))); }\n\
f16vec4 neo_f16_clamp(f16vec4 v, f16vec4 lo, f16vec4 hi) { return f16vec4(clamp(vec4(v), vec4(lo), vec4(hi))); }\n",
            );
            for line in body.lines() {
                if line.contains("V4(") {
                    rewritten.push_str(
                        &line
                            .replace("max(", "neo_f16_max(")
                            .replace("min(", "neo_f16_min(")
                            .replace("clamp(", "neo_f16_clamp("),
                    );
                } else {
                    rewritten.push_str(line);
                }
                rewritten.push('\n');
            }
            body = rewritten;
        }
    }
    if pass.compute.is_some() && body.contains("f16vec") {
        // Naga requires the value passed to an rgba16f storage image to be a
        // regular vec4, while OpenGL accepts f16vec4 directly.
        prelude.push_str(
            "#define imageStore(image, coord, value) imageStore(image, coord, vec4(value))\n",
        );
    }
    if body.contains("tanh(") {
        prelude.push_str(
            "float neo_tanh(float x) { float e = exp(clamp(2.0*x,-80.0,80.0)); return (e-1.0)/(e+1.0); }\n\
vec4 neo_tanh(vec4 x) { vec4 e = exp(clamp(2.0*x,vec4(-80.0),vec4(80.0))); return (e-vec4(1.0))/(e+vec4(1.0)); }\n",
        );
        if body.contains("f16vec4") {
            prelude
                .push_str("f16vec4 neo_tanh(f16vec4 x) { return f16vec4(neo_tanh(vec4(x))); }\n");
        }
        body = body.replace("tanh(", "neo_tanh(");
    }
    if body.contains("matrixCompMult(") {
        prelude.push_str(
            "mat4x3 neo_matrix_comp_mult(mat4x3 a, mat4x3 b) { return mat4x3(a[0]*b[0],a[1]*b[1],a[2]*b[2],a[3]*b[3]); }\n",
        );
        body = body.replace("matrixCompMult(", "neo_matrix_comp_mult(");
    }
    if body.contains("outerProduct(") && body.contains("mat4x3") {
        prelude.push_str(
            "mat4x3 neo_outer_product(vec3 a, vec4 b) { return mat4x3(a*b.x,a*b.y,a*b.z,a*b.w); }\n",
        );
        body = body.replace("outerProduct(", "neo_outer_product(");
    }
    super::glsl_engine::append_mpv_transfer_helpers(&mut prelude, &body);

    // The OpenGL interpreter supplies these common aliases. Keep them cheap and
    // deterministic in the compiler milestone; runtime phase/frame values are
    // introduced when the executor is wired.
    if let Some(first) = binds.first() {
        prelude.push_str(&format!(
            "const vec2 input_size = {}_size;\nconst vec2 tex_offset = vec2({}, {});\n",
            first.shader_name,
            glsl_float_literal(tex_offset.0),
            glsl_float_literal(tex_offset.1),
        ));
    } else {
        prelude.push_str("const vec2 input_size = out_size;\nconst vec2 tex_offset = vec2(0.0);\n");
    }

    // Read the compute built-in once at the entry point. Naga otherwise
    // duplicates GlobalInvocationId while lowering large mpv shaders whose
    // *_pos macros are expanded through several helper functions.
    prelude.push_str("ivec2 neo_invocation_p;\n");

    let main_body = if pass.compute.is_some() {
        format!(
            "void main() {{\n    neo_invocation_p = ivec2(gl_GlobalInvocationID.xy);\n    ivec2 p = neo_invocation_p;\n    if (p.x >= {out_w} || p.y >= {out_h}) return;\n    hook();\n}}\n"
        )
    } else {
        format!(
            "void main() {{\n    neo_invocation_p = ivec2(gl_GlobalInvocationID.xy);\n    ivec2 p = neo_invocation_p;\n    if (p.x >= {out_w} || p.y >= {out_h}) return;\n    vec4 neo_result = hook();\n    imageStore(out_image, p, neo_result);\n}}\n"
        )
    };
    Ok(format!("{prelude}\n{body}\n\n{main_body}"))
}

fn compile_pass_to_spirv(source: &str) -> Result<Vec<u32>> {
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options::from(naga::ShaderStage::Compute);
    let module = frontend.parse(&options, source).map_err(|error| {
        let numbered = source
            .lines()
            .enumerate()
            .map(|(line, text)| format!("{:04}: {text}", line + 1))
            .collect::<Vec<_>>()
            .join("\n");
        anyhow!("Naga multi-pass GLSL parse failed: {error:?}\nGenerated GLSL:\n{numbered}")
    })?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| {
        let numbered = source
            .lines()
            .enumerate()
            .map(|(line, text)| format!("{:04}: {text}", line + 1))
            .collect::<Vec<_>>()
            .join("\n");
        anyhow!("Naga multi-pass validation failed: {error:?}\nGenerated GLSL:\n{numbered}")
    })?;
    let spv_options = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    naga::back::spv::write_vec(&module, &info, &spv_options, None)
        .map_err(|error| anyhow!("Naga multi-pass SPIR-V generation failed: {error:?}"))
}

fn initial_images(
    input_w: u32,
    input_h: u32,
    output_w: u32,
    output_h: u32,
) -> HashMap<String, PlannedImage> {
    let mut images = HashMap::new();
    let main = PlannedImage {
        name: "MAIN".to_string(),
        width: input_w,
        height: input_h,
        components: 4,
        offset_x: 0.0,
        offset_y: 0.0,
    };
    for name in ["MAIN", "RGB", "NATIVE", "MAINPRESUB"] {
        let mut image = main.clone();
        image.name = name.to_string();
        images.insert(name.to_string(), image);
    }
    images.insert(
        "OUTPUT".to_string(),
        PlannedImage {
            name: "OUTPUT".to_string(),
            width: output_w,
            height: output_h,
            components: 4,
            offset_x: 0.0,
            offset_y: 0.0,
        },
    );
    images.insert(
        "LUMA".to_string(),
        PlannedImage {
            name: "LUMA".to_string(),
            width: input_w,
            height: input_h,
            components: 1,
            offset_x: 0.0,
            offset_y: 0.0,
        },
    );
    images.insert(
        "CHROMA".to_string(),
        PlannedImage {
            name: "CHROMA".to_string(),
            width: input_w.div_ceil(2),
            height: input_h.div_ceil(2),
            components: 2,
            offset_x: 0.0,
            offset_y: 0.0,
        },
    );
    images
}

fn add_shader_textures(images: &mut HashMap<String, PlannedImage>, shader: &UserShader) {
    for texture in &shader.textures {
        images.insert(
            texture.name.clone(),
            PlannedImage {
                name: texture.name.clone(),
                width: texture.w.max(1) as u32,
                height: texture.h.max(1) as u32,
                components: texture.comps,
                offset_x: 0.0,
                offset_y: 0.0,
            },
        );
    }
}

fn image_sizes(images: &HashMap<String, PlannedImage>) -> Sizes {
    images
        .iter()
        .map(|(name, image)| (name.clone(), (image.width as f64, image.height as f64)))
        .collect()
}

fn param_values(shader: &UserShader) -> Params {
    shader
        .params
        .iter()
        .map(|p| (p.name.clone(), p.value as f64))
        .collect()
}

fn luma_extract_compute_source(width: u32, height: u32) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(r16f, set = 0, binding = 0) uniform image2D out_image;
layout(set = 0, binding = 1) uniform texture2D MAIN_texture;
layout(set = 0, binding = 2) uniform sampler MAIN_sampler;
#define MAIN_tx sampler2D(MAIN_texture, MAIN_sampler)
void main() {{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= {width} || p.y >= {height}) return;
    vec2 uv = (vec2(p) + vec2(0.5)) / vec2({width}.0, {height}.0);
    vec3 rgb = textureLod(MAIN_tx, uv, 0.0).rgb;
    imageStore(out_image, p, vec4(dot(rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0, 0.0, 1.0));
}}
"#
    )
}

fn luma_substitute_compute_source(width: u32, height: u32) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(rgba16f, set = 0, binding = 0) uniform image2D out_image;
layout(set = 0, binding = 1) uniform texture2D MAIN_texture;
layout(set = 0, binding = 2) uniform sampler MAIN_sampler;
layout(set = 0, binding = 3) uniform texture2D Y_texture;
layout(set = 0, binding = 4) uniform sampler Y_sampler;
#define MAIN_tx sampler2D(MAIN_texture, MAIN_sampler)
#define Y_tx sampler2D(Y_texture, Y_sampler)
void main() {{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= {width} || p.y >= {height}) return;
    vec2 uv = (vec2(p) + vec2(0.5)) / vec2({width}.0, {height}.0);
    vec3 rgb = textureLod(MAIN_tx, uv, 0.0).rgb;
    float y = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    float cb = (rgb.b - y) / (2.0 * (1.0 - 0.0722));
    float cr = (rgb.r - y) / (2.0 * (1.0 - 0.2126));
    float yp = textureLod(Y_tx, uv, 0.0).r;
    float r = yp + 2.0 * (1.0 - 0.2126) * cr;
    float b = yp + 2.0 * (1.0 - 0.0722) * cb;
    float g = (yp - 0.2126 * r - 0.0722 * b) / 0.7152;
    imageStore(out_image, p, vec4(r, g, b, 1.0));
}}
"#
    )
}

fn chroma_extract_compute_source(width: u32, height: u32) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(rg16f, set = 0, binding = 0) uniform image2D out_image;
layout(set = 0, binding = 1) uniform texture2D MAIN_texture;
layout(set = 0, binding = 2) uniform sampler MAIN_sampler;
#define MAIN_tx sampler2D(MAIN_texture, MAIN_sampler)
void main() {{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= {width} || p.y >= {height}) return;
    vec2 uv = (vec2(p) + vec2(0.5)) / vec2({width}.0, {height}.0);
    vec3 rgb = textureLod(MAIN_tx, uv, 0.0).rgb;
    float y = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    float cb = (rgb.b - y) / (2.0 * (1.0 - 0.0722)) + 0.5;
    float cr = (rgb.r - y) / (2.0 * (1.0 - 0.2126)) + 0.5;
    imageStore(out_image, p, vec4(cb, cr, 0.0, 1.0));
}}
"#
    )
}

fn yuv_to_rgb_compute_source(width: u32, height: u32) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(rgba16f, set = 0, binding = 0) uniform image2D out_image;
layout(set = 0, binding = 1) uniform texture2D Y_texture;
layout(set = 0, binding = 2) uniform sampler Y_sampler;
layout(set = 0, binding = 3) uniform texture2D C_texture;
layout(set = 0, binding = 4) uniform sampler C_sampler;
#define Y_tx sampler2D(Y_texture, Y_sampler)
#define C_tx sampler2D(C_texture, C_sampler)
void main() {{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= {width} || p.y >= {height}) return;
    vec2 uv = (vec2(p) + vec2(0.5)) / vec2({width}.0, {height}.0);
    float yp = textureLod(Y_tx, uv, 0.0).r;
    vec2 cc = textureLod(C_tx, uv, 0.0).rg - vec2(0.5);
    float r = yp + 2.0 * (1.0 - 0.2126) * cc.y;
    float b = yp + 2.0 * (1.0 - 0.0722) * cc.x;
    float g = (yp - 0.2126 * r - 0.0722 * b) / 0.7152;
    imageStore(out_image, p, vec4(clamp(vec3(r, g, b), 0.0, 1.0), 1.0));
}}
"#
    )
}

fn resolve_bind(
    bind_name: &str,
    hooked: &str,
    images: &HashMap<String, PlannedImage>,
    shader: &UserShader,
) -> Result<PlannedBind> {
    let resource_name = if bind_name.eq_ignore_ascii_case("HOOKED") {
        hooked.to_string()
    } else {
        let canonical = canonical_tex_name(bind_name);
        if images.contains_key(bind_name) {
            bind_name.to_string()
        } else if images.contains_key(canonical) {
            canonical.to_string()
        } else {
            return Err(anyhow!("BIND {bind_name} is not available"));
        }
    };
    let image = images
        .get(&resource_name)
        .ok_or_else(|| anyhow!("BIND resource {resource_name} disappeared"))?;
    let embedded = shader.textures.iter().find(|t| t.name == resource_name);
    Ok(PlannedBind {
        shader_name: bind_name.to_string(),
        resource_name,
        width: image.width,
        height: image.height,
        components: image.components,
        depth: embedded.map(|t| t.d.max(1) as u32).unwrap_or(1),
        storage: embedded.map(|t| t.storage).unwrap_or(false),
    })
}

pub fn build_multipass_plan(
    shader: &UserShader,
    input_w: u32,
    input_h: u32,
    output_w: u32,
    output_h: u32,
) -> Result<MultiPassPlan> {
    if shader.passes.is_empty() {
        return Err(anyhow!("shader has no passes"));
    }
    let input_w = input_w.max(1);
    let input_h = input_h.max(1);
    let output_w = output_w.max(1);
    let output_h = output_h.max(1);
    let mut images = initial_images(input_w, input_h, output_w, output_h);
    add_shader_textures(&mut images, shader);
    let pmap = param_values(shader);
    let mut planned = Vec::new();
    let mut total_words = 0usize;
    let mut final_luma_resource = None;
    let mut final_chroma_resource = None;
    let has_luma_hook = shader
        .passes
        .iter()
        .flat_map(|pass| pass.hooks.iter())
        .any(|hook| canonical_tex_name(hook) == "LUMA");
    let luma_only = has_luma_hook && !shader.uses_chroma;

    if luma_only {
        let spirv = compile_pass_to_spirv(&luma_extract_compute_source(input_w, input_h))
            .context("BT.709 LUMA extraction compile failed")?;
        total_words += spirv.len();
        planned.push(PlannedPass {
            source_index: usize::MAX,
            desc: "[Neo] BT.709 LUMA extraction".to_string(),
            hooked: "MAIN".to_string(),
            save: "LUMA".to_string(),
            width: input_w,
            height: input_h,
            components: 1,
            dispatch_block_w: 16,
            dispatch_block_h: 8,
            binds: vec![PlannedBind {
                shader_name: "MAIN".to_string(),
                resource_name: "MAIN".to_string(),
                width: input_w,
                height: input_h,
                components: 4,
                depth: 1,
                storage: false,
            }],
            spirv,
        });
        final_luma_resource = Some("LUMA".to_string());
    } else if shader.uses_chroma {
        let luma_spirv = compile_pass_to_spirv(&luma_extract_compute_source(input_w, input_h))
            .context("BT.709 YUV LUMA extraction compile failed")?;
        total_words += luma_spirv.len();
        planned.push(PlannedPass {
            source_index: usize::MAX,
            desc: "[Neo] BT.709 YUV LUMA extraction".to_string(),
            hooked: "MAIN".to_string(),
            save: "LUMA".to_string(),
            width: input_w,
            height: input_h,
            components: 1,
            dispatch_block_w: 16,
            dispatch_block_h: 8,
            binds: vec![PlannedBind {
                shader_name: "MAIN".to_string(),
                resource_name: "MAIN".to_string(),
                width: input_w,
                height: input_h,
                components: 4,
                depth: 1,
                storage: false,
            }],
            spirv: luma_spirv,
        });
        let cw = input_w.div_ceil(2);
        let ch = input_h.div_ceil(2);
        let chroma_spirv = compile_pass_to_spirv(&chroma_extract_compute_source(cw, ch))
            .context("BT.709 CHROMA extraction compile failed")?;
        total_words += chroma_spirv.len();
        planned.push(PlannedPass {
            source_index: usize::MAX,
            desc: "[Neo] BT.709 4:2:0 CHROMA extraction".to_string(),
            hooked: "MAIN".to_string(),
            save: "CHROMA".to_string(),
            width: cw,
            height: ch,
            components: 2,
            dispatch_block_w: 16,
            dispatch_block_h: 8,
            binds: vec![PlannedBind {
                shader_name: "MAIN".to_string(),
                resource_name: "MAIN".to_string(),
                width: input_w,
                height: input_h,
                components: 4,
                depth: 1,
                storage: false,
            }],
            spirv: chroma_spirv,
        });
        final_luma_resource = Some("LUMA".to_string());
        final_chroma_resource = Some("CHROMA".to_string());
    }

    for (index, pass) in shader.passes.iter().enumerate() {
        let hooked = hooked_texture_key(pass, &images);
        let Some(hooked_image) = images.get(&hooked).cloned() else {
            // Match the OpenGL interpreter: a hook that does not exist for this
            // invocation is skipped rather than killing the whole shader.
            continue;
        };

        // HOOKED is a transient alias for expression evaluation and BIND.
        images.insert(
            "HOOKED".to_string(),
            PlannedImage {
                name: "HOOKED".to_string(),
                ..hooked_image.clone()
            },
        );
        let sizes = image_sizes(&images);
        if let Some(when) = &pass.when {
            let value = eval_rpn_p(when, &sizes, &pmap).unwrap_or(0.0);
            if value <= 0.0 {
                continue;
            }
        }
        let ow = match &pass.width {
            Some(expr) => {
                let Some(value) = eval_rpn_p(expr, &sizes, &pmap) else {
                    continue;
                };
                value as i32
            }
            None => hooked_image.width as i32,
        }
        .max(1) as u32;
        let oh = match &pass.height {
            Some(expr) => {
                let Some(value) = eval_rpn_p(expr, &sizes, &pmap) else {
                    continue;
                };
                value as i32
            }
            None => hooked_image.height as i32,
        }
        .max(1) as u32;
        let save = pass.save.clone().unwrap_or_else(|| hooked.clone());
        let out_components = match canonical_tex_name(&save) {
            "MAIN" | "RGB" | "OUTPUT" => 4,
            _ => pass.components,
        };
        output_layout(out_components)?;

        let mut binds = Vec::with_capacity(pass.binds.len().max(1));
        let mut seen = HashSet::new();
        for bind in &pass.binds {
            if seen.insert(bind.clone()) {
                binds.push(resolve_bind(bind, &hooked, &images, shader)?);
            }
        }
        // Some generated mpv shaders rely on the hooked texture aliases even
        // when they did not spell out BIND HOOKED. Ensure the raw hook can be
        // materialized without changing the user's declared bind order.
        if binds.is_empty() {
            binds.push(PlannedBind {
                shader_name: "HOOKED".to_string(),
                resource_name: hooked.clone(),
                width: hooked_image.width,
                height: hooked_image.height,
                components: hooked_image.components,
                depth: 1,
                storage: false,
            });
        }

        let raw_hook = raw_hook_name(pass, &hooked);
        let source = multipass_compute_source(
            shader,
            pass,
            &raw_hook,
            &binds,
            ow,
            oh,
            out_components,
            (hooked_image.offset_x, hooked_image.offset_y),
        )
        .map_err(|e| anyhow!("pass {index} `{}` wrapper failed: {e:#}", pass.desc))?;
        let spirv = compile_pass_to_spirv(&source)
            .map_err(|e| anyhow!("pass {index} `{}` compile failed: {e:#}", pass.desc))?;
        total_words += spirv.len();
        planned.push(PlannedPass {
            source_index: index,
            desc: pass.desc.clone(),
            hooked: hooked.clone(),
            save: save.clone(),
            width: ow,
            height: oh,
            components: out_components,
            dispatch_block_w: pass.compute.map(|spec| spec.block_w).unwrap_or(16),
            dispatch_block_h: pass.compute.map(|spec| spec.block_h).unwrap_or(8),
            binds,
            spirv,
        });
        let scale_x = ow as f32 / hooked_image.width.max(1) as f32;
        let scale_y = oh as f32 / hooked_image.height.max(1) as f32;
        let mut out_offset = (
            hooked_image.offset_x * scale_x,
            hooked_image.offset_y * scale_y,
        );
        if save == hooked {
            match pass.offset {
                PassOffset::Pixels(x, y) => {
                    out_offset.0 += x;
                    out_offset.1 += y;
                }
                PassOffset::Align => out_offset = (0.0, 0.0),
                PassOffset::None => {}
            }
        }
        images.insert(
            save.clone(),
            PlannedImage {
                name: save.clone(),
                width: ow,
                height: oh,
                components: out_components,
                offset_x: out_offset.0,
                offset_y: out_offset.1,
            },
        );
        if raw_hook == "LUMA" || hooked == "LUMA" {
            final_luma_resource = Some(save.clone());
        } else if raw_hook == "CHROMA" || hooked == "CHROMA" {
            final_chroma_resource = Some(save.clone());
        }
    }

    if planned.is_empty() {
        return Err(anyhow!(
            "no active passes for the supplied input/output size"
        ));
    }

    if luma_only {
        let y_name = final_luma_resource
            .clone()
            .unwrap_or_else(|| "LUMA".to_string());
        let y_image = images
            .get(&y_name)
            .cloned()
            .ok_or_else(|| anyhow!("final LUMA resource {y_name} is unavailable"))?;
        let spirv = compile_pass_to_spirv(&luma_substitute_compute_source(
            y_image.width,
            y_image.height,
        ))
        .context("BT.709 LUMA substitution compile failed")?;
        total_words += spirv.len();
        planned.push(PlannedPass {
            source_index: usize::MAX,
            desc: "[Neo] BT.709 LUMA substitution".to_string(),
            hooked: "LUMA".to_string(),
            save: "NEO_LUMA_RGB".to_string(),
            width: y_image.width,
            height: y_image.height,
            components: 4,
            dispatch_block_w: 16,
            dispatch_block_h: 8,
            binds: vec![
                PlannedBind {
                    shader_name: "MAIN".to_string(),
                    resource_name: "MAIN".to_string(),
                    width: input_w,
                    height: input_h,
                    components: 4,
                    depth: 1,
                    storage: false,
                },
                PlannedBind {
                    shader_name: "Y".to_string(),
                    resource_name: y_name,
                    width: y_image.width,
                    height: y_image.height,
                    components: 1,
                    depth: 1,
                    storage: false,
                },
            ],
            spirv,
        });
        images.insert(
            "NEO_LUMA_RGB".to_string(),
            PlannedImage {
                name: "NEO_LUMA_RGB".to_string(),
                width: y_image.width,
                height: y_image.height,
                components: 4,
                offset_x: y_image.offset_x,
                offset_y: y_image.offset_y,
            },
        );
    } else if shader.uses_chroma {
        let y_name = final_luma_resource
            .clone()
            .unwrap_or_else(|| "LUMA".to_string());
        let c_name = final_chroma_resource
            .clone()
            .unwrap_or_else(|| "CHROMA".to_string());
        let y_image = images
            .get(&y_name)
            .cloned()
            .ok_or_else(|| anyhow!("final Y resource {y_name} is unavailable"))?;
        let c_image = images
            .get(&c_name)
            .cloned()
            .ok_or_else(|| anyhow!("final CHROMA resource {c_name} is unavailable"))?;
        let spirv =
            compile_pass_to_spirv(&yuv_to_rgb_compute_source(y_image.width, y_image.height))
                .context("BT.709 YUV to RGB compile failed")?;
        total_words += spirv.len();
        planned.push(PlannedPass {
            source_index: usize::MAX,
            desc: "[Neo] BT.709 YUV to RGB".to_string(),
            hooked: "LUMA".to_string(),
            save: "NEO_YUV_RGB".to_string(),
            width: y_image.width,
            height: y_image.height,
            components: 4,
            dispatch_block_w: 16,
            dispatch_block_h: 8,
            binds: vec![
                PlannedBind {
                    shader_name: "Y".to_string(),
                    resource_name: y_name,
                    width: y_image.width,
                    height: y_image.height,
                    components: 1,
                    depth: 1,
                    storage: false,
                },
                PlannedBind {
                    shader_name: "C".to_string(),
                    resource_name: c_name,
                    width: c_image.width,
                    height: c_image.height,
                    components: 2,
                    depth: 1,
                    storage: false,
                },
            ],
            spirv,
        });
        images.insert(
            "NEO_YUV_RGB".to_string(),
            PlannedImage {
                name: "NEO_YUV_RGB".to_string(),
                width: y_image.width,
                height: y_image.height,
                components: 4,
                offset_x: y_image.offset_x,
                offset_y: y_image.offset_y,
            },
        );
    }

    let final_resource = if let Some(last) = planned.last() {
        last.save.clone()
    } else {
        return Err(anyhow!("no final resource"));
    };
    let final_image = images
        .get(&final_resource)
        .or_else(|| planned.last().and_then(|p| images.get(&p.save)))
        .ok_or_else(|| anyhow!("final resource {final_resource} is unavailable"))?;

    Ok(MultiPassPlan {
        shader_name: shader.name(),
        passes: planned,
        final_resource,
        final_width: final_image.width,
        final_height: final_image.height,
        final_components: final_image.components,
        total_spirv_words: total_words,
        textures: shader.textures.clone(),
        final_luma_resource,
        final_chroma_resource,
    })
}

/// Build one Vulkan plan from a sequence of ordinary mpv shaders while
/// preserving each shader's standalone boundary semantics.  This differs from
/// simply concatenating raw passes: LUMA-only shaders need their RGB->LUMA and
/// LUMA->RGB compatibility passes to finish before the following shader sees
/// the result as its new MAIN image.  Building each stage independently first
/// gives us that exact boundary, then resource names are namespaced and the
/// next stage's initial MAIN/RGB aliases are rebound to the previous stage's
/// final Vulkan image.  No intermediate CPU readback or U8 quantization is
/// required.
fn build_multipass_sequence_plan(
    shaders: &[&UserShader],
    input_w: u32,
    input_h: u32,
    output_w: u32,
    output_h: u32,
) -> Result<MultiPassPlan> {
    if shaders.len() < 2 {
        return Err(anyhow!(
            "Vulkan shader sequence requires at least two stages"
        ));
    }

    let mut merged_passes = Vec::new();
    let mut merged_textures = Vec::new();
    let mut total_spirv_words = 0usize;
    let mut current_w = input_w.max(1);
    let mut current_h = input_h.max(1);
    let mut current_components = 4u8;
    let mut previous_final: Option<String> = None;
    let mut stage_names = Vec::with_capacity(shaders.len());

    for (stage_index, shader) in shaders.iter().enumerate() {
        let plan = build_multipass_plan(
            shader,
            current_w,
            current_h,
            output_w.max(1),
            output_h.max(1),
        )
        .with_context(|| {
            format!(
                "Vulkan sequence stage {} ({}) plan failed",
                stage_index,
                shader.name()
            )
        })?;
        stage_names.push(shader.name());
        total_spirv_words = total_spirv_words.saturating_add(plan.total_spirv_words);

        let prefix = format!("__neo_seq{stage_index}_");
        let mut local_resources: HashMap<String, String> = HashMap::new();

        // Embedded resources are descriptor-bound by slot after SPIR-V
        // compilation, so only the runtime resource keys need namespacing.
        for mut texture in plan.textures.iter().cloned() {
            let original = texture.name.clone();
            let renamed = format!("{prefix}{original}");
            texture.name = renamed.clone();
            local_resources.insert(original, renamed);
            merged_textures.push(texture);
        }

        let stage_input = previous_final.clone();
        for mut pass in plan.passes.iter().cloned() {
            for bind in &mut pass.binds {
                if let Some(mapped) = local_resources.get(&bind.resource_name) {
                    bind.resource_name = mapped.clone();
                } else if let Some(input_resource) = stage_input.as_ref() {
                    // A fresh standalone runtime maps all of these aliases to
                    // its input image.  In the sequence, stage N's input is
                    // stage N-1's final Vulkan image instead.
                    if matches!(
                        bind.resource_name.as_str(),
                        "MAIN" | "RGB" | "NATIVE" | "MAINPRESUB" | "OUTPUT"
                    ) {
                        bind.resource_name = input_resource.clone();
                    }
                }
            }

            let original_save = pass.save.clone();
            let renamed_save = format!("{prefix}{original_save}");
            pass.save = renamed_save.clone();
            // Update after resolving the pass inputs so in-place resources
            // such as MAIN/LUMA/MODEL21 retain the same overwrite semantics as
            // the standalone planner.
            local_resources.insert(original_save, renamed_save);
            merged_passes.push(pass);
        }

        let stage_final = local_resources
            .get(&plan.final_resource)
            .cloned()
            .or_else(|| {
                stage_input.clone().filter(|_| {
                    matches!(
                        plan.final_resource.as_str(),
                        "MAIN" | "RGB" | "NATIVE" | "MAINPRESUB" | "OUTPUT"
                    )
                })
            })
            .or_else(|| {
                (stage_index == 0
                    && matches!(
                        plan.final_resource.as_str(),
                        "MAIN" | "RGB" | "NATIVE" | "MAINPRESUB" | "OUTPUT"
                    ))
                .then(|| plan.final_resource.clone())
            })
            .ok_or_else(|| {
                anyhow!(
                    "Vulkan sequence stage {} ({}) final resource {} could not be rebound",
                    stage_index,
                    shader.name(),
                    plan.final_resource
                )
            })?;

        previous_final = Some(stage_final);
        current_w = plan.final_width.max(1);
        current_h = plan.final_height.max(1);
        current_components = plan.final_components;
    }

    let final_resource = previous_final.ok_or_else(|| anyhow!("Vulkan sequence has no output"))?;
    Ok(MultiPassPlan {
        shader_name: format!("VulkanSequence[{}]", stage_names.join("+")),
        passes: merged_passes,
        final_resource,
        final_width: current_w,
        final_height: current_h,
        final_components: current_components,
        total_spirv_words,
        textures: merged_textures,
        final_luma_resource: None,
        final_chroma_resource: None,
    })
}

pub fn analyze_multipass_compatibility(
    shader: &UserShader,
    input_w: u32,
    input_h: u32,
    output_w: u32,
    output_h: u32,
) -> MultiPassCompatibility {
    match build_multipass_plan(shader, input_w, input_h, output_w, output_h) {
        Ok(plan) => MultiPassCompatibility {
            compatible: true,
            reason: "parsed-multipass-and-all-active-passes-naga-compatible".to_string(),
            active_passes: plan.passes.len(),
            total_passes: shader.passes.len(),
            spirv_words: plan.total_spirv_words,
            final_width: plan.final_width,
            final_height: plan.final_height,
        },
        Err(error) => MultiPassCompatibility {
            compatible: false,
            reason: error.to_string(),
            active_passes: 0,
            total_passes: shader.passes.len(),
            spirv_words: 0,
            final_width: 0,
            final_height: 0,
        },
    }
}

pub fn compatibility_scan_requested() -> bool {
    std::env::var("NEO_VULKAN_MULTIPASS_COMPAT_SCAN")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn collect_glsl_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_glsl_files(&path, out);
        } else if path
            .extension()
            .and_then(|v| v.to_str())
            .map(|v| v.eq_ignore_ascii_case("glsl"))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
}

fn clean_diag_text(value: &str) -> String {
    value
        .replace('\r', " ")
        .replace('\n', " ")
        .replace('\'', "_")
}

pub fn scan_bundled_compatibility(app_dir: &Path) {
    let shader_root = app_dir.join("shaders");
    let mut files = Vec::new();
    collect_glsl_files(&shader_root, &mut files);
    files.sort();

    // 448x252 -> 896x504 intentionally activates Anime4K upscale WHEN clauses.
    const INPUT_W: u32 = 448;
    const INPUT_H: u32 = 252;
    const OUTPUT_W: u32 = 896;
    const OUTPUT_H: u32 = 504;

    let mut compatible = 0usize;
    let mut rejected = 0usize;
    let mut load_failed = 0usize;
    let mut anime4k_total = 0usize;
    let mut anime4k_compatible = 0usize;
    let mut fsrcnnx_total = 0usize;
    let mut fsrcnnx_compatible = 0usize;

    for path in &files {
        let relative = path
            .strip_prefix(app_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let is_anime4k = relative.contains("/Anime4K/");
        let is_fsrcnnx = relative.contains("/FSRCNNX/") || relative.contains("/Denoise/FSRCNNX_");
        if is_anime4k {
            anime4k_total += 1;
        }
        if is_fsrcnnx {
            fsrcnnx_total += 1;
        }

        match UserShader::load(path.to_string_lossy().as_ref()) {
            Ok(shader) => {
                let result =
                    analyze_multipass_compatibility(&shader, INPUT_W, INPUT_H, OUTPUT_W, OUTPUT_H);
                if result.compatible {
                    compatible += 1;
                    if is_anime4k {
                        anime4k_compatible += 1;
                    }
                    if is_fsrcnnx {
                        fsrcnnx_compatible += 1;
                    }
                    let line = format!(
                        "vulkan-multipass-scan: result=compatible shader='{}' passes={}/{} spirv_words={} final={}x{} execution=anime4k-cnn-test-only production=unchanged",
                        clean_diag_text(&relative),
                        result.active_passes,
                        result.total_passes,
                        result.spirv_words,
                        result.final_width,
                        result.final_height
                    );
                    log::info!("{line}");
                    super::vulkan_gpu::record_probe_result(&line);
                } else {
                    rejected += 1;
                    let line = format!(
                        "vulkan-multipass-scan: result=rejected shader='{}' passes={} reason='{}'",
                        clean_diag_text(&relative),
                        result.total_passes,
                        clean_diag_text(&result.reason)
                    );
                    log::debug!("{line}");
                    super::vulkan_gpu::record_probe_result(&line);
                }
            }
            Err(error) => {
                load_failed += 1;
                let line = format!(
                    "vulkan-multipass-scan: result=load-failed shader='{}' reason='{}'",
                    clean_diag_text(&relative),
                    clean_diag_text(&error.to_string())
                );
                log::warn!("{line}");
                super::vulkan_gpu::record_probe_result(&line);
            }
        }
    }

    let summary = format!(
        "vulkan-multipass-scan: result=summary total={} compatible={} rejected={} load_failed={} anime4k={}/{} fsrcnnx={}/{} input={}x{} output={}x{} parser=neo-existing-mpv backend=vulkan-pass-compiler execution=anime4k-cnn-test-only vulkan_init=false display=OpenGL-unchanged",
        files.len(),
        compatible,
        rejected,
        load_failed,
        anime4k_compatible,
        anime4k_total,
        fsrcnnx_compatible,
        fsrcnnx_total,
        INPUT_W,
        INPUT_H,
        OUTPUT_W,
        OUTPUT_H
    );
    log::info!("{summary}");
    super::vulkan_gpu::record_probe_result(&summary);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_manifest_shader(relative: &str) -> UserShader {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
        UserShader::load(path.to_string_lossy().as_ref())
            .unwrap_or_else(|error| panic!("failed to load {}: {error}", path.display()))
    }

    #[test]
    fn simple_saved_intermediate_builds_two_pass_plan() {
        let src = r#"
//!HOOK MAIN
//!BIND MAIN
//!SAVE X
//!WIDTH MAIN.w
//!HEIGHT MAIN.h
//!COMPONENTS 4
vec4 hook(){ return MAIN_tex(MAIN_pos); }
//!HOOK MAIN
//!BIND MAIN
//!BIND X
//!SAVE MAIN
//!WIDTH X.w 2 *
//!HEIGHT X.h 2 *
//!COMPONENTS 4
vec4 hook(){ return MAIN_tex(MAIN_pos) + X_tex(X_pos); }
"#;
        let shader = UserShader::parse("two.glsl", src);
        let plan = build_multipass_plan(&shader, 320, 180, 640, 360).unwrap();
        assert_eq!(plan.passes.len(), 2);
        assert_eq!(plan.final_resource, "MAIN");
        assert_eq!((plan.final_width, plan.final_height), (640, 360));
    }

    #[test]
    fn priority_shader_set_compiles_for_vulkan() {
        let shaders = [
            "shaders/AA/AXAA.glsl",
            "shaders/AA/Intel_CMAA2_lite.glsl",
            "shaders/Deinterlace/deint_swa.glsl",
            "shaders/Anime4K/Anime4K_Restore_CNN_S.glsl",
        ];
        for relative in shaders {
            let shader = load_manifest_shader(relative);
            let result = analyze_multipass_compatibility(&shader, 448, 252, 896, 504);
            assert!(result.compatible, "{relative}: {}", result.reason);
        }
    }

    fn test_discrete_vulkan_luid() -> Result<u64> {
        let entry = unsafe { Entry::load() }.context("Vulkan loader unavailable")?;
        let app_name = CString::new("Neo Vulkan multipass test")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(vk::API_VERSION_1_1);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
        }
        .context("test vkCreateInstance failed")?;
        let result = (|| {
            let devices = unsafe { instance.enumerate_physical_devices() }
                .context("test vkEnumeratePhysicalDevices failed")?;
            let mut fallback = None;
            for physical_device in devices {
                let mut id = vk::PhysicalDeviceIDProperties::default();
                let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
                unsafe {
                    instance.get_physical_device_properties2(physical_device, &mut properties2)
                };
                let device_type = properties2.properties.device_type;
                let _ = properties2;
                if id.device_luid_valid != vk::TRUE {
                    continue;
                }
                let luid = luid_from_vk(&id.device_luid);
                fallback.get_or_insert(luid);
                if device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    return Ok(luid);
                }
            }
            fallback.ok_or_else(|| anyhow!("no Vulkan device exposes a DXGI LUID"))
        })();
        unsafe { instance.destroy_instance(None) };
        result
    }

    #[test]
    #[ignore = "real selected-GPU Vulkan execution test"]
    fn real_gpu_identity_multipass_preserves_pixels_on_initial_and_steady_paths() {
        let src = r#"
//!HOOK MAIN
//!BIND MAIN
//!SAVE MAIN
//!WIDTH MAIN.w
//!HEIGHT MAIN.h
//!COMPONENTS 4
vec4 hook() { return MAIN_tex(MAIN_pos); }
"#;
        let shader = UserShader::parse("vulkan_identity.glsl", src);
        let plan = build_multipass_plan(&shader, 8, 8, 8, 8).unwrap();
        let key = RuntimeKey {
            luid: test_discrete_vulkan_luid().unwrap(),
            input_width: 8,
            input_height: 8,
            output_ref_width: 8,
            output_ref_height: 8,
            shader_hash: shader_runtime_hash(&shader),
        };
        let mut input = Vec::with_capacity(8 * 8 * 4);
        for y in 0..8u8 {
            for x in 0..8u8 {
                input.extend_from_slice(&[
                    x.wrapping_mul(29).wrapping_add(y),
                    y.wrapping_mul(31).wrapping_add(x.wrapping_mul(3)),
                    x.wrapping_mul(11).wrapping_add(y.wrapping_mul(17)),
                    255,
                ]);
            }
        }
        let mut runtime = VulkanMultiPassRuntime::create(key, plan).unwrap();
        let first = runtime.process(&input).unwrap();
        let steady = runtime.process(&input).unwrap();
        assert_eq!(
            first.output_rgba8, input,
            "initial command path changed pixels"
        );
        assert_eq!(
            steady.output_rgba8, input,
            "steady command path changed pixels"
        );
    }

    #[test]
    #[ignore = "real selected-GPU Vulkan execution test"]
    fn real_gpu_two_pass_intermediate_and_barrier_preserve_pixels() {
        let src = r#"
//!HOOK MAIN
//!BIND MAIN
//!SAVE SWAPPED
//!WIDTH MAIN.w
//!HEIGHT MAIN.h
//!COMPONENTS 4
vec4 hook() { return MAIN_tex(MAIN_pos).bgra; }
//!HOOK MAIN
//!BIND SWAPPED
//!SAVE MAIN
//!WIDTH SWAPPED.w
//!HEIGHT SWAPPED.h
//!COMPONENTS 4
vec4 hook() { return SWAPPED_tex(SWAPPED_pos).bgra; }
"#;
        let shader = UserShader::parse("vulkan_two_pass.glsl", src);
        let plan = build_multipass_plan(&shader, 8, 8, 8, 8).unwrap();
        assert_eq!(plan.passes.len(), 2);
        let key = RuntimeKey {
            luid: test_discrete_vulkan_luid().unwrap(),
            input_width: 8,
            input_height: 8,
            output_ref_width: 8,
            output_ref_height: 8,
            shader_hash: shader_runtime_hash(&shader),
        };
        let mut input = Vec::with_capacity(8 * 8 * 4);
        for y in 0..8u8 {
            for x in 0..8u8 {
                input.extend_from_slice(&[
                    x.wrapping_mul(23).wrapping_add(y.wrapping_mul(5)),
                    y.wrapping_mul(19).wrapping_add(x),
                    x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13)),
                    255,
                ]);
            }
        }
        let mut runtime = VulkanMultiPassRuntime::create(key, plan).unwrap();
        let first = runtime.process(&input).unwrap();
        let steady = runtime.process(&input).unwrap();
        assert_eq!(
            first.output_rgba8, input,
            "initial two-pass path changed pixels"
        );
        assert_eq!(
            steady.output_rgba8, input,
            "steady two-pass path changed pixels"
        );
    }

    #[test]
    #[ignore = "real selected-GPU Vulkan execution test"]
    fn real_gpu_luma_identity_preserves_rgb_with_rounding_tolerance() {
        let src = r#"
//!HOOK LUMA
//!BIND LUMA
//!SAVE LUMA
//!WIDTH LUMA.w
//!HEIGHT LUMA.h
//!COMPONENTS 1
vec4 hook() { return LUMA_tex(LUMA_pos); }
"#;
        let shader = UserShader::parse("vulkan_luma_identity.glsl", src);
        let plan = build_multipass_plan(&shader, 16, 12, 16, 12).unwrap();
        assert_eq!(plan.passes.len(), 3, "extract + shader + substitute");
        let key = RuntimeKey {
            luid: test_discrete_vulkan_luid().unwrap(),
            input_width: 16,
            input_height: 12,
            output_ref_width: 16,
            output_ref_height: 12,
            shader_hash: shader_runtime_hash(&shader),
        };
        let mut input = Vec::with_capacity(16 * 12 * 4);
        for y in 0..12u8 {
            for x in 0..16u8 {
                input.extend_from_slice(&[
                    24u8.wrapping_add(x.wrapping_mul(11)),
                    17u8.wrapping_add(y.wrapping_mul(17)),
                    31u8.wrapping_add(x.wrapping_mul(5))
                        .wrapping_add(y.wrapping_mul(7)),
                    255,
                ]);
            }
        }
        let mut runtime = VulkanMultiPassRuntime::create(key, plan).unwrap();
        for output in [
            runtime.process(&input).unwrap(),
            runtime.process(&input).unwrap(),
        ] {
            let max_error = output
                .output_rgba8
                .iter()
                .zip(&input)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap_or(0);
            assert!(
                max_error <= 1,
                "LUMA identity RGB max error was {max_error}"
            );
        }
    }

    #[test]
    #[ignore = "real selected-GPU Vulkan execution test"]
    fn real_gpu_fsrcnnx_luma_chain_is_stable_between_initial_and_steady_paths() {
        let shader = load_manifest_shader("shaders/FSRCNNX/FSRCNNX_x2_8_0_4_1.glsl");
        assert!(production_shader_admitted(&shader));
        let plan = build_multipass_plan(&shader, 16, 12, 32, 24).unwrap();
        assert!(
            plan.passes.len() > 3,
            "expected a real FSRCNNX multi-pass graph"
        );
        let key = RuntimeKey {
            luid: test_discrete_vulkan_luid().unwrap(),
            input_width: 16,
            input_height: 12,
            output_ref_width: 32,
            output_ref_height: 24,
            shader_hash: shader_runtime_hash(&shader),
        };
        let mut input = Vec::with_capacity(16 * 12 * 4);
        for y in 0..12u8 {
            for x in 0..16u8 {
                input.extend_from_slice(&[
                    32u8.wrapping_add(x.wrapping_mul(9)),
                    24u8.wrapping_add(y.wrapping_mul(13)),
                    40u8.wrapping_add(x.wrapping_mul(3))
                        .wrapping_add(y.wrapping_mul(5)),
                    255,
                ]);
            }
        }
        let mut runtime = VulkanMultiPassRuntime::create(key, plan).unwrap();
        let first = runtime.process(&input).unwrap();
        let steady = runtime.process(&input).unwrap();
        assert_eq!((first.output_width, first.output_height), (32, 24));
        assert_eq!(first.output_rgba8, steady.output_rgba8);
        assert!(first.output_rgba8.iter().any(|&value| value != 0));
        assert!(
            first
                .output_rgba8
                .chunks_exact(4)
                .all(|pixel| pixel[3] == 255)
        );
    }

    #[test]
    #[ignore = "real selected-GPU Vulkan execution test"]
    fn real_gpu_embedded_rgba8_texture_upload_and_sampling_are_stable() {
        let src = r#"
//!TEXTURE LUT
//!SIZE 1 1
//!FILTER NEAREST
//!BORDER CLAMP
//!FORMAT rgba8
123456ff

//!HOOK MAIN
//!BIND LUT
//!SAVE MAIN
//!WIDTH MAIN.w
//!HEIGHT MAIN.h
//!COMPONENTS 4
vec4 hook() { return LUT_tex(vec2(0.5)); }
"#;
        let shader = UserShader::parse("vulkan_embedded.glsl", src);
        let plan = build_multipass_plan(&shader, 8, 8, 8, 8).unwrap();
        let key = RuntimeKey {
            luid: test_discrete_vulkan_luid().unwrap(),
            input_width: 8,
            input_height: 8,
            output_ref_width: 8,
            output_ref_height: 8,
            shader_hash: shader_runtime_hash(&shader),
        };
        let input = vec![0u8; 8 * 8 * 4];
        let expected = [0x12, 0x34, 0x56, 0xff];
        let mut runtime = VulkanMultiPassRuntime::create(key, plan).unwrap();
        for output in [
            runtime.process(&input).unwrap(),
            runtime.process(&input).unwrap(),
        ] {
            assert!(
                output
                    .output_rgba8
                    .chunks_exact(4)
                    .all(|pixel| pixel == expected),
                "embedded LUT output did not match uploaded texel"
            );
        }
    }

    #[test]
    #[ignore = "real selected-GPU Vulkan execution test"]
    fn real_gpu_embedded_3d_texture_upload_and_sampling_are_stable() {
        let src = r#"
//!TEXTURE LUT
//!SIZE 1 1 2
//!FILTER LINEAR
//!BORDER CLAMP
//!FORMAT rgba8
24486cff24486cff

//!HOOK MAIN
//!BIND LUT
//!SAVE MAIN
//!WIDTH MAIN.w
//!HEIGHT MAIN.h
//!COMPONENTS 4
vec4 hook() { return textureLod(LUT, vec3(0.5), 0.0); }
"#;
        let shader = UserShader::parse("vulkan_embedded_3d.glsl", src);
        let plan = build_multipass_plan(&shader, 8, 8, 8, 8).unwrap();
        let key = RuntimeKey {
            luid: test_discrete_vulkan_luid().unwrap(),
            input_width: 8,
            input_height: 8,
            output_ref_width: 8,
            output_ref_height: 8,
            shader_hash: shader_runtime_hash(&shader),
        };
        let input = vec![0u8; 8 * 8 * 4];
        let expected = [0x24, 0x48, 0x6c, 0xff];
        let mut runtime = VulkanMultiPassRuntime::create(key, plan).unwrap();
        for output in [
            runtime.process(&input).unwrap(),
            runtime.process(&input).unwrap(),
        ] {
            assert!(
                output
                    .output_rgba8
                    .chunks_exact(4)
                    .all(|pixel| pixel == expected),
                "embedded 3D LUT output did not match uploaded texel"
            );
        }
    }

    #[test]
    #[ignore = "real selected-GPU Vulkan execution test"]
    fn real_gpu_storage_image_persists_across_frames() {
        let src = r#"
//!TEXTURE PREV
//!SIZE 4 4
//!FORMAT rgba16f
//!STORAGE

//!HOOK MAIN
//!BIND MAIN
//!BIND PREV
//!SAVE MAIN
//!WIDTH MAIN.w
//!HEIGHT MAIN.h
//!COMPONENTS 4
vec4 hook() {
    ivec2 q = clamp(ivec2(MAIN_pos * MAIN_size), ivec2(0), ivec2(MAIN_size) - ivec2(1));
    vec4 old = imageLoad(PREV, q);
    imageStore(PREV, q, vec4(0.25, 0.5, 0.75, 1.0));
    return old;
}
"#;
        let shader = UserShader::parse("vulkan_storage.glsl", src);
        let plan = build_multipass_plan(&shader, 4, 4, 4, 4).unwrap();
        let key = RuntimeKey {
            luid: test_discrete_vulkan_luid().unwrap(),
            input_width: 4,
            input_height: 4,
            output_ref_width: 4,
            output_ref_height: 4,
            shader_hash: shader_runtime_hash(&shader),
        };
        let input = vec![0u8; 4 * 4 * 4];
        let mut runtime = VulkanMultiPassRuntime::create(key, plan).unwrap();
        let first = runtime.process(&input).unwrap();
        assert!(first.output_rgba8.iter().all(|&value| value == 0));
        let second = runtime.process(&input).unwrap();
        let expected = [64, 128, 191, 255];
        assert!(
            second
                .output_rgba8
                .chunks_exact(4)
                .all(|pixel| pixel.iter().zip(expected).all(|(&a, b)| a.abs_diff(b) <= 1)),
            "storage image did not retain the previous frame value; first={:?}",
            &second.output_rgba8[..4]
        );
    }

    #[test]
    #[ignore = "real selected-GPU Vulkan execution test"]
    fn real_gpu_luma_chroma_identity_preserves_constant_rgb() {
        let src = r#"
//!HOOK LUMA
//!BIND HOOKED
//!SAVE LUMA
//!WIDTH LUMA.w
//!HEIGHT LUMA.h
//!COMPONENTS 1
vec4 hook() { return HOOKED_tex(HOOKED_pos); }
//!HOOK CHROMA
//!BIND HOOKED
//!SAVE CHROMA
//!WIDTH CHROMA.w
//!HEIGHT CHROMA.h
//!COMPONENTS 2
vec4 hook() { return HOOKED_tex(HOOKED_pos); }
"#;
        let shader = UserShader::parse("vulkan_yuv_identity.glsl", src);
        assert!(shader.uses_chroma);
        let plan = build_multipass_plan(&shader, 16, 12, 16, 12).unwrap();
        assert_eq!(plan.passes.len(), 5, "extract Y/C + two hooks + combine");
        let key = RuntimeKey {
            luid: test_discrete_vulkan_luid().unwrap(),
            input_width: 16,
            input_height: 12,
            output_ref_width: 16,
            output_ref_height: 12,
            shader_hash: shader_runtime_hash(&shader),
        };
        let expected = [73u8, 121, 188, 255];
        let input = expected.repeat(16 * 12);
        let mut runtime = VulkanMultiPassRuntime::create(key, plan).unwrap();
        for output in [
            runtime.process(&input).unwrap(),
            runtime.process(&input).unwrap(),
        ] {
            assert!(
                output
                    .output_rgba8
                    .chunks_exact(4)
                    .all(|pixel| { pixel.iter().zip(expected).all(|(&a, b)| a.abs_diff(b) <= 1) })
            );
        }
    }

    #[test]
    #[ignore = "diagnostic corpus scan; set NEO_SHADER_CORPUS to run"]
    fn external_shader_corpus_inventory() {
        let root = std::env::var_os("NEO_SHADER_CORPUS")
            .map(PathBuf::from)
            .expect("NEO_SHADER_CORPUS is required");
        let mut files = Vec::new();
        collect_glsl_files(&root, &mut files);
        files.sort();
        let mut compatible = 0usize;
        let mut rejected = Vec::new();
        for path in &files {
            match UserShader::load(path.to_string_lossy().as_ref()) {
                Ok(shader) => {
                    let scenarios = [
                        (448, 252, 448, 252),
                        (448, 252, 896, 504),
                        (448, 252, 672, 378),
                        (448, 252, 1344, 756),
                        (448, 252, 1792, 1008),
                        (640, 480, 1920, 1080),
                        (1280, 720, 640, 360),
                    ];
                    let mut failure = None;
                    for (iw, ih, ow, oh) in scenarios {
                        let result = analyze_multipass_compatibility(&shader, iw, ih, ow, oh);
                        if !result.compatible
                            && !result.reason.contains("no active passes")
                            && !result.reason.contains(" is not available")
                        {
                            failure = Some(format!("{iw}x{ih}->{ow}x{oh}: {}", result.reason));
                            break;
                        }
                    }
                    if failure.is_none() {
                        compatible += 1;
                    } else {
                        rejected.push((
                            path.clone(),
                            failure.unwrap_or_else(|| {
                                "no active passes in compatibility resolution matrix".to_string()
                            }),
                        ));
                    }
                }
                Err(error) => rejected.push((path.clone(), format!("load failed: {error}"))),
            }
        }
        eprintln!(
            "external Vulkan shader corpus: total={} compatible={} rejected={}",
            files.len(),
            compatible,
            rejected.len()
        );
        for (path, reason) in &rejected {
            eprintln!("REJECTED {}: {reason}", path.display());
        }
        assert!(
            !files.is_empty(),
            "shader corpus is empty: {}",
            root.display()
        );
    }

    #[test]
    #[ignore = "diagnostic corpus admission scan; set NEO_SHADER_CORPUS to run"]
    fn external_shader_production_admission_inventory() {
        let root = std::env::var_os("NEO_SHADER_CORPUS")
            .map(PathBuf::from)
            .expect("NEO_SHADER_CORPUS is required");
        let mut files = Vec::new();
        collect_glsl_files(&root, &mut files);
        files.sort();
        let mut admitted = 0usize;
        let mut embedded = 0usize;
        let mut chroma = 0usize;
        let mut load_failed = 0usize;
        for path in &files {
            match UserShader::load(path.to_string_lossy().as_ref()) {
                Ok(shader) => {
                    if production_shader_admitted(&shader) {
                        admitted += 1;
                    } else {
                        embedded += usize::from(!shader.textures.is_empty());
                        chroma += usize::from(shader.uses_chroma);
                    }
                }
                Err(_) => load_failed += 1,
            }
        }
        eprintln!(
            "external Vulkan production admission: total={} admitted={} blocked_embedded={} blocked_chroma={} load_failed={} overlap_is_counted",
            files.len(),
            admitted,
            embedded,
            chroma,
            load_failed
        );
        assert_eq!(load_failed, 0);
    }
}

// ============================================================================
// v634 experimental Anime4K multi-pass executor
// ============================================================================

use ash::{Entry, vk};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use windows::Win32::{
    Foundation::{CloseHandle, HANDLE},
    Graphics::{
        Direct3D::D3D_FEATURE_LEVEL_11_0,
        Direct3D12::{
            D3D12_HEAP_FLAG_SHARED, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12CreateDevice, ID3D12Device, ID3D12Resource,
        },
        Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC},
        Dxgi::{CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1},
    },
};
use windows::core::PCWSTR;

#[derive(Clone, Debug)]
pub struct MultiPassProductionResult {
    pub output_rgba8: Vec<u8>,
    pub output_width: u32,
    pub output_height: u32,
    pub gpu_name: String,
    pub elapsed_ms: f64,
    pub active_passes: usize,
    pub first_frame_active: bool,
}

/// v664 direct mapped-readback -> OpenGL upload result.  The Vulkan runtime
/// keeps ownership of its persistently mapped readback allocation and passes
/// that memory directly to glTexSubImage2D before returning.  This avoids the
/// previous full-frame `to_vec()` memcpy (about 59 MiB at 5120x2880) while
/// retaining the proven CPU-visible Vulkan->OpenGL bridge and its fallback
/// semantics.
#[derive(Clone, Debug)]
pub struct MultiPassGlUploadResult {
    pub output: GpuTex,
    pub output_width: u32,
    pub output_height: u32,
    pub gpu_name: String,
    /// Vulkan submit + fence wait up to the point where mapped readback is ready.
    pub vulkan_ms: f64,
    /// OpenGL texture upload from the Vulkan mapped readback pointer.
    pub gl_upload_ms: f64,
    /// True when the Vulkan result stayed GPU-resident and OpenGL consumed an
    /// imported external RGBA8 buffer rather than CPU-visible mapped readback.
    pub output_external_buffer: bool,
    /// CPU time spent preparing the Vulkan->GL ownership handoff. In steady
    /// state this is primarily the wait for the previous GL SSBO read fence.
    pub external_sync_ms: f64,
    /// Total call time, including input copy/prepass where applicable.
    pub elapsed_ms: f64,
    pub active_passes: usize,
    pub first_frame_active: bool,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct RuntimeKey {
    luid: u64,
    input_width: u32,
    input_height: u32,
    output_ref_width: u32,
    output_ref_height: u32,
    shader_hash: u64,
}

#[derive(Clone, Copy, Debug)]
enum RuntimeImageRef {
    Input,
    Pass(usize),
    Embedded(usize),
}

struct RuntimeImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    width: u32,
    height: u32,
    depth: u32,
}

struct RuntimeEmbeddedImage {
    name: String,
    image: RuntimeImage,
    sampler: Option<vk::Sampler>,
    upload_offset: vk::DeviceSize,
    upload_size: vk::DeviceSize,
    storage: bool,
}

struct RuntimePass {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_set: vk::DescriptorSet,
    output_index: usize,
    width: u32,
    height: u32,
    dispatch_block_w: u32,
    dispatch_block_h: u32,
}

/// Imported app-owned DirectML interpolation output. The D3D12 committed
/// resource stays alive on the ONNX side; Vulkan owns only its imported
/// VkDeviceMemory reference. v664 converts NCHW FP16/FP32 directly into the
/// runtime's device-local RGBA input image.  The previous path wrote a
/// HOST_VISIBLE upload buffer and then copied that buffer back into the image,
/// causing a needless GPU -> system-memory-visible -> GPU round trip.
struct RuntimeExternalRgbaOutput {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    allocation_byte_len: u64,
    key: u64,
    // v667: own the D3D12 committed resource that backs this Vulkan buffer.
    // The underscore-prefixed fields intentionally keep the COM objects alive
    // for both Vulkan and OpenGL imports without triggering dead-code warnings.
    _d3d12_resource: ID3D12Resource,
    d3d12_device: ID3D12Device,
}

impl RuntimeExternalRgbaOutput {
    fn shared_handle(&self) -> Result<HANDLE> {
        Ok(unsafe {
            self.d3d12_device.CreateSharedHandle(
                &self._d3d12_resource,
                None,
                0x1000_0000,
                PCWSTR::null(),
            )?
        })
    }
}

static NEXT_VULKAN_OUTPUT_KEY: AtomicU64 = AtomicU64::new(1);

fn pending_gl_external_output_retire() -> &'static Mutex<Vec<u64>> {
    static PENDING: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
    PENDING.get_or_init(|| Mutex::new(Vec::new()))
}

fn drain_gl_external_output_retire(gc: &mut GlContext) {
    let keys = {
        let mut pending = pending_gl_external_output_retire().lock().unwrap();
        std::mem::take(&mut *pending)
    };
    for key in keys {
        gc.clear_external_buffer_key(key);
    }
}

struct RuntimeExternalNchwInput {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    initial_command_buffer: vk::CommandBuffer,
    steady_command_buffer: vk::CommandBuffer,
    byte_len: usize,
    padded_width: u32,
    padded_height: u32,
    fp16: bool,
}

struct VulkanMultiPassRuntime {
    _entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    requested_luid: u64,
    device: ash::Device,
    external_memory_win32: Option<ash::khr::external_memory_win32::Device>,
    queue: vk::Queue,
    gpu_name: String,
    input_byte_count: usize,
    output_byte_count: usize,
    upload_buffer: vk::Buffer,
    upload_memory: vk::DeviceMemory,
    upload_ptr: *mut u8,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    readback_ptr: *mut u8,
    external_rgba_output: Option<RuntimeExternalRgbaOutput>,
    external_rgba_gl_failed: bool,
    input_image: RuntimeImage,
    embedded_images: Vec<RuntimeEmbeddedImage>,
    pass_images: Vec<RuntimeImage>,
    pack_image: RuntimeImage,
    sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    passes: Vec<RuntimePass>,
    pack_descriptor_set_layout: vk::DescriptorSetLayout,
    pack_pipeline_layout: vk::PipelineLayout,
    pack_shader_module: vk::ShaderModule,
    pack_pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    initial_command_buffer: vk::CommandBuffer,
    steady_command_buffer: vk::CommandBuffer,
    initial_external_input_command_buffer: vk::CommandBuffer,
    steady_external_input_command_buffer: vk::CommandBuffer,
    initial_external_output_command_buffer: vk::CommandBuffer,
    steady_external_output_command_buffer: vk::CommandBuffer,
    initial_external_input_output_command_buffer: vk::CommandBuffer,
    steady_external_input_output_command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    output_width: u32,
    output_height: u32,
    submission_in_flight: bool,
    image_initialized: bool,
    first_frame_reported: bool,
    session_reusable: bool,
    external_nchw_inputs: HashMap<u64, RuntimeExternalNchwInput>,
}

static ABANDON_MULTIPASS_ON_PROCESS_EXIT: AtomicBool = AtomicBool::new(false);

const MAX_MULTIPASS_RUNTIME_CACHE: usize = 16;

#[derive(Default)]
struct VulkanMultiPassRuntimeCache {
    runtimes: HashMap<RuntimeKey, VulkanMultiPassRuntime>,
    lru: VecDeque<RuntimeKey>,
}

impl VulkanMultiPassRuntimeCache {
    fn touch(&mut self, key: &RuntimeKey) {
        if let Some(pos) = self.lru.iter().position(|candidate| candidate == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key.clone());
    }

    fn prepare_insert(&mut self, key: &RuntimeKey) {
        // Geometry changes must not retain stale large allocations for the same
        // shader/GPU indefinitely. Keep one runtime per shader hash and LUID,
        // while allowing all shaders in the active chain to remain warm.
        let stale = self
            .runtimes
            .keys()
            .filter(|candidate| {
                candidate.luid == key.luid
                    && candidate.shader_hash == key.shader_hash
                    && *candidate != key
            })
            .cloned()
            .collect::<Vec<_>>();
        for stale_key in stale {
            self.runtimes.remove(&stale_key);
            if let Some(pos) = self
                .lru
                .iter()
                .position(|candidate| candidate == &stale_key)
            {
                self.lru.remove(pos);
            }
        }
        while self.runtimes.len() >= MAX_MULTIPASS_RUNTIME_CACHE {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.runtimes.remove(&oldest);
        }
    }
}

thread_local! {
    static MULTIPASS_RUNTIMES: RefCell<VulkanMultiPassRuntimeCache> = RefCell::new(VulkanMultiPassRuntimeCache::default());
}

fn multipass_failed_keys() -> &'static Mutex<HashSet<RuntimeKey>> {
    static FAILED: OnceLock<Mutex<HashSet<RuntimeKey>>> = OnceLock::new();
    FAILED.get_or_init(|| Mutex::new(HashSet::new()))
}

fn d3d12_import_failed_keys() -> &'static Mutex<HashSet<RuntimeKey>> {
    static FAILED: OnceLock<Mutex<HashSet<RuntimeKey>>> = OnceLock::new();
    FAILED.get_or_init(|| Mutex::new(HashSet::new()))
}

pub fn prepare_runtime_for_process_exit() {
    ABANDON_MULTIPASS_ON_PROCESS_EXIT.store(true, Ordering::Release);
}

pub fn reset_runtime() {
    MULTIPASS_RUNTIMES.with(|slot| {
        let mut cache = slot.borrow_mut();
        cache.runtimes.clear();
        cache.lru.clear();
    });
    multipass_failed_keys().lock().unwrap().clear();
    d3d12_import_failed_keys().lock().unwrap().clear();
}

/// Capture-session reset that keeps only stateless runtimes warm. GPU changes
/// still call `reset_runtime()` and drop everything. This preserves temporal
/// shader correctness while avoiding repeated Vulkan pipeline creation for the
/// common resize/Anime4K chain on Stop -> Start.
pub fn reset_session_runtime() {
    let (kept, dropped) = MULTIPASS_RUNTIMES.with(|slot| {
        let mut cache = slot.borrow_mut();
        let before = cache.runtimes.len();
        for runtime in cache.runtimes.values_mut() {
            runtime.clear_external_nchw_inputs();
        }
        cache.runtimes.retain(|_, runtime| runtime.session_reusable);
        let remaining = cache.runtimes.keys().cloned().collect::<HashSet<_>>();
        cache.lru.retain(|key| remaining.contains(key));
        let kept = cache.runtimes.len();
        (kept, before.saturating_sub(kept))
    });
    multipass_failed_keys().lock().unwrap().clear();
    d3d12_import_failed_keys().lock().unwrap().clear();
    if kept > 0 || dropped > 0 {
        log::info!(
            "vulkan-multipass-session-cache: kept_stateless={} dropped_stateful={} policy=warm-restart",
            kept,
            dropped
        );
    }
}

pub fn production_requested() -> bool {
    // The caller additionally requires an explicitly selected DXGI LUID, so
    // GPU=Auto never initializes Vulkan. Keep an opt-out for diagnostics, but
    // make the completed selected-GPU backend usable in normal distributions.
    std::env::var("NEO_VULKAN_MULTIPASS_PRODUCTION")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

pub fn production_shader_admitted(shader: &UserShader) -> bool {
    !shader.passes.is_empty()
}

fn shader_runtime_hash(shader: &UserShader) -> u64 {
    let mut hasher = DefaultHasher::new();
    shader.source_hash.hash(&mut hasher);
    for param in &shader.params {
        param.name.hash(&mut hasher);
        param.value.to_bits().hash(&mut hasher);
        std::mem::discriminant(&param.ty).hash(&mut hasher);
    }
    hasher.finish()
}

fn luid_from_vk(bytes: &[u8; vk::LUID_SIZE]) -> u64 {
    u64::from_le_bytes(*bytes)
}

fn vk_device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .trim()
        .to_string()
}

fn find_memory_type_with_flags(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|index| {
        (type_bits & (1u32 << index)) != 0
            && properties.memory_types[*index as usize]
                .property_flags
                .contains(required)
    })
}

fn vk_format_for_components(components: u8) -> Result<vk::Format> {
    match components {
        1 => Ok(vk::Format::R16_SFLOAT),
        2 => Ok(vk::Format::R16G16_SFLOAT),
        3 | 4 => Ok(vk::Format::R16G16B16A16_SFLOAT),
        other => Err(anyhow!(
            "unsupported Vulkan intermediate COMPONENTS={other}"
        )),
    }
}

fn vk_format_for_embedded(texture: &ShaderTexture) -> Result<vk::Format> {
    match (texture.f16, texture.comps) {
        (true, 1) => Ok(vk::Format::R16_SFLOAT),
        (true, 2) => Ok(vk::Format::R16G16_SFLOAT),
        (true, 3 | 4) => Ok(vk::Format::R16G16B16A16_SFLOAT),
        (false, 1) => Ok(vk::Format::R8_UNORM),
        (false, 2) => Ok(vk::Format::R8G8_UNORM),
        (false, 3 | 4) => Ok(vk::Format::R8G8B8A8_UNORM),
        (_, other) => Err(anyhow!("unsupported embedded COMPONENTS={other}")),
    }
}

fn embedded_upload_bytes(texture: &ShaderTexture) -> Vec<u8> {
    if texture.comps != 3 {
        return texture.data.clone();
    }
    if texture.f16 {
        let one = half::f16::ONE.to_le_bytes();
        texture
            .data
            .chunks_exact(6)
            .flat_map(|rgb| {
                [
                    rgb[0], rgb[1], rgb[2], rgb[3], rgb[4], rgb[5], one[0], one[1],
                ]
            })
            .collect()
    } else {
        texture
            .data
            .chunks_exact(3)
            .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
            .collect()
    }
}

fn embedded_address_mode(border: TexBorder) -> vk::SamplerAddressMode {
    match border {
        TexBorder::Clamp => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        TexBorder::Repeat => vk::SamplerAddressMode::REPEAT,
        TexBorder::Mirror => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

fn create_runtime_image(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    width: u32,
    height: u32,
    usage: vk::ImageUsageFlags,
) -> Result<RuntimeImage> {
    create_runtime_image_depth(
        instance,
        device,
        physical_device,
        format,
        width,
        height,
        1,
        usage,
    )
}

fn create_runtime_image_depth(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    width: u32,
    height: u32,
    depth: u32,
    usage: vk::ImageUsageFlags,
) -> Result<RuntimeImage> {
    let extent = vk::Extent3D {
        width,
        height,
        depth,
    };
    let info = vk::ImageCreateInfo::default()
        .image_type(if depth > 1 {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        })
        .format(format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image =
        unsafe { device.create_image(&info, None) }.context("vkCreateImage multi-pass failed")?;
    let req = unsafe { device.get_image_memory_requirements(image) };
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let memory_type = find_memory_type_with_flags(
        &memory_properties,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .or_else(|| {
        find_memory_type_with_flags(
            &memory_properties,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::empty(),
        )
    })
    .ok_or_else(|| anyhow!("no compatible Vulkan image memory"))?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(memory_type);
    let memory = unsafe { device.allocate_memory(&alloc, None) }
        .context("vkAllocateMemory multi-pass image failed")?;
    unsafe { device.bind_image_memory(image, memory, 0) }
        .context("vkBindImageMemory multi-pass failed")?;
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(if depth > 1 {
                    vk::ImageViewType::TYPE_3D
                } else {
                    vk::ImageViewType::TYPE_2D
                })
                .format(format)
                .subresource_range(range),
            None,
        )
    }
    .context("vkCreateImageView multi-pass failed")?;
    Ok(RuntimeImage {
        image,
        memory,
        view,
        width,
        height,
        depth,
    })
}

fn ensure_format_features(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    required: vk::FormatFeatureFlags,
) -> Result<()> {
    let props = unsafe { instance.get_physical_device_format_properties(physical_device, format) };
    if !props.optimal_tiling_features.contains(required) {
        return Err(anyhow!(
            "selected Vulkan GPU format {:?} lacks required optimal features {:?}; available={:?}",
            format,
            required,
            props.optimal_tiling_features
        ));
    }
    Ok(())
}

fn pack_compute_source(width: u32, height: u32) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(rgba8, set = 0, binding = 0) uniform image2D out_image;
layout(set = 0, binding = 1) uniform texture2D src_texture;
layout(set = 0, binding = 2) uniform sampler src_sampler;
#define src_tx sampler2D(src_texture, src_sampler)
void main() {{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= {width} || p.y >= {height}) return;
    vec2 uv = (vec2(p) + vec2(0.5)) / vec2({width}.0, {height}.0);
    imageStore(out_image, p, clamp(textureLod(src_tx, uv, 0.0), 0.0, 1.0));
}}
"#
    )
}

fn external_nchw_to_rgba8_compute_source(
    width: u32,
    height: u32,
    padded_width: u32,
    padded_height: u32,
    fp16: bool,
) -> String {
    let load = if fp16 {
        r#"
float neo_load(uint index) {
    uint word = src.data[index >> 1u];
    vec2 pair = unpackHalf2x16(word);
    return ((index & 1u) == 0u) ? pair.x : pair.y;
}
"#
    } else {
        r#"
float neo_load(uint index) {
    return uintBitsToFloat(src.data[index]);
}
"#
    };
    format!(
        r#"#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(std430, set = 0, binding = 0) readonly buffer SrcBuffer {{ uint data[]; }} src;
layout(rgba8, set = 0, binding = 1) uniform image2D dst_image;
{load}
void main() {{
    uvec2 p = gl_GlobalInvocationID.xy;
    if (p.x >= {width}u || p.y >= {height}u) return;
    uint pixel = p.y * {padded_width}u + p.x;
    uint plane = {padded_width}u * {padded_height}u;
    float r = clamp(neo_load(pixel), 0.0, 1.0);
    float g = clamp(neo_load(plane + pixel), 0.0, 1.0);
    float b = clamp(neo_load(2u * plane + pixel), 0.0, 1.0);
    imageStore(dst_image, ivec2(p), vec4(r, g, b, 1.0));
}}
"#
    )
}

fn create_compute_pipeline(
    device: &ash::Device,
    spirv: &[u32],
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<(vk::PipelineLayout, vk::ShaderModule, vk::Pipeline)> {
    let set_layouts = [descriptor_set_layout];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
            None,
        )
    }
    .context("vkCreatePipelineLayout multi-pass failed")?;
    let shader_module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
    }
    .context("vkCreateShaderModule multi-pass failed")?;
    let entry_name = CString::new("main")?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_name);
    let infos = [vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout)];
    let pipelines =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None) }
            .map_err(|(partial, error)| {
                unsafe {
                    for pipeline in partial {
                        device.destroy_pipeline(pipeline, None);
                    }
                }
                anyhow!("vkCreateComputePipelines multi-pass failed: {error:?}")
            })?;
    Ok((pipeline_layout, shader_module, pipelines[0]))
}

fn runtime_image<'a>(
    input: &'a RuntimeImage,
    embedded_images: &'a [RuntimeEmbeddedImage],
    pass_images: &'a [RuntimeImage],
    image_ref: RuntimeImageRef,
) -> &'a RuntimeImage {
    match image_ref {
        RuntimeImageRef::Input => input,
        RuntimeImageRef::Pass(index) => &pass_images[index],
        RuntimeImageRef::Embedded(index) => &embedded_images[index].image,
    }
}

fn image_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

fn record_multipass_commands(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    initialized: bool,
    upload_input: bool,
    upload_buffer: vk::Buffer,
    readback_buffer: vk::Buffer,
    input_image: &RuntimeImage,
    embedded_images: &[RuntimeEmbeddedImage],
    pass_images: &[RuntimeImage],
    passes: &[RuntimePass],
    pack_image: &RuntimeImage,
    pack_pipeline: vk::Pipeline,
    pack_pipeline_layout: vk::PipelineLayout,
    pack_descriptor_set: vk::DescriptorSet,
    upload_byte_count: usize,
    output_byte_count: usize,
    output_host_readback: bool,
) -> Result<()> {
    let range = image_range();
    let layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);
    let input_copy = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(layers)
        .image_extent(vk::Extent3D {
            width: input_image.width,
            height: input_image.height,
            depth: 1,
        });
    let output_copy = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(layers)
        .image_extent(vk::Extent3D {
            width: pack_image.width,
            height: pack_image.height,
            depth: 1,
        });
    unsafe {
        device
            .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
            .context("vkBeginCommandBuffer multi-pass failed")?;

        let upload_barrier = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(upload_buffer)
            .offset(0)
            .size(upload_byte_count.max(1) as vk::DeviceSize)];
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &upload_barrier,
            &[],
        );

        if upload_input {
            let input_old = if initialized {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::UNDEFINED
            };
            let input_src_access = if initialized {
                vk::AccessFlags::SHADER_READ
            } else {
                vk::AccessFlags::empty()
            };
            let input_src_stage = if initialized {
                vk::PipelineStageFlags::COMPUTE_SHADER
            } else {
                vk::PipelineStageFlags::TOP_OF_PIPE
            };
            let to_upload = [vk::ImageMemoryBarrier::default()
                .src_access_mask(input_src_access)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(input_old)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(input_image.image)
                .subresource_range(range)];
            device.cmd_pipeline_barrier(
                command_buffer,
                input_src_stage,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_upload,
            );
            device.cmd_copy_buffer_to_image(
                command_buffer,
                upload_buffer,
                input_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[input_copy],
            );
            let input_ready = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(input_image.image)
                .subresource_range(range)];
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &input_ready,
            );
        }

        if !initialized {
            for embedded in embedded_images {
                let to_upload = [vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(embedded.image.image)
                    .subresource_range(range)];
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &to_upload,
                );
                if embedded.upload_size > 0 {
                    let copy = vk::BufferImageCopy::default()
                        .buffer_offset(embedded.upload_offset)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(layers)
                        .image_extent(vk::Extent3D {
                            width: embedded.image.width,
                            height: embedded.image.height,
                            depth: embedded.image.depth,
                        });
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        upload_buffer,
                        embedded.image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[copy],
                    );
                }
                let ready = [vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(if embedded.storage {
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
                    } else {
                        vk::AccessFlags::SHADER_READ
                    })
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(embedded.image.image)
                    .subresource_range(range)];
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &ready,
                );
            }
        }

        for pass in passes {
            let output = &pass_images[pass.output_index];
            let old_layout = if initialized {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::UNDEFINED
            };
            let src_access = if initialized {
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
            } else {
                vk::AccessFlags::empty()
            };
            let src_stage = if initialized {
                vk::PipelineStageFlags::COMPUTE_SHADER
            } else {
                vk::PipelineStageFlags::TOP_OF_PIPE
            };
            let output_ready = [vk::ImageMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(output.image)
                .subresource_range(range)];
            device.cmd_pipeline_barrier(
                command_buffer,
                src_stage,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &output_ready,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pass.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pass.pipeline_layout,
                0,
                &[pass.descriptor_set],
                &[],
            );
            device.cmd_dispatch(
                command_buffer,
                pass.width.div_ceil(pass.dispatch_block_w.max(1)),
                pass.height.div_ceil(pass.dispatch_block_h.max(1)),
                1,
            );
            let mut make_readable = vec![
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(output.image)
                    .subresource_range(range),
            ];
            make_readable.extend(
                embedded_images
                    .iter()
                    .filter(|embedded| embedded.storage)
                    .map(|embedded| {
                        vk::ImageMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::GENERAL)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(embedded.image.image)
                            .subresource_range(range)
                    }),
            );
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &make_readable,
            );
        }

        let pack_old = if initialized {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        let pack_src_access = if initialized {
            vk::AccessFlags::TRANSFER_READ
        } else {
            vk::AccessFlags::empty()
        };
        let pack_src_stage = if initialized {
            vk::PipelineStageFlags::TRANSFER
        } else {
            vk::PipelineStageFlags::TOP_OF_PIPE
        };
        let pack_ready = [vk::ImageMemoryBarrier::default()
            .src_access_mask(pack_src_access)
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(pack_old)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(pack_image.image)
            .subresource_range(range)];
        device.cmd_pipeline_barrier(
            command_buffer,
            pack_src_stage,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &pack_ready,
        );
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pack_pipeline,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pack_pipeline_layout,
            0,
            &[pack_descriptor_set],
            &[],
        );
        device.cmd_dispatch(
            command_buffer,
            pack_image.width.div_ceil(16),
            pack_image.height.div_ceil(8),
            1,
        );

        let pack_to_copy = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(pack_image.image)
            .subresource_range(range)];
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &pack_to_copy,
        );
        device.cmd_copy_image_to_buffer(
            command_buffer,
            pack_image.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback_buffer,
            &[output_copy],
        );
        let output_dst_access = if output_host_readback {
            vk::AccessFlags::HOST_READ
        } else {
            vk::AccessFlags::MEMORY_READ
        };
        let output_dst_stage = if output_host_readback {
            vk::PipelineStageFlags::HOST
        } else {
            vk::PipelineStageFlags::ALL_COMMANDS
        };
        let readback_barrier = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(output_dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(readback_buffer)
            .offset(0)
            .size(output_byte_count as vk::DeviceSize)];
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            output_dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &readback_barrier,
            &[],
        );
        device
            .end_command_buffer(command_buffer)
            .context("vkEndCommandBuffer multi-pass failed")?;
    }
    Ok(())
}

fn d3d12_device_for_luid_u64(wanted: u64) -> Result<ID3D12Device> {
    let factory = unsafe { CreateDXGIFactory1::<IDXGIFactory1>()? };
    for index in 0u32.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
            break;
        };
        let desc = unsafe { adapter.GetDesc1()? };
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }
        let luid =
            ((desc.AdapterLuid.HighPart as u32 as u64) << 32) | desc.AdapterLuid.LowPart as u64;
        if luid != wanted {
            continue;
        }
        let mut device = None;
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device)? };
        return device.ok_or_else(|| anyhow!("D3D12CreateDevice returned no device"));
    }
    Err(anyhow!(
        "no DXGI adapter matched selected Vulkan LUID {wanted:016x}"
    ))
}

/// v667 Vulkan->OpenGL zero-copy output.
///
/// v666 asked Vulkan to *export* a D3D12_RESOURCE buffer. On the tested
/// RTX 5070 Ti driver the buffer handle type is importable but not exportable,
/// so the optional route silently stayed on mapped readback.  v667 reverses
/// ownership: create a shareable D3D12 committed buffer on the selected LUID,
/// then import that same resource into Vulkan. OpenGL already imports the same
/// D3D12_RESOURCE handle type successfully for Neo's DirectML bridge, so this
/// keeps both sides on the proven Windows interop direction.
fn create_shared_rgba_output_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    external: Option<&ash::khr::external_memory_win32::Device>,
    requested_luid: u64,
    byte_len: usize,
) -> Result<Option<RuntimeExternalRgbaOutput>> {
    let Some(external) = external else {
        log::info!(
            "vulkan-gl-external-output: result=unavailable phase=capability requested_luid={requested_luid:016x} reason=VK_KHR_external_memory_win32-unavailable fallback=mapped-readback"
        );
        return Ok(None);
    };
    let handle_type = vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE;
    let usage = vk::BufferUsageFlags::TRANSFER_DST;
    let external_info = vk::PhysicalDeviceExternalBufferInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(usage)
        .handle_type(handle_type);
    let mut props = vk::ExternalBufferProperties::default();
    unsafe {
        instance.get_physical_device_external_buffer_properties(
            physical_device,
            &external_info,
            &mut props,
        );
    }
    let features = props.external_memory_properties.external_memory_features;
    let importable = features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE);
    let exportable = features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE);
    log::info!(
        "vulkan-gl-external-output: phase=capability requested_luid={requested_luid:016x} ownership=d3d12-committed-buffer handle=D3D12_RESOURCE importable={importable} exportable={exportable} features=0x{:x}",
        features.as_raw()
    );
    if !importable {
        log::info!(
            "vulkan-gl-external-output: result=unavailable phase=capability requested_luid={requested_luid:016x} reason=D3D12_RESOURCE-buffer-not-importable fallback=mapped-readback"
        );
        return Ok(None);
    }

    let d3d12_device = d3d12_device_for_luid_u64(requested_luid)?;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: byte_len.max(1) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let d3d12_allocation = unsafe { d3d12_device.GetResourceAllocationInfo(0, &[desc]) };
    anyhow::ensure!(
        d3d12_allocation.SizeInBytes != 0 && d3d12_allocation.SizeInBytes != u64::MAX,
        "invalid Vulkan->OpenGL D3D12 shared allocation size"
    );
    let mut d3d12_resource = None;
    unsafe {
        d3d12_device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_SHARED,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut d3d12_resource,
        )?;
    }
    let d3d12_resource = d3d12_resource
        .ok_or_else(|| anyhow!("D3D12 shared Vulkan output creation returned no resource"))?;
    let shared_handle = unsafe {
        d3d12_device.CreateSharedHandle(&d3d12_resource, None, 0x1000_0000, PCWSTR::null())?
    };

    let mut external_create =
        vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
    let buffer_info = vk::BufferCreateInfo::default()
        .size(byte_len.max(1) as vk::DeviceSize)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external_create);
    let buffer = match unsafe { device.create_buffer(&buffer_info, None) } {
        Ok(buffer) => buffer,
        Err(error) => {
            unsafe {
                let _ = CloseHandle(shared_handle);
            }
            return Err(anyhow!(
                "vkCreateBuffer D3D12-owned Vulkan output failed: {error:?}"
            ));
        }
    };
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mut handle_props = vk::MemoryWin32HandlePropertiesKHR::default();
    if let Err(error) = unsafe {
        external.get_memory_win32_handle_properties(
            handle_type,
            shared_handle.0 as isize,
            &mut handle_props,
        )
    } {
        unsafe {
            device.destroy_buffer(buffer, None);
            let _ = CloseHandle(shared_handle);
        }
        return Err(anyhow!(
            "vkGetMemoryWin32HandlePropertiesKHR D3D12-owned output failed: {error:?}"
        ));
    }
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let compatible_bits = req.memory_type_bits & handle_props.memory_type_bits;
    let memory_type = match find_memory_type_with_flags(
        &memory_properties,
        compatible_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .or_else(|| {
        find_memory_type_with_flags(
            &memory_properties,
            compatible_bits,
            vk::MemoryPropertyFlags::empty(),
        )
    }) {
        Some(memory_type) => memory_type,
        None => {
            unsafe {
                device.destroy_buffer(buffer, None);
                let _ = CloseHandle(shared_handle);
            }
            return Err(anyhow!(
                "no compatible Vulkan memory type for D3D12-owned output import"
            ));
        }
    };
    let mut import_info = vk::ImportMemoryWin32HandleInfoKHR::default()
        .handle_type(handle_type)
        .handle(shared_handle.0 as isize);
    let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(memory_type)
        .push_next(&mut import_info)
        .push_next(&mut dedicated_info);
    let memory_result = unsafe { device.allocate_memory(&alloc, None) };
    // Win32 NT handle ownership is retained by the application for Vulkan
    // imports. The imported VkDeviceMemory and the D3D12 resource each retain
    // the underlying payload after this handle is closed.
    unsafe {
        let _ = CloseHandle(shared_handle);
    }
    let memory = match memory_result {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(anyhow!(
                "vkAllocateMemory D3D12-owned Vulkan output import failed: {error:?}"
            ));
        }
    };
    if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return Err(anyhow!(
            "vkBindBufferMemory D3D12-owned Vulkan output failed: {error:?}"
        ));
    }

    let serial = NEXT_VULKAN_OUTPUT_KEY.fetch_add(1, Ordering::Relaxed);
    let key = 0x8000_0000_0000_0000u64 | (serial & 0x7fff_ffff_ffff_ffff);
    log::info!(
        "vulkan-gl-external-output: result=prepared key={key} bytes={} allocation_bytes={} ownership=d3d12-committed-buffer vulkan_import=true",
        byte_len,
        d3d12_allocation.SizeInBytes
    );
    Ok(Some(RuntimeExternalRgbaOutput {
        buffer,
        memory,
        allocation_byte_len: d3d12_allocation.SizeInBytes,
        key,
        _d3d12_resource: d3d12_resource,
        d3d12_device,
    }))
}

impl VulkanMultiPassRuntime {
    fn create(key: RuntimeKey, plan: MultiPassPlan) -> Result<Self> {
        // Persistent STORAGE textures intentionally carry temporal state and
        // must never survive a capture-session boundary. Ordinary resize /
        // Anime4K-style shaders are stateless and can safely keep their VkDevice,
        // pipelines and images warm across Stop -> Start on the same GPU.
        let session_reusable = !plan.textures.iter().any(|texture| texture.storage);
        let entry = unsafe { Entry::load() }.context("Vulkan loader could not be loaded")?;
        let app_name = CString::new("cHiDeScaler-Neo Vulkan MultiPass v634")?;
        let engine_name = CString::new("cHiDeScaler-Neo")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_1);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
        }
        .context("vkCreateInstance multi-pass failed")?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .context("vkEnumeratePhysicalDevices multi-pass failed")?;
        let mut selected = None;
        let mut seen = Vec::new();
        for physical_device in physical_devices {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
            let properties = properties2.properties;
            let name = vk_device_name(&properties);
            let luid = (id.device_luid_valid == vk::TRUE).then(|| luid_from_vk(&id.device_luid));
            seen.push(format!(
                "{}:{}",
                name,
                luid.map(|v| format!("{v:016x}"))
                    .unwrap_or_else(|| "no-luid".into())
            ));
            if luid != Some(key.luid) {
                continue;
            }
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let queue_family_index = queue_families
                .iter()
                .enumerate()
                .find(|(_, family)| {
                    family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                })
                .map(|(index, _)| index as u32)
                .ok_or_else(|| anyhow!("selected Vulkan GPU has no compute queue"))?;
            selected = Some((physical_device, name, queue_family_index));
            break;
        }
        let (physical_device, gpu_name, queue_family_index) = selected.ok_or_else(|| {
            anyhow!(
                "no Vulkan physical device matched selected DXGI LUID {:016x}; enumerated=[{}]",
                key.luid,
                seen.join(", ")
            )
        })?;

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let external_memory_win32_supported =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }
                .context("vkEnumerateDeviceExtensionProperties failed")?
                .iter()
                .any(|property| unsafe {
                    CStr::from_ptr(property.extension_name.as_ptr())
                        == ash::khr::external_memory_win32::NAME
                });
        let mut enabled_extensions = Vec::new();
        if external_memory_win32_supported {
            enabled_extensions.push(ash::khr::external_memory_win32::NAME.as_ptr());
        }
        let device = unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_infos)
                    .enabled_extension_names(&enabled_extensions),
                None,
            )
        }
        .context("vkCreateDevice multi-pass failed")?;
        let external_memory_win32 = external_memory_win32_supported
            .then(|| ash::khr::external_memory_win32::Device::new(&instance, &device));
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let intermediate_features = vk::FormatFeatureFlags::STORAGE_IMAGE
            | vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
        let mut checked_formats = Vec::new();
        for pass in &plan.passes {
            let format = vk_format_for_components(pass.components)?;
            if !checked_formats.contains(&format) {
                checked_formats.push(format);
                ensure_format_features(&instance, physical_device, format, intermediate_features)?;
            }
        }
        ensure_format_features(
            &instance,
            physical_device,
            vk::Format::R8G8B8A8_UNORM,
            vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        )?;
        ensure_format_features(
            &instance,
            physical_device,
            vk::Format::R8G8B8A8_UNORM,
            vk::FormatFeatureFlags::STORAGE_IMAGE,
        )?;

        let input_byte_count = (key.input_width as usize) * (key.input_height as usize) * 4;
        let output_byte_count = (plan.final_width as usize) * (plan.final_height as usize) * 4;
        let mut embedded_payloads = Vec::with_capacity(plan.textures.len());
        let mut upload_byte_count = input_byte_count;
        for texture in &plan.textures {
            upload_byte_count = upload_byte_count.div_ceil(16) * 16;
            let bytes = embedded_upload_bytes(texture);
            let offset = upload_byte_count;
            upload_byte_count = upload_byte_count.saturating_add(bytes.len());
            embedded_payloads.push((offset, bytes));
        }
        let host_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let upload_buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(upload_byte_count.max(1) as vk::DeviceSize)
                    .usage(
                        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .context("vkCreateBuffer multi-pass upload failed")?;
        let upload_req = unsafe { device.get_buffer_memory_requirements(upload_buffer) };
        let upload_type = find_memory_type_with_flags(
            &memory_properties,
            upload_req.memory_type_bits,
            host_flags,
        )
        .ok_or_else(|| anyhow!("no HOST_VISIBLE|HOST_COHERENT upload memory"))?;
        let upload_memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(upload_req.size)
                    .memory_type_index(upload_type),
                None,
            )
        }
        .context("vkAllocateMemory multi-pass upload failed")?;
        unsafe { device.bind_buffer_memory(upload_buffer, upload_memory, 0) }
            .context("vkBindBufferMemory multi-pass upload failed")?;
        let upload_ptr = unsafe {
            device.map_memory(
                upload_memory,
                0,
                upload_byte_count.max(1) as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .context("vkMapMemory multi-pass upload failed")?
        .cast::<u8>();
        for (offset, bytes) in &embedded_payloads {
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), upload_ptr.add(*offset), bytes.len());
            }
        }

        let readback_buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(output_byte_count as vk::DeviceSize)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .context("vkCreateBuffer multi-pass readback failed")?;
        let readback_req = unsafe { device.get_buffer_memory_requirements(readback_buffer) };
        let cached_flags = host_flags | vk::MemoryPropertyFlags::HOST_CACHED;
        let readback_type = find_memory_type_with_flags(
            &memory_properties,
            readback_req.memory_type_bits,
            cached_flags,
        )
        .or_else(|| {
            find_memory_type_with_flags(
                &memory_properties,
                readback_req.memory_type_bits,
                host_flags,
            )
        })
        .ok_or_else(|| anyhow!("no HOST_VISIBLE readback memory"))?;
        let readback_memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(readback_req.size)
                    .memory_type_index(readback_type),
                None,
            )
        }
        .context("vkAllocateMemory multi-pass readback failed")?;
        unsafe { device.bind_buffer_memory(readback_buffer, readback_memory, 0) }
            .context("vkBindBufferMemory multi-pass readback failed")?;
        let readback_ptr = unsafe {
            device.map_memory(
                readback_memory,
                0,
                output_byte_count as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .context("vkMapMemory multi-pass readback failed")?
        .cast::<u8>();

        let external_rgba_output = match create_shared_rgba_output_buffer(
            &instance,
            &device,
            physical_device,
            external_memory_win32.as_ref(),
            key.luid,
            output_byte_count,
        ) {
            Ok(output) => output,
            Err(error) => {
                log::info!(
                    "vulkan-gl-external-output: result=unavailable requested_luid={:016x} reason={:#} fallback=mapped-readback",
                    key.luid,
                    error
                );
                None
            }
        };

        let input_image = create_runtime_image(
            &instance,
            &device,
            physical_device,
            vk::Format::R8G8B8A8_UNORM,
            key.input_width,
            key.input_height,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE,
        )?;
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_lod(0.0)
                    .max_lod(0.0),
                None,
            )
        }
        .context("vkCreateSampler multi-pass failed")?;

        let mut embedded_images = Vec::with_capacity(plan.textures.len());
        for (texture, (upload_offset, bytes)) in plan.textures.iter().zip(embedded_payloads.iter())
        {
            let format = vk_format_for_embedded(texture)?;
            let required = if texture.storage {
                vk::FormatFeatureFlags::STORAGE_IMAGE
            } else {
                vk::FormatFeatureFlags::SAMPLED_IMAGE
                    | if texture.filter_linear {
                        vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
                    } else {
                        vk::FormatFeatureFlags::empty()
                    }
            };
            ensure_format_features(&instance, physical_device, format, required)?;
            let image = create_runtime_image_depth(
                &instance,
                &device,
                physical_device,
                format,
                texture.w.max(1) as u32,
                texture.h.max(1) as u32,
                texture.d.max(1) as u32,
                vk::ImageUsageFlags::TRANSFER_DST
                    | if texture.storage {
                        vk::ImageUsageFlags::STORAGE
                    } else {
                        vk::ImageUsageFlags::SAMPLED
                    },
            )?;
            let texture_sampler = if texture.storage {
                None
            } else {
                let filter = if texture.filter_linear {
                    vk::Filter::LINEAR
                } else {
                    vk::Filter::NEAREST
                };
                let address = embedded_address_mode(texture.border);
                Some(
                    unsafe {
                        device.create_sampler(
                            &vk::SamplerCreateInfo::default()
                                .mag_filter(filter)
                                .min_filter(filter)
                                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                                .address_mode_u(address)
                                .address_mode_v(address)
                                .address_mode_w(address)
                                .min_lod(0.0)
                                .max_lod(0.0),
                            None,
                        )
                    }
                    .with_context(|| format!("vkCreateSampler embedded {} failed", texture.name))?,
                )
            };
            embedded_images.push(RuntimeEmbeddedImage {
                name: texture.name.clone(),
                image,
                sampler: texture_sampler,
                upload_offset: *upload_offset as vk::DeviceSize,
                upload_size: bytes.len() as vk::DeviceSize,
                storage: texture.storage,
            });
        }

        let mut resource_map: HashMap<String, RuntimeImageRef> = HashMap::from([
            ("MAIN".to_string(), RuntimeImageRef::Input),
            ("RGB".to_string(), RuntimeImageRef::Input),
            ("NATIVE".to_string(), RuntimeImageRef::Input),
            ("MAINPRESUB".to_string(), RuntimeImageRef::Input),
            ("OUTPUT".to_string(), RuntimeImageRef::Input),
        ]);
        for (index, embedded) in embedded_images.iter().enumerate() {
            resource_map.insert(embedded.name.clone(), RuntimeImageRef::Embedded(index));
        }
        let mut pass_images = Vec::with_capacity(plan.passes.len());
        let mut pass_bind_refs: Vec<Vec<RuntimeImageRef>> = Vec::with_capacity(plan.passes.len());
        for pass in &plan.passes {
            let refs = pass
                .binds
                .iter()
                .map(|bind| {
                    resource_map
                        .get(&bind.resource_name)
                        .copied()
                        .ok_or_else(|| {
                            anyhow!("runtime BIND resource {} missing", bind.resource_name)
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            pass_bind_refs.push(refs);
            let output_index = pass_images.len();
            pass_images.push(create_runtime_image(
                &instance,
                &device,
                physical_device,
                vk_format_for_components(pass.components)?,
                pass.width,
                pass.height,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            )?);
            resource_map.insert(pass.save.clone(), RuntimeImageRef::Pass(output_index));
        }
        let final_ref = resource_map
            .get(&plan.final_resource)
            .copied()
            .or_else(|| {
                plan.passes
                    .last()
                    .map(|_| RuntimeImageRef::Pass(pass_images.len() - 1))
            })
            .ok_or_else(|| {
                anyhow!(
                    "final multi-pass runtime resource {} missing",
                    plan.final_resource
                )
            })?;
        let pack_image = create_runtime_image(
            &instance,
            &device,
            physical_device,
            vk::Format::R8G8B8A8_UNORM,
            plan.final_width,
            plan.final_height,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;

        let total_sampled = plan
            .passes
            .iter()
            .map(|p| p.binds.iter().filter(|bind| !bind.storage).count() as u32)
            .sum::<u32>()
            + 1;
        let total_storage = plan.passes.len() as u32
            + plan
                .passes
                .iter()
                .map(|p| p.binds.iter().filter(|bind| bind.storage).count() as u32)
                .sum::<u32>()
            + 1;
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: total_storage,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: total_sampled,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: total_sampled,
            },
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(plan.passes.len() as u32 + 1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .context("vkCreateDescriptorPool multi-pass failed")?;

        let mut runtime_passes = Vec::with_capacity(plan.passes.len());
        for (pass_index, planned) in plan.passes.iter().enumerate() {
            let mut bindings = Vec::with_capacity(planned.binds.len() + 1);
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
            let mut next_binding = 1u32;
            for bind in &planned.binds {
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(next_binding)
                        .descriptor_type(if bind.storage {
                            vk::DescriptorType::STORAGE_IMAGE
                        } else {
                            vk::DescriptorType::SAMPLED_IMAGE
                        })
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
                next_binding += 1;
                if !bind.storage {
                    bindings.push(
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(next_binding)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    );
                    next_binding += 1;
                }
            }
            let descriptor_set_layout = unsafe {
                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
            }
            .context("vkCreateDescriptorSetLayout multi-pass pass failed")?;
            let (pipeline_layout, shader_module, pipeline) =
                create_compute_pipeline(&device, &planned.spirv, descriptor_set_layout)?;
            let set_layouts = [descriptor_set_layout];
            let descriptor_set = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set_layouts),
                )
            }
            .context("vkAllocateDescriptorSets multi-pass pass failed")?[0];
            let output_index = pass_index;
            let output = &pass_images[output_index];
            let output_info = vk::DescriptorImageInfo::default()
                .image_view(output.view)
                .image_layout(vk::ImageLayout::GENERAL);
            let mut texture_infos = Vec::with_capacity(planned.binds.len());
            for image_ref in &pass_bind_refs[pass_index] {
                let image = runtime_image(&input_image, &embedded_images, &pass_images, *image_ref);
                texture_infos.push(
                    vk::DescriptorImageInfo::default()
                        .image_view(image.view)
                        .image_layout(vk::ImageLayout::GENERAL),
                );
            }
            let sampler_infos = pass_bind_refs[pass_index]
                .iter()
                .map(|image_ref| {
                    let bind_sampler = match image_ref {
                        RuntimeImageRef::Embedded(index) => embedded_images[*index]
                            .sampler
                            .unwrap_or(vk::Sampler::null()),
                        _ => sampler,
                    };
                    vk::DescriptorImageInfo::default().sampler(bind_sampler)
                })
                .collect::<Vec<_>>();
            let mut writes = Vec::with_capacity(1 + planned.binds.len() * 2);
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&output_info)),
            );
            let mut next_binding = 1u32;
            for (index, ((bind, _image_ref), info)) in planned
                .binds
                .iter()
                .zip(pass_bind_refs[pass_index].iter())
                .zip(texture_infos.iter())
                .enumerate()
            {
                if bind.storage {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(descriptor_set)
                            .dst_binding(next_binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(std::slice::from_ref(info)),
                    );
                    next_binding += 1;
                    continue;
                }
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(next_binding)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(std::slice::from_ref(info)),
                );
                next_binding += 1;
                if sampler_infos[index].sampler == vk::Sampler::null() {
                    return Err(anyhow!("sampled embedded image has no sampler"));
                }
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(next_binding)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(std::slice::from_ref(&sampler_infos[index])),
                );
                next_binding += 1;
            }
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            runtime_passes.push(RuntimePass {
                descriptor_set_layout,
                pipeline_layout,
                shader_module,
                pipeline,
                descriptor_set,
                output_index,
                width: planned.width,
                height: planned.height,
                dispatch_block_w: planned.dispatch_block_w,
                dispatch_block_h: planned.dispatch_block_h,
            });
        }

        let pack_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let pack_descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&pack_bindings),
                None,
            )
        }
        .context("vkCreateDescriptorSetLayout pack failed")?;
        let pack_spirv =
            compile_pass_to_spirv(&pack_compute_source(plan.final_width, plan.final_height))?;
        let (pack_pipeline_layout, pack_shader_module, pack_pipeline) =
            create_compute_pipeline(&device, &pack_spirv, pack_descriptor_set_layout)?;
        let pack_layouts = [pack_descriptor_set_layout];
        let pack_descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&pack_layouts),
            )
        }
        .context("vkAllocateDescriptorSets pack failed")?[0];
        let final_image = runtime_image(&input_image, &embedded_images, &pass_images, final_ref);
        let pack_output_info = vk::DescriptorImageInfo::default()
            .image_view(pack_image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let pack_texture_info = vk::DescriptorImageInfo::default()
            .image_view(final_image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let pack_sampler_info = vk::DescriptorImageInfo::default().sampler(sampler);
        let pack_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(pack_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&pack_output_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(pack_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(&pack_texture_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(pack_descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(std::slice::from_ref(&pack_sampler_info)),
        ];
        unsafe { device.update_descriptor_sets(&pack_writes, &[]) };

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index),
                None,
            )
        }
        .context("vkCreateCommandPool multi-pass failed")?;
        let command_buffers = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(8),
            )
        }
        .context("vkAllocateCommandBuffers multi-pass failed")?;
        let initial_command_buffer = command_buffers[0];
        let steady_command_buffer = command_buffers[1];
        let initial_external_input_command_buffer = command_buffers[2];
        let steady_external_input_command_buffer = command_buffers[3];
        let initial_external_output_command_buffer = command_buffers[4];
        let steady_external_output_command_buffer = command_buffers[5];
        let initial_external_input_output_command_buffer = command_buffers[6];
        let steady_external_input_output_command_buffer = command_buffers[7];
        record_multipass_commands(
            &device,
            initial_command_buffer,
            false,
            true,
            upload_buffer,
            readback_buffer,
            &input_image,
            &embedded_images,
            &pass_images,
            &runtime_passes,
            &pack_image,
            pack_pipeline,
            pack_pipeline_layout,
            pack_descriptor_set,
            upload_byte_count,
            output_byte_count,
            true,
        )?;
        record_multipass_commands(
            &device,
            steady_command_buffer,
            true,
            true,
            upload_buffer,
            readback_buffer,
            &input_image,
            &embedded_images,
            &pass_images,
            &runtime_passes,
            &pack_image,
            pack_pipeline,
            pack_pipeline_layout,
            pack_descriptor_set,
            upload_byte_count,
            output_byte_count,
            true,
        )?;
        record_multipass_commands(
            &device,
            initial_external_input_command_buffer,
            false,
            false,
            upload_buffer,
            readback_buffer,
            &input_image,
            &embedded_images,
            &pass_images,
            &runtime_passes,
            &pack_image,
            pack_pipeline,
            pack_pipeline_layout,
            pack_descriptor_set,
            upload_byte_count,
            output_byte_count,
            true,
        )?;
        record_multipass_commands(
            &device,
            steady_external_input_command_buffer,
            true,
            false,
            upload_buffer,
            readback_buffer,
            &input_image,
            &embedded_images,
            &pass_images,
            &runtime_passes,
            &pack_image,
            pack_pipeline,
            pack_pipeline_layout,
            pack_descriptor_set,
            upload_byte_count,
            output_byte_count,
            true,
        )?;
        if let Some(external_output) = external_rgba_output.as_ref() {
            for (command_buffer, initialized, upload_input) in [
                (initial_external_output_command_buffer, false, true),
                (steady_external_output_command_buffer, true, true),
                (initial_external_input_output_command_buffer, false, false),
                (steady_external_input_output_command_buffer, true, false),
            ] {
                record_multipass_commands(
                    &device,
                    command_buffer,
                    initialized,
                    upload_input,
                    upload_buffer,
                    external_output.buffer,
                    &input_image,
                    &embedded_images,
                    &pass_images,
                    &runtime_passes,
                    &pack_image,
                    pack_pipeline,
                    pack_pipeline_layout,
                    pack_descriptor_set,
                    upload_byte_count,
                    output_byte_count,
                    false,
                )?;
            }
        }
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .context("vkCreateFence multi-pass failed")?;

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            requested_luid: key.luid,
            device,
            external_memory_win32,
            queue,
            gpu_name,
            input_byte_count,
            output_byte_count,
            upload_buffer,
            upload_memory,
            upload_ptr,
            readback_buffer,
            readback_memory,
            readback_ptr,
            external_rgba_output,
            external_rgba_gl_failed: false,
            input_image,
            embedded_images,
            pass_images,
            pack_image,
            sampler,
            descriptor_pool,
            passes: runtime_passes,
            pack_descriptor_set_layout,
            pack_pipeline_layout,
            pack_shader_module,
            pack_pipeline,
            command_pool,
            initial_command_buffer,
            steady_command_buffer,
            initial_external_input_command_buffer,
            steady_external_input_command_buffer,
            initial_external_output_command_buffer,
            steady_external_output_command_buffer,
            initial_external_input_output_command_buffer,
            steady_external_input_output_command_buffer,
            fence,
            output_width: plan.final_width,
            output_height: plan.final_height,
            submission_in_flight: false,
            image_initialized: false,
            first_frame_reported: false,
            session_reusable,
            external_nchw_inputs: HashMap::new(),
        })
    }

    fn clear_external_nchw_inputs(&mut self) {
        let inputs = std::mem::take(&mut self.external_nchw_inputs);
        unsafe {
            for (_, input) in inputs {
                self.device.free_command_buffers(
                    self.command_pool,
                    &[input.initial_command_buffer, input.steady_command_buffer],
                );
                self.device.destroy_pipeline(input.pipeline, None);
                self.device.destroy_shader_module(input.shader_module, None);
                self.device
                    .destroy_pipeline_layout(input.pipeline_layout, None);
                self.device
                    .destroy_descriptor_pool(input.descriptor_pool, None);
                self.device
                    .destroy_descriptor_set_layout(input.descriptor_set_layout, None);
                self.device.destroy_buffer(input.buffer, None);
                self.device.free_memory(input.memory, None);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_external_nchw_input(
        &mut self,
        shared_key: u64,
        shared_handle: isize,
        byte_len: usize,
        padded_width: u32,
        padded_height: u32,
        fp16: bool,
    ) -> Result<()> {
        let external = self.external_memory_win32.as_ref().ok_or_else(|| {
            anyhow!("VK_KHR_external_memory_win32 is unavailable on selected GPU")
        })?;
        let handle_type = vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE;
        let mut external_props = vk::ExternalBufferProperties::default();
        let external_info = vk::PhysicalDeviceExternalBufferInfo::default()
            .flags(vk::BufferCreateFlags::empty())
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .handle_type(handle_type);
        unsafe {
            self.instance
                .get_physical_device_external_buffer_properties(
                    self.physical_device,
                    &external_info,
                    &mut external_props,
                );
        }
        anyhow::ensure!(
            external_props
                .external_memory_properties
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE),
            "selected Vulkan GPU cannot import D3D12_RESOURCE buffers"
        );

        let mut external_create =
            vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(byte_len.max(1) as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_create);
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None) }
            .context("vkCreateBuffer DirectML import failed")?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mut handle_props = vk::MemoryWin32HandlePropertiesKHR::default();
        if let Err(error) = unsafe {
            external.get_memory_win32_handle_properties(
                handle_type,
                shared_handle,
                &mut handle_props,
            )
        } {
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Err(anyhow!(
                "vkGetMemoryWin32HandlePropertiesKHR failed: {error:?}"
            ));
        }
        let memory_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let compatible_bits = req.memory_type_bits & handle_props.memory_type_bits;
        let memory_type = find_memory_type_with_flags(
            &memory_properties,
            compatible_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| {
            find_memory_type_with_flags(
                &memory_properties,
                compatible_bits,
                vk::MemoryPropertyFlags::empty(),
            )
        })
        .ok_or_else(|| anyhow!("no compatible Vulkan memory type for D3D12 resource import"))?;
        let mut import_info = vk::ImportMemoryWin32HandleInfoKHR::default()
            .handle_type(handle_type)
            .handle(shared_handle);
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(memory_type)
            .push_next(&mut import_info)
            .push_next(&mut dedicated_info);
        let memory = match unsafe { self.device.allocate_memory(&alloc, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(anyhow!("vkAllocateMemory D3D12 import failed: {error:?}"));
            }
        };
        if let Err(error) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
            }
            return Err(anyhow!("vkBindBufferMemory D3D12 import failed: {error:?}"));
        }

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let descriptor_set_layout = unsafe {
            self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .context("vkCreateDescriptorSetLayout DirectML bridge failed")?;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        ];
        let descriptor_pool = unsafe {
            self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .context("vkCreateDescriptorPool DirectML bridge failed")?;
        let layouts = [descriptor_set_layout];
        let descriptor_set = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .context("vkAllocateDescriptorSets DirectML bridge failed")?[0];
        let source_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(byte_len as vk::DeviceSize);
        let dest_info = vk::DescriptorImageInfo::default()
            .image_view(self.input_image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&source_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&dest_info)),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        let source = external_nchw_to_rgba8_compute_source(
            self.input_image.width,
            self.input_image.height,
            padded_width,
            padded_height,
            fp16,
        );
        let spirv = compile_pass_to_spirv(&source)
            .context("DirectML NCHW -> Vulkan RGBA image bridge compile failed")?;
        let (pipeline_layout, shader_module, pipeline) =
            create_compute_pipeline(&self.device, &spirv, descriptor_set_layout)?;
        let command_buffers = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(2),
            )
        }
        .context("vkAllocateCommandBuffers DirectML bridge failed")?;
        let initial_command_buffer = command_buffers[0];
        let steady_command_buffer = command_buffers[1];
        let range = image_range();

        for (command_buffer, initialized) in [
            (initial_command_buffer, false),
            (steady_command_buffer, true),
        ] {
            unsafe {
                self.device
                    .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                    .context("vkBeginCommandBuffer DirectML bridge failed")?;
                let source_before = [vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffer)
                    .offset(0)
                    .size(byte_len as vk::DeviceSize)];
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &source_before,
                    &[],
                );

                let input_before = [vk::ImageMemoryBarrier::default()
                    .src_access_mask(if initialized {
                        vk::AccessFlags::SHADER_READ
                    } else {
                        vk::AccessFlags::empty()
                    })
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .old_layout(if initialized {
                        vk::ImageLayout::GENERAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    })
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.input_image.image)
                    .subresource_range(range)];
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    if initialized {
                        vk::PipelineStageFlags::COMPUTE_SHADER
                    } else {
                        vk::PipelineStageFlags::TOP_OF_PIPE
                    },
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &input_before,
                );
                self.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline_layout,
                    0,
                    &[descriptor_set],
                    &[],
                );
                self.device.cmd_dispatch(
                    command_buffer,
                    self.input_image.width.div_ceil(16),
                    self.input_image.height.div_ceil(8),
                    1,
                );
                let input_ready = [vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.input_image.image)
                    .subresource_range(range)];
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &input_ready,
                );
                self.device
                    .end_command_buffer(command_buffer)
                    .context("vkEndCommandBuffer DirectML bridge failed")?;
            }
        }

        self.external_nchw_inputs.insert(
            shared_key,
            RuntimeExternalNchwInput {
                buffer,
                memory,
                descriptor_pool,
                descriptor_set_layout,
                pipeline_layout,
                shader_module,
                pipeline,
                initial_command_buffer,
                steady_command_buffer,
                byte_len,
                padded_width,
                padded_height,
                fp16,
            },
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn process_external_nchw(
        &mut self,
        shared_key: u64,
        shared_handle: isize,
        byte_len: usize,
        padded_width: u32,
        padded_height: u32,
        fp16: bool,
    ) -> Result<MultiPassProductionResult> {
        let element_bytes = if fp16 { 2usize } else { 4usize };
        let required = (padded_width as usize)
            .checked_mul(padded_height as usize)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_mul(element_bytes))
            .ok_or_else(|| anyhow!("DirectML Vulkan bridge size overflow"))?;
        anyhow::ensure!(
            byte_len >= required,
            "DirectML shared output is too small: {} < {}",
            byte_len,
            required
        );
        let stale = self
            .external_nchw_inputs
            .get(&shared_key)
            .is_some_and(|input| {
                input.byte_len != byte_len
                    || input.padded_width != padded_width
                    || input.padded_height != padded_height
                    || input.fp16 != fp16
            });
        if stale {
            self.clear_external_nchw_inputs();
        }
        if !self.external_nchw_inputs.contains_key(&shared_key) {
            const MAX_EXTERNAL_NCHW_INPUTS: usize = 16;
            if self.external_nchw_inputs.len() >= MAX_EXTERNAL_NCHW_INPUTS {
                log::info!(
                    "vulkan-dml-shared-input-cache: action=trim entries={} limit={} policy=clear-safe-synchronous-cache",
                    self.external_nchw_inputs.len(),
                    MAX_EXTERNAL_NCHW_INPUTS
                );
                self.clear_external_nchw_inputs();
            }
            self.create_external_nchw_input(
                shared_key,
                shared_handle,
                byte_len,
                padded_width,
                padded_height,
                fp16,
            )?;
        }
        let external_command = {
            let imported = self
                .external_nchw_inputs
                .get(&shared_key)
                .ok_or_else(|| anyhow!("DirectML Vulkan imported input disappeared"))?;
            if self.image_initialized {
                imported.steady_command_buffer
            } else {
                imported.initial_command_buffer
            }
        };
        let started = Instant::now();
        unsafe {
            self.device
                .reset_fences(&[self.fence])
                .context("vkResetFences DirectML Vulkan bridge failed")?;
        }
        let main_command = if self.image_initialized {
            self.steady_external_input_command_buffer
        } else {
            self.initial_external_input_command_buffer
        };
        let buffers = [external_command, main_command];
        let submits = [vk::SubmitInfo::default().command_buffers(&buffers)];
        unsafe {
            self.device
                .queue_submit(self.queue, &submits, self.fence)
                .context("vkQueueSubmit DirectML Vulkan bridge failed")?;
        }
        self.submission_in_flight = true;
        let waited = unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, 1_000_000_000)
        };
        if waited.is_ok() {
            self.submission_in_flight = false;
        }
        waited.context("DirectML -> Vulkan resident chain timed out")?;
        self.image_initialized = true;
        let output =
            unsafe { std::slice::from_raw_parts(self.readback_ptr, self.output_byte_count) }
                .to_vec();
        let first_frame_active = !self.first_frame_reported;
        self.first_frame_reported = true;
        Ok(MultiPassProductionResult {
            output_rgba8: output,
            output_width: self.output_width,
            output_height: self.output_height,
            gpu_name: self.gpu_name.clone(),
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            active_passes: self.passes.len(),
            first_frame_active,
        })
    }

    /// Prepare the v667 Vulkan->OpenGL external-buffer handoff.  This route is
    /// intentionally limited to GL drivers that already pass Neo's aggressive
    /// external-memory checks; conservative AMD contexts keep the proven mapped
    /// readback fallback.  The OpenGL fence protects the previous frame's SSBO
    /// read before Vulkan overwrites the shared buffer.
    fn prepare_external_rgba_output_for_gl(&mut self, gc: &mut GlContext) -> Option<u64> {
        // Runtime-cache eviction happens without a GlContext borrow. Retire any
        // stale Vulkan/GL external imports here, on the render thread, before
        // admitting another external output. This prevents live filter/geometry
        // changes from accumulating large 4K/5K external allocations.
        drain_gl_external_output_retire(gc);
        if self.external_rgba_gl_failed || gc.external_interop_conservative_recommended() {
            return None;
        }
        let (key, allocation_byte_len) = {
            let output = self.external_rgba_output.as_ref()?;
            (output.key, output.allocation_byte_len)
        };
        if !gc.has_external_import(key) {
            let handle = match self
                .external_rgba_output
                .as_ref()
                .expect("external output disappeared")
                .shared_handle()
            {
                Ok(handle) => handle,
                Err(error) => {
                    self.external_rgba_gl_failed = true;
                    log::info!(
                        "vulkan-gl-external-output: result=fallback phase=d3d12-share-handle reason={error:#} fallback=mapped-readback"
                    );
                    return None;
                }
            };
            let device_luid = self.requested_luid.to_le_bytes();
            if let Err(error) = gc.import_external_d3d12_buffer(
                key,
                handle,
                self.output_byte_count,
                allocation_byte_len,
                device_luid,
            ) {
                self.external_rgba_gl_failed = true;
                log::info!(
                    "vulkan-gl-external-output: result=fallback phase=gl-import key={} reason={} fallback=mapped-readback",
                    key,
                    error
                );
                return None;
            }
            log::info!(
                "vulkan-gl-external-output: result=active key={} bytes={} allocation_bytes={} transfer=Vulkan-D3D12_RESOURCE-to-OpenGL-SSBO cpu_image_copy=0",
                key,
                self.output_byte_count,
                allocation_byte_len
            );
        }
        if let Err(error) = gc.wait_external_buffer_idle(key) {
            self.external_rgba_gl_failed = true;
            log::info!(
                "vulkan-gl-external-output: result=fallback phase=gl-fence key={} reason={} fallback=mapped-readback",
                key,
                error
            );
            return None;
        }
        Some(key)
    }

    /// v664 same-thread mapped readback -> OpenGL upload. The Vulkan fence is
    /// still authoritative, so GL never samples a partially written frame.
    /// Unlike `process_external_nchw`, this does not clone the complete
    /// readback allocation into a temporary Vec before uploading it.
    fn process_external_nchw_to_gl(
        &mut self,
        shared_key: u64,
        shared_handle: isize,
        byte_len: usize,
        padded_width: u32,
        padded_height: u32,
        fp16: bool,
        gc: &mut GlContext,
    ) -> Result<MultiPassGlUploadResult> {
        let element_bytes = if fp16 { 2usize } else { 4usize };
        let required = (padded_width as usize)
            .checked_mul(padded_height as usize)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_mul(element_bytes))
            .ok_or_else(|| anyhow!("DirectML Vulkan bridge size overflow"))?;
        anyhow::ensure!(
            byte_len >= required,
            "DirectML shared output is too small: {} < {}",
            byte_len,
            required
        );
        let stale = self
            .external_nchw_inputs
            .get(&shared_key)
            .is_some_and(|input| {
                input.byte_len != byte_len
                    || input.padded_width != padded_width
                    || input.padded_height != padded_height
                    || input.fp16 != fp16
            });
        if stale {
            self.clear_external_nchw_inputs();
        }
        if !self.external_nchw_inputs.contains_key(&shared_key) {
            const MAX_EXTERNAL_NCHW_INPUTS: usize = 16;
            if self.external_nchw_inputs.len() >= MAX_EXTERNAL_NCHW_INPUTS {
                log::info!(
                    "vulkan-dml-shared-input-cache: action=trim entries={} limit={} policy=clear-safe-synchronous-cache",
                    self.external_nchw_inputs.len(),
                    MAX_EXTERNAL_NCHW_INPUTS
                );
                self.clear_external_nchw_inputs();
            }
            self.create_external_nchw_input(
                shared_key,
                shared_handle,
                byte_len,
                padded_width,
                padded_height,
                fp16,
            )?;
        }
        let external_command = {
            let imported = self
                .external_nchw_inputs
                .get(&shared_key)
                .ok_or_else(|| anyhow!("DirectML Vulkan imported input disappeared"))?;
            if self.image_initialized {
                imported.steady_command_buffer
            } else {
                imported.initial_command_buffer
            }
        };
        let call_started = Instant::now();
        let sync_started = Instant::now();
        let external_output_key = self.prepare_external_rgba_output_for_gl(gc);
        let external_sync_ms = sync_started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        unsafe {
            self.device
                .reset_fences(&[self.fence])
                .context("vkResetFences DirectML Vulkan bridge failed")?;
        }
        let main_command = if external_output_key.is_some() {
            if self.image_initialized {
                self.steady_external_input_output_command_buffer
            } else {
                self.initial_external_input_output_command_buffer
            }
        } else if self.image_initialized {
            self.steady_external_input_command_buffer
        } else {
            self.initial_external_input_command_buffer
        };
        let buffers = [external_command, main_command];
        let submits = [vk::SubmitInfo::default().command_buffers(&buffers)];
        unsafe {
            self.device
                .queue_submit(self.queue, &submits, self.fence)
                .context("vkQueueSubmit DirectML Vulkan bridge failed")?;
        }
        self.submission_in_flight = true;
        let waited = unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, 1_000_000_000)
        };
        if waited.is_ok() {
            self.submission_in_flight = false;
        }
        waited.context("DirectML -> Vulkan resident chain timed out")?;
        self.image_initialized = true;
        let vulkan_ms = started.elapsed().as_secs_f64() * 1000.0;

        if let Some(output_key) = external_output_key {
            let gl_started = Instant::now();
            match gc.external_rgba8_buffer_to_texture(
                output_key,
                self.output_width as i32,
                self.output_height as i32,
            ) {
                Ok(output) => {
                    let gl_upload_ms = gl_started.elapsed().as_secs_f64() * 1000.0;
                    let first_frame_active = !self.first_frame_reported;
                    self.first_frame_reported = true;
                    return Ok(MultiPassGlUploadResult {
                        output,
                        output_width: self.output_width,
                        output_height: self.output_height,
                        gpu_name: self.gpu_name.clone(),
                        vulkan_ms,
                        gl_upload_ms,
                        output_external_buffer: true,
                        external_sync_ms,
                        elapsed_ms: call_started.elapsed().as_secs_f64() * 1000.0,
                        active_passes: self.passes.len(),
                        first_frame_active,
                    });
                }
                Err(error) => {
                    self.external_rgba_gl_failed = true;
                    log::warn!(
                        "vulkan-gl-external-output: result=fallback phase=gl-convert key={} reason={} action=rerun-mapped-readback",
                        output_key,
                        error
                    );
                    // Recover the current frame immediately. The imported NCHW
                    // input has already been converted into input_image; rerun
                    // only the shader/pack command into the proven readback buffer.
                    unsafe {
                        self.device
                            .reset_fences(&[self.fence])
                            .context("vkResetFences Vulkan mapped fallback failed")?;
                    }
                    let fallback = self.steady_external_input_command_buffer;
                    let buffers = [fallback];
                    let submits = [vk::SubmitInfo::default().command_buffers(&buffers)];
                    unsafe {
                        self.device
                            .queue_submit(self.queue, &submits, self.fence)
                            .context("vkQueueSubmit Vulkan mapped fallback failed")?;
                    }
                    self.submission_in_flight = true;
                    let waited = unsafe {
                        self.device
                            .wait_for_fences(&[self.fence], true, 1_000_000_000)
                    };
                    if waited.is_ok() {
                        self.submission_in_flight = false;
                    }
                    waited.context("Vulkan mapped fallback timed out")?;
                }
            }
        }

        let upload_started = Instant::now();
        let readback =
            unsafe { std::slice::from_raw_parts(self.readback_ptr, self.output_byte_count) };
        let output = gc.upload_rgba8(
            self.output_width as i32,
            self.output_height as i32,
            readback,
        );
        let gl_upload_ms = upload_started.elapsed().as_secs_f64() * 1000.0;
        let first_frame_active = !self.first_frame_reported;
        self.first_frame_reported = true;
        Ok(MultiPassGlUploadResult {
            output,
            output_width: self.output_width,
            output_height: self.output_height,
            gpu_name: self.gpu_name.clone(),
            vulkan_ms,
            gl_upload_ms,
            output_external_buffer: false,
            external_sync_ms,
            elapsed_ms: call_started.elapsed().as_secs_f64() * 1000.0,
            active_passes: self.passes.len(),
            first_frame_active,
        })
    }

    /// v664 clone-free CPU-input Vulkan bridge. The input still comes from the
    /// established GL readback path, but the 4K/5K Vulkan result is uploaded to
    /// OpenGL directly from persistently mapped readback memory.
    fn process_to_gl(
        &mut self,
        input: &[u8],
        gc: &mut GlContext,
    ) -> Result<MultiPassGlUploadResult> {
        if input.len() != self.input_byte_count {
            return Err(anyhow!(
                "multi-pass input byte mismatch: got {}, expected {}",
                input.len(),
                self.input_byte_count
            ));
        }
        let call_started = Instant::now();
        let sync_started = Instant::now();
        let external_output_key = self.prepare_external_rgba_output_for_gl(gc);
        let external_sync_ms = sync_started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), self.upload_ptr, input.len());
            self.device
                .reset_fences(&[self.fence])
                .context("vkResetFences multi-pass failed")?;
        }
        let command_buffer = if external_output_key.is_some() {
            if self.image_initialized {
                self.steady_external_output_command_buffer
            } else {
                self.initial_external_output_command_buffer
            }
        } else if self.image_initialized {
            self.steady_command_buffer
        } else {
            self.initial_command_buffer
        };
        let buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&buffers)];
        unsafe {
            self.device
                .queue_submit(self.queue, &submits, self.fence)
                .context("vkQueueSubmit multi-pass failed")?;
        }
        self.submission_in_flight = true;
        let waited = unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, 1_000_000_000)
        };
        if waited.is_ok() {
            self.submission_in_flight = false;
        }
        waited.context("Vulkan multi-pass timed out")?;
        self.image_initialized = true;
        let vulkan_ms = started.elapsed().as_secs_f64() * 1000.0;

        if let Some(output_key) = external_output_key {
            let gl_started = Instant::now();
            match gc.external_rgba8_buffer_to_texture(
                output_key,
                self.output_width as i32,
                self.output_height as i32,
            ) {
                Ok(output) => {
                    let gl_upload_ms = gl_started.elapsed().as_secs_f64() * 1000.0;
                    let first_frame_active = !self.first_frame_reported;
                    self.first_frame_reported = true;
                    return Ok(MultiPassGlUploadResult {
                        output,
                        output_width: self.output_width,
                        output_height: self.output_height,
                        gpu_name: self.gpu_name.clone(),
                        vulkan_ms,
                        gl_upload_ms,
                        output_external_buffer: true,
                        external_sync_ms,
                        elapsed_ms: call_started.elapsed().as_secs_f64() * 1000.0,
                        active_passes: self.passes.len(),
                        first_frame_active,
                    });
                }
                Err(error) => {
                    self.external_rgba_gl_failed = true;
                    log::warn!(
                        "vulkan-gl-external-output: result=fallback phase=gl-convert key={} reason={} action=rerun-mapped-readback",
                        output_key,
                        error
                    );
                    unsafe {
                        self.device
                            .reset_fences(&[self.fence])
                            .context("vkResetFences Vulkan mapped fallback failed")?;
                    }
                    let fallback = self.steady_command_buffer;
                    let buffers = [fallback];
                    let submits = [vk::SubmitInfo::default().command_buffers(&buffers)];
                    unsafe {
                        self.device
                            .queue_submit(self.queue, &submits, self.fence)
                            .context("vkQueueSubmit Vulkan mapped fallback failed")?;
                    }
                    self.submission_in_flight = true;
                    let waited = unsafe {
                        self.device
                            .wait_for_fences(&[self.fence], true, 1_000_000_000)
                    };
                    if waited.is_ok() {
                        self.submission_in_flight = false;
                    }
                    waited.context("Vulkan mapped fallback timed out")?;
                }
            }
        }

        let upload_started = Instant::now();
        let readback =
            unsafe { std::slice::from_raw_parts(self.readback_ptr, self.output_byte_count) };
        let output = gc.upload_rgba8(
            self.output_width as i32,
            self.output_height as i32,
            readback,
        );
        let gl_upload_ms = upload_started.elapsed().as_secs_f64() * 1000.0;
        let first_frame_active = !self.first_frame_reported;
        self.first_frame_reported = true;
        Ok(MultiPassGlUploadResult {
            output,
            output_width: self.output_width,
            output_height: self.output_height,
            gpu_name: self.gpu_name.clone(),
            vulkan_ms,
            gl_upload_ms,
            output_external_buffer: false,
            external_sync_ms,
            elapsed_ms: call_started.elapsed().as_secs_f64() * 1000.0,
            active_passes: self.passes.len(),
            first_frame_active,
        })
    }

    fn process(&mut self, input: &[u8]) -> Result<MultiPassProductionResult> {
        if input.len() != self.input_byte_count {
            return Err(anyhow!(
                "multi-pass input byte mismatch: got {}, expected {}",
                input.len(),
                self.input_byte_count
            ));
        }
        let started = Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), self.upload_ptr, input.len());
            self.device
                .reset_fences(&[self.fence])
                .context("vkResetFences multi-pass failed")?;
        }
        let command_buffer = if self.image_initialized {
            self.steady_command_buffer
        } else {
            self.initial_command_buffer
        };
        let buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&buffers)];
        unsafe {
            self.device
                .queue_submit(self.queue, &submits, self.fence)
                .context("vkQueueSubmit multi-pass failed")?;
        }
        self.submission_in_flight = true;
        let waited = unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, 1_000_000_000)
        };
        if waited.is_ok() {
            self.submission_in_flight = false;
        }
        waited.context("Vulkan multi-pass timed out")?;
        self.image_initialized = true;
        let output =
            unsafe { std::slice::from_raw_parts(self.readback_ptr, self.output_byte_count) }
                .to_vec();
        let first_frame_active = !self.first_frame_reported;
        self.first_frame_reported = true;
        Ok(MultiPassProductionResult {
            output_rgba8: output,
            output_width: self.output_width,
            output_height: self.output_height,
            gpu_name: self.gpu_name.clone(),
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            active_passes: self.passes.len(),
            first_frame_active,
        })
    }
}

impl Drop for VulkanMultiPassRuntime {
    fn drop(&mut self) {
        if ABANDON_MULTIPASS_ON_PROCESS_EXIT.load(Ordering::Acquire) || self.submission_in_flight {
            return;
        }
        self.clear_external_nchw_inputs();
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pack_pipeline, None);
            self.device
                .destroy_shader_module(self.pack_shader_module, None);
            self.device
                .destroy_pipeline_layout(self.pack_pipeline_layout, None);
            for pass in self.passes.iter().rev() {
                self.device.destroy_pipeline(pass.pipeline, None);
                self.device.destroy_shader_module(pass.shader_module, None);
                self.device
                    .destroy_pipeline_layout(pass.pipeline_layout, None);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.pack_descriptor_set_layout, None);
            for pass in self.passes.iter().rev() {
                self.device
                    .destroy_descriptor_set_layout(pass.descriptor_set_layout, None);
            }
            self.device.destroy_sampler(self.sampler, None);
            for embedded in self.embedded_images.iter().rev() {
                if let Some(sampler) = embedded.sampler {
                    self.device.destroy_sampler(sampler, None);
                }
                self.device.destroy_image_view(embedded.image.view, None);
                self.device.destroy_image(embedded.image.image, None);
                self.device.free_memory(embedded.image.memory, None);
            }
            for image in self.pass_images.iter().rev() {
                self.device.destroy_image_view(image.view, None);
                self.device.destroy_image(image.image, None);
                self.device.free_memory(image.memory, None);
            }
            self.device.destroy_image_view(self.pack_image.view, None);
            self.device.destroy_image(self.pack_image.image, None);
            self.device.free_memory(self.pack_image.memory, None);
            self.device.destroy_image_view(self.input_image.view, None);
            self.device.destroy_image(self.input_image.image, None);
            self.device.free_memory(self.input_image.memory, None);
            self.device.unmap_memory(self.upload_memory);
            self.device.unmap_memory(self.readback_memory);
            self.device.destroy_buffer(self.upload_buffer, None);
            self.device.destroy_buffer(self.readback_buffer, None);
            if let Some(output) = self.external_rgba_output.take() {
                // The GL import owns an NT-handle reference to the payload.
                // Queue its GL-side retirement for the next render-thread
                // handoff before destroying this runtime's Vulkan objects.
                pending_gl_external_output_retire()
                    .lock()
                    .unwrap()
                    .push(output.key);
                self.device.destroy_buffer(output.buffer, None);
                self.device.free_memory(output.memory, None);
            }
            self.device.free_memory(self.upload_memory, None);
            self.device.free_memory(self.readback_memory, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub fn process_rgba8(
    requested_luid: u64,
    shader: &UserShader,
    input_width: u32,
    input_height: u32,
    output_ref_width: u32,
    output_ref_height: u32,
    input: &[u8],
) -> Result<Option<MultiPassProductionResult>> {
    if !production_requested() || !production_shader_admitted(shader) {
        return Ok(None);
    }
    let key = RuntimeKey {
        luid: requested_luid,
        input_width: input_width.max(1),
        input_height: input_height.max(1),
        output_ref_width: output_ref_width.max(1),
        output_ref_height: output_ref_height.max(1),
        shader_hash: shader_runtime_hash(shader),
    };
    if multipass_failed_keys().lock().unwrap().contains(&key) {
        return Ok(None);
    }

    MULTIPASS_RUNTIMES.with(|slot| -> Result<Option<MultiPassProductionResult>> {
        let mut cache = slot.borrow_mut();
        if !cache.runtimes.contains_key(&key) {
            // Compile each shader/geometry runtime only once. v635 held a single
            // runtime, which was fine for its first-GLSL-only route but would
            // thrash (compile/create every frame) once a mixed chain routes all
            // GLSL stages through Vulkan.
            let compile_started = Instant::now();
            let plan = match build_multipass_plan(
                shader,
                key.input_width,
                key.input_height,
                key.output_ref_width,
                key.output_ref_height,
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            };
            let compile_ms = compile_started.elapsed().as_secs_f64() * 1000.0;
            let compile_words = plan.total_spirv_words;
            let pass_count = plan.passes.len();
            let final_size = (plan.final_width, plan.final_height);
            let create_started = Instant::now();
            match VulkanMultiPassRuntime::create(key.clone(), plan) {
                Ok(runtime) => {
                    let line = format!(
                        "vulkan-multipass-glsl: phase=runtime-create shader='{}' requested_luid={:016x} passes={} spirv_words={} final={}x{} compile_ms={:.3} create_ms={:.3} intermediate=fp16 cache=multi-runtime-lru",
                        shader.name(), requested_luid, pass_count, compile_words,
                        final_size.0, final_size.1, compile_ms, create_started.elapsed().as_secs_f64() * 1000.0
                    );
                    log::info!("{line}");
                    super::vulkan_gpu::record_probe_result(&line);
                    cache.prepare_insert(&key);
                    cache.runtimes.insert(key.clone(), runtime);
                }
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            }
        }
        cache.touch(&key);
        let processed = {
            let runtime = cache
                .runtimes
                .get_mut(&key)
                .ok_or_else(|| anyhow!("multi-pass runtime missing after create"))?;
            runtime.process(input)
        };
        match processed {
            Ok(result) => Ok(Some(result)),
            Err(error) => {
                // A runtime that failed during execution must not be retried on
                // every frame. Retire it for this shader/geometry key and keep
                // the established OpenGL fallback stable for the session.
                cache.runtimes.remove(&key);
                if let Some(pos) = cache.lru.iter().position(|candidate| candidate == &key) {
                    cache.lru.remove(pos);
                }
                multipass_failed_keys().lock().unwrap().insert(key.clone());
                Err(error)
            }
        }
    })
}

/// Execute two or more consecutive mpv shaders inside one Vulkan runtime.
/// Each shader is first lowered with its ordinary standalone semantics, then
/// the plans are chained on-GPU.  This removes the v661 per-stage
/// Vulkan->CPU->OpenGL->CPU->Vulkan bridge that becomes dominant at 4K/5K.
pub fn process_rgba8_sequence(
    requested_luid: u64,
    shaders: &[&UserShader],
    input_width: u32,
    input_height: u32,
    output_ref_width: u32,
    output_ref_height: u32,
    input: &[u8],
) -> Result<Option<MultiPassProductionResult>> {
    if shaders.len() < 2
        || !production_requested()
        || shaders
            .iter()
            .any(|shader| !production_shader_admitted(shader))
    {
        return Ok(None);
    }

    let mut sequence_hasher = DefaultHasher::new();
    "NeoVulkanSequenceV664".hash(&mut sequence_hasher);
    shaders.len().hash(&mut sequence_hasher);
    for shader in shaders {
        shader_runtime_hash(shader).hash(&mut sequence_hasher);
    }
    let sequence_hash = sequence_hasher.finish();
    let key = RuntimeKey {
        luid: requested_luid,
        input_width: input_width.max(1),
        input_height: input_height.max(1),
        output_ref_width: output_ref_width.max(1),
        output_ref_height: output_ref_height.max(1),
        shader_hash: sequence_hash,
    };
    if multipass_failed_keys().lock().unwrap().contains(&key) {
        return Ok(None);
    }

    MULTIPASS_RUNTIMES.with(|slot| -> Result<Option<MultiPassProductionResult>> {
        let mut cache = slot.borrow_mut();
        if !cache.runtimes.contains_key(&key) {
            let compile_started = Instant::now();
            let plan = match build_multipass_sequence_plan(
                shaders,
                key.input_width,
                key.input_height,
                key.output_ref_width,
                key.output_ref_height,
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            };
            let compile_ms = compile_started.elapsed().as_secs_f64() * 1000.0;
            let compile_words = plan.total_spirv_words;
            let pass_count = plan.passes.len();
            let final_size = (plan.final_width, plan.final_height);
            let shader_name = plan.shader_name.clone();
            let create_started = Instant::now();
            match VulkanMultiPassRuntime::create(key.clone(), plan) {
                Ok(runtime) => {
                    let line = format!(
                        "vulkan-sequence-chain: phase=runtime-create shader='{}' requested_luid={:016x} stages={} passes={} spirv_words={} final={}x{} compile_ms={:.3} create_ms={:.3} intermediate=fp16 cache=multi-runtime-lru",
                        shader_name,
                        requested_luid,
                        shaders.len(),
                        pass_count,
                        compile_words,
                        final_size.0,
                        final_size.1,
                        compile_ms,
                        create_started.elapsed().as_secs_f64() * 1000.0,
                    );
                    log::info!("{line}");
                    super::vulkan_gpu::record_probe_result(&line);
                    cache.prepare_insert(&key);
                    cache.runtimes.insert(key.clone(), runtime);
                }
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            }
        }
        cache.touch(&key);
        let processed = {
            let runtime = cache
                .runtimes
                .get_mut(&key)
                .ok_or_else(|| anyhow!("Vulkan sequence runtime missing after create"))?;
            runtime.process(input)
        };
        match processed {
            Ok(result) => Ok(Some(result)),
            Err(error) => {
                cache.runtimes.remove(&key);
                if let Some(pos) = cache.lru.iter().position(|candidate| candidate == &key) {
                    cache.lru.remove(pos);
                }
                multipass_failed_keys().lock().unwrap().insert(key.clone());
                Err(error)
            }
        }
    })
}

/// v664 clone-free variant of `process_rgba8_sequence`.  The Vulkan readback
/// allocation is persistently mapped; upload it to the current OpenGL context
/// directly instead of copying a 4K/5K frame into an intermediate Vec first.
pub fn process_rgba8_sequence_to_gl(
    requested_luid: u64,
    shaders: &[&UserShader],
    input_width: u32,
    input_height: u32,
    output_ref_width: u32,
    output_ref_height: u32,
    input: &[u8],
    gc: &mut GlContext,
) -> Result<Option<MultiPassGlUploadResult>> {
    if shaders.len() < 2
        || !production_requested()
        || shaders
            .iter()
            .any(|shader| !production_shader_admitted(shader))
    {
        return Ok(None);
    }

    let mut sequence_hasher = DefaultHasher::new();
    "NeoVulkanSequenceV664".hash(&mut sequence_hasher);
    shaders.len().hash(&mut sequence_hasher);
    for shader in shaders {
        shader_runtime_hash(shader).hash(&mut sequence_hasher);
    }
    let sequence_hash = sequence_hasher.finish();
    let key = RuntimeKey {
        luid: requested_luid,
        input_width: input_width.max(1),
        input_height: input_height.max(1),
        output_ref_width: output_ref_width.max(1),
        output_ref_height: output_ref_height.max(1),
        shader_hash: sequence_hash,
    };
    if multipass_failed_keys().lock().unwrap().contains(&key) {
        return Ok(None);
    }

    MULTIPASS_RUNTIMES.with(|slot| -> Result<Option<MultiPassGlUploadResult>> {
        let mut cache = slot.borrow_mut();
        if !cache.runtimes.contains_key(&key) {
            let compile_started = Instant::now();
            let plan = match build_multipass_sequence_plan(
                shaders,
                key.input_width,
                key.input_height,
                key.output_ref_width,
                key.output_ref_height,
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            };
            let compile_ms = compile_started.elapsed().as_secs_f64() * 1000.0;
            let compile_words = plan.total_spirv_words;
            let pass_count = plan.passes.len();
            let final_size = (plan.final_width, plan.final_height);
            let shader_name = plan.shader_name.clone();
            let create_started = Instant::now();
            match VulkanMultiPassRuntime::create(key.clone(), plan) {
                Ok(runtime) => {
                    let line = format!(
                        "vulkan-sequence-chain: phase=runtime-create shader='{}' requested_luid={:016x} stages={} passes={} spirv_words={} final={}x{} compile_ms={:.3} create_ms={:.3} intermediate=fp16 cache=multi-runtime-lru workgroup=16x8",
                        shader_name,
                        requested_luid,
                        shaders.len(),
                        pass_count,
                        compile_words,
                        final_size.0,
                        final_size.1,
                        compile_ms,
                        create_started.elapsed().as_secs_f64() * 1000.0,
                    );
                    log::info!("{line}");
                    super::vulkan_gpu::record_probe_result(&line);
                    cache.prepare_insert(&key);
                    cache.runtimes.insert(key.clone(), runtime);
                }
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            }
        }
        cache.touch(&key);
        let processed = {
            let runtime = cache
                .runtimes
                .get_mut(&key)
                .ok_or_else(|| anyhow!("Vulkan sequence runtime missing after create"))?;
            runtime.process_to_gl(input, gc)
        };
        match processed {
            Ok(result) => Ok(Some(result)),
            Err(error) => {
                cache.runtimes.remove(&key);
                if let Some(pos) = cache.lru.iter().position(|candidate| candidate == &key) {
                    cache.lru.remove(pos);
                }
                multipass_failed_keys().lock().unwrap().insert(key.clone());
                Err(error)
            }
        }
    })
}

/// Execute the production multi-pass chain from a DirectML-owned shareable
/// D3D12 NCHW buffer on the same physical GPU as the selected Vulkan device.
/// The D3D12 handle is imported directly as VkDeviceMemory; a Vulkan compute
/// prepass converts NCHW FP16/FP32 straight into the device-local RGBA input
/// image, then the normal resident multi-pass command buffer continues without
/// a CPU copy or an intermediate HOST_VISIBLE buffer round trip.
#[allow(clippy::too_many_arguments)]
pub fn process_d3d12_nchw(
    requested_luid: u64,
    shader: &UserShader,
    input_width: u32,
    input_height: u32,
    output_ref_width: u32,
    output_ref_height: u32,
    shared_key: u64,
    shared_handle: isize,
    shared_byte_len: usize,
    padded_width: u32,
    padded_height: u32,
    fp16: bool,
) -> Result<Option<MultiPassProductionResult>> {
    if !production_requested() || !production_shader_admitted(shader) {
        return Ok(None);
    }
    let key = RuntimeKey {
        luid: requested_luid,
        input_width: input_width.max(1),
        input_height: input_height.max(1),
        output_ref_width: output_ref_width.max(1),
        output_ref_height: output_ref_height.max(1),
        shader_hash: shader_runtime_hash(shader),
    };
    if multipass_failed_keys().lock().unwrap().contains(&key)
        || d3d12_import_failed_keys().lock().unwrap().contains(&key)
    {
        return Ok(None);
    }

    MULTIPASS_RUNTIMES.with(|slot| -> Result<Option<MultiPassProductionResult>> {
        let mut cache = slot.borrow_mut();
        if !cache.runtimes.contains_key(&key) {
            let compile_started = Instant::now();
            let plan = match build_multipass_plan(
                shader,
                key.input_width,
                key.input_height,
                key.output_ref_width,
                key.output_ref_height,
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            };
            let compile_ms = compile_started.elapsed().as_secs_f64() * 1000.0;
            let compile_words = plan.total_spirv_words;
            let pass_count = plan.passes.len();
            let final_size = (plan.final_width, plan.final_height);
            let create_started = Instant::now();
            match VulkanMultiPassRuntime::create(key.clone(), plan) {
                Ok(runtime) => {
                    let line = format!(
                        "vulkan-multipass-glsl: phase=runtime-create shader='{}' requested_luid={:016x} passes={} spirv_words={} final={}x{} compile_ms={:.3} create_ms={:.3} intermediate=fp16 cache=multi-runtime-lru",
                        shader.name(), requested_luid, pass_count, compile_words,
                        final_size.0, final_size.1, compile_ms, create_started.elapsed().as_secs_f64() * 1000.0
                    );
                    log::info!("{line}");
                    super::vulkan_gpu::record_probe_result(&line);
                    cache.prepare_insert(&key);
                    cache.runtimes.insert(key.clone(), runtime);
                }
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            }
        }
        cache.touch(&key);
        let processed = {
            let runtime = cache
                .runtimes
                .get_mut(&key)
                .ok_or_else(|| anyhow!("multi-pass runtime missing after create"))?;
            runtime.process_external_nchw(
                shared_key,
                shared_handle,
                shared_byte_len,
                padded_width.max(1),
                padded_height.max(1),
                fp16,
            )
        };
        match processed {
            Ok(result) => Ok(Some(result)),
            Err(error) => {
                d3d12_import_failed_keys().lock().unwrap().insert(key.clone());
                log::warn!(
                    "vulkan-dml-shared-input: result=fallback requested_luid={:016x} reason={:#} fallback=cpu-staging",
                    requested_luid,
                    error
                );
                Ok(None)
            }
        }
    })
}

/// v664 clone-free DirectML -> Vulkan -> OpenGL path.  The leading DirectML
/// tensor is still imported directly as D3D12 external memory, and the final
/// Vulkan readback mapping is fed straight to OpenGL without a temporary Vec.
#[allow(clippy::too_many_arguments)]
pub fn process_d3d12_nchw_to_gl(
    requested_luid: u64,
    shader: &UserShader,
    input_width: u32,
    input_height: u32,
    output_ref_width: u32,
    output_ref_height: u32,
    shared_key: u64,
    shared_handle: isize,
    shared_byte_len: usize,
    padded_width: u32,
    padded_height: u32,
    fp16: bool,
    gc: &mut GlContext,
) -> Result<Option<MultiPassGlUploadResult>> {
    if !production_requested() || !production_shader_admitted(shader) {
        return Ok(None);
    }
    let key = RuntimeKey {
        luid: requested_luid,
        input_width: input_width.max(1),
        input_height: input_height.max(1),
        output_ref_width: output_ref_width.max(1),
        output_ref_height: output_ref_height.max(1),
        shader_hash: shader_runtime_hash(shader),
    };
    if multipass_failed_keys().lock().unwrap().contains(&key)
        || d3d12_import_failed_keys().lock().unwrap().contains(&key)
    {
        return Ok(None);
    }

    MULTIPASS_RUNTIMES.with(|slot| -> Result<Option<MultiPassGlUploadResult>> {
        let mut cache = slot.borrow_mut();
        if !cache.runtimes.contains_key(&key) {
            let compile_started = Instant::now();
            let plan = match build_multipass_plan(
                shader,
                key.input_width,
                key.input_height,
                key.output_ref_width,
                key.output_ref_height,
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            };
            let compile_ms = compile_started.elapsed().as_secs_f64() * 1000.0;
            let compile_words = plan.total_spirv_words;
            let pass_count = plan.passes.len();
            let final_size = (plan.final_width, plan.final_height);
            let create_started = Instant::now();
            match VulkanMultiPassRuntime::create(key.clone(), plan) {
                Ok(runtime) => {
                    let line = format!(
                        "vulkan-multipass-glsl: phase=runtime-create shader='{}' requested_luid={:016x} passes={} spirv_words={} final={}x{} compile_ms={:.3} create_ms={:.3} intermediate=fp16 cache=multi-runtime-lru workgroup=16x8",
                        shader.name(), requested_luid, pass_count, compile_words,
                        final_size.0, final_size.1, compile_ms,
                        create_started.elapsed().as_secs_f64() * 1000.0
                    );
                    log::info!("{line}");
                    super::vulkan_gpu::record_probe_result(&line);
                    cache.prepare_insert(&key);
                    cache.runtimes.insert(key.clone(), runtime);
                }
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            }
        }
        cache.touch(&key);
        let processed = {
            let runtime = cache
                .runtimes
                .get_mut(&key)
                .ok_or_else(|| anyhow!("multi-pass runtime missing after create"))?;
            runtime.process_external_nchw_to_gl(
                shared_key,
                shared_handle,
                shared_byte_len,
                padded_width.max(1),
                padded_height.max(1),
                fp16,
                gc,
            )
        };
        match processed {
            Ok(result) => Ok(Some(result)),
            Err(error) => {
                d3d12_import_failed_keys().lock().unwrap().insert(key.clone());
                log::warn!(
                    "vulkan-dml-shared-input: result=fallback requested_luid={:016x} reason={:#} fallback=cpu-staging",
                    requested_luid,
                    error
                );
                Ok(None)
            }
        }
    })
}

/// v665 DirectML D3D12 shared NCHW -> Vulkan sequence -> OpenGL path.
///
/// v664 could import DirectML directly only when the following GLSL range
/// collapsed into the older conservative resident-batch form.  LUMA-based
/// shaders such as FSRCNNX are valid Vulkan shaders but intentionally do not
/// collapse that way, so ONNX + FSRCNNX (+ RGB post shaders) still fell back
/// through OpenGL/CPU.  Build the same boundary-preserving sequence plan used
/// by `process_rgba8_sequence_to_gl`, then feed the DirectML allocation into
/// that runtime's external-NCHW input.
#[allow(clippy::too_many_arguments)]
pub fn process_d3d12_nchw_sequence_to_gl(
    requested_luid: u64,
    shaders: &[&UserShader],
    input_width: u32,
    input_height: u32,
    output_ref_width: u32,
    output_ref_height: u32,
    shared_key: u64,
    shared_handle: isize,
    shared_byte_len: usize,
    padded_width: u32,
    padded_height: u32,
    fp16: bool,
    gc: &mut GlContext,
) -> Result<Option<MultiPassGlUploadResult>> {
    if shaders.len() < 2
        || !production_requested()
        || shaders
            .iter()
            .any(|shader| !production_shader_admitted(shader))
    {
        return Ok(None);
    }

    // Deliberately share the v664 CPU-input sequence key.  The Vulkan runtime
    // owns both ordinary upload command buffers and the external-NCHW bridge,
    // so a runtime already warmed by one route can be reused by the other.
    let mut sequence_hasher = DefaultHasher::new();
    "NeoVulkanSequenceV664".hash(&mut sequence_hasher);
    shaders.len().hash(&mut sequence_hasher);
    for shader in shaders {
        shader_runtime_hash(shader).hash(&mut sequence_hasher);
    }
    let sequence_hash = sequence_hasher.finish();
    let key = RuntimeKey {
        luid: requested_luid,
        input_width: input_width.max(1),
        input_height: input_height.max(1),
        output_ref_width: output_ref_width.max(1),
        output_ref_height: output_ref_height.max(1),
        shader_hash: sequence_hash,
    };
    if multipass_failed_keys().lock().unwrap().contains(&key)
        || d3d12_import_failed_keys().lock().unwrap().contains(&key)
    {
        return Ok(None);
    }

    MULTIPASS_RUNTIMES.with(|slot| -> Result<Option<MultiPassGlUploadResult>> {
        let mut cache = slot.borrow_mut();
        if !cache.runtimes.contains_key(&key) {
            let compile_started = Instant::now();
            let plan = match build_multipass_sequence_plan(
                shaders,
                key.input_width,
                key.input_height,
                key.output_ref_width,
                key.output_ref_height,
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            };
            let compile_ms = compile_started.elapsed().as_secs_f64() * 1000.0;
            let compile_words = plan.total_spirv_words;
            let pass_count = plan.passes.len();
            let final_size = (plan.final_width, plan.final_height);
            let shader_name = plan.shader_name.clone();
            let create_started = Instant::now();
            match VulkanMultiPassRuntime::create(key.clone(), plan) {
                Ok(runtime) => {
                    let line = format!(
                        "vulkan-dml-sequence-chain: phase=runtime-create shader='{}' requested_luid={:016x} stages={} passes={} spirv_words={} final={}x{} compile_ms={:.3} create_ms={:.3} intermediate=fp16 cache=multi-runtime-lru workgroup=16x8",
                        shader_name,
                        requested_luid,
                        shaders.len(),
                        pass_count,
                        compile_words,
                        final_size.0,
                        final_size.1,
                        compile_ms,
                        create_started.elapsed().as_secs_f64() * 1000.0,
                    );
                    log::info!("{line}");
                    super::vulkan_gpu::record_probe_result(&line);
                    cache.prepare_insert(&key);
                    cache.runtimes.insert(key.clone(), runtime);
                }
                Err(error) => {
                    multipass_failed_keys().lock().unwrap().insert(key.clone());
                    return Err(error);
                }
            }
        }
        cache.touch(&key);
        let processed = {
            let runtime = cache
                .runtimes
                .get_mut(&key)
                .ok_or_else(|| anyhow!("Vulkan DML sequence runtime missing after create"))?;
            runtime.process_external_nchw_to_gl(
                shared_key,
                shared_handle,
                shared_byte_len,
                padded_width.max(1),
                padded_height.max(1),
                fp16,
                gc,
            )
        };
        match processed {
            Ok(result) => Ok(Some(result)),
            Err(error) => {
                d3d12_import_failed_keys().lock().unwrap().insert(key.clone());
                log::warn!(
                    "vulkan-dml-sequence-input: result=fallback requested_luid={:016x} reason={:#} fallback=cpu-staging",
                    requested_luid,
                    error
                );
                Ok(None)
            }
        }
    })
}
