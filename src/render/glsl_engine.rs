//! mpv user-shader interpreter on glow (no mpv/libplacebo dependency).
//!
//! Compatibility rules used by the shader interpreter:
//! - every BIND name N gets a prelude: N_tx/N_size/N_pt uniforms and
//!   N_pos/N_tex/N_texOff macros in NORMALIZED coords (pos = v_uv for all
//!   binds  Empv normalizes coordinates, which is what makes this simple).
//! - texture() results are swizzled by the bound texture's component count
//!   (LUMA(1ch) -> float, MAIN -> vec4). Wrong typing breaks FSRCNNX/hdeband.
//! - RGB/MAIN shaders run on the colour image; LUMA-only shaders (FSRCNNX)
//!   run on an extracted Y plane and are recombined by luma substitution.
//! - intermediates: F16 textures, NEAREST, clamp-to-edge.

use super::gl::{Dtype, GlContext, GpuTex};
use super::mpv::{ComputeSpec, ParamTy, Params, Pass, Sizes, UserShader, eval_rpn_p};
use anyhow::{Result, anyhow};
use glow::HasContext;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// BT.709 luma coefficients
const KR: f64 = 0.2126;
const KG: f64 = 0.7152;
const KB: f64 = 0.0722;

pub struct GlslEngine;

static GLSL_DIAG_SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();

fn glsl_diag_once(key: String, message: impl FnOnce() -> String) {
    let seen = GLSL_DIAG_SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let mut seen = seen.lock().unwrap();
    if seen.insert(key) {
        log::info!("{}", message());
    }
}

impl GlslEngine {
    /// Run `shader` on `main` (colour tex). `out_size` = target display size
    /// (drives WHEN/OUTPUT). Returns the new MAIN texture.
    pub fn apply(
        gc: &mut GlContext,
        shader: &UserShader,
        main: GpuTex,
        out_size: (i32, i32),
    ) -> Result<GpuTex> {
        if shader.is_post {
            return Self::run_post(gc, shader, main);
        }
        if shader.uses_yuv {
            return Self::run_yuv_emulated(gc, shader, main, out_size);
        }
        if shader.is_rgb {
            Self::run_rgb(gc, shader, main, out_size)
        } else {
            Self::run_luma(gc, shader, main, out_size)
        }
    }

    /// OUTPUT/SCALED-hook shaders (sharpeners etc.): run on the final
    /// display-size image; all size refs equal the image size.
    pub fn run_post(gc: &mut GlContext, shader: &UserShader, img: GpuTex) -> Result<GpuTex> {
        let mut textures = HashMap::new();
        let mut comps = HashMap::new();
        for k in ["OUTPUT", "SCALED", "PREKERNEL", "POSTKERNEL", "MAIN", "RGB"] {
            textures.insert(k.to_string(), img);
            comps.insert(k.to_string(), 4u8);
        }
        let mut sizes = Self::base_sizes(img.w(), img.h(), (img.w(), img.h()));
        for k in ["SCALED", "PREKERNEL", "POSTKERNEL"] {
            sizes.insert(k.into(), (img.w() as f64, img.h() as f64));
        }
        Self::run_passes(gc, shader, &mut textures, &mut comps, &mut sizes, None)?;
        // result = whichever hooked plane was written by the passes
        for k in ["OUTPUT", "SCALED", "PREKERNEL", "POSTKERNEL", "MAIN"] {
            if let Some(t) = textures.get(k) {
                if t.tex != img.tex {
                    return Ok(*t);
                }
            }
        }
        Ok(img)
    }

    fn program_for(
        gc: &mut GlContext,
        cache_key: &str,
        p: &Pass,
        raw_hook: &str,
        binds_comps: &[(String, u8, bool)],
        storage: &[StorageBind],
        params: &[(String, f32, ParamTy)],
    ) -> Result<glow::Program> {
        if let Some(program) = gc.cached_program(cache_key) {
            return Ok(program);
        }
        let code = driver_f16_code(gc, compat_shader_code(p, binds_comps));
        let src = fragment_src(&code, raw_hook, binds_comps, storage, params);
        match gc.program_named(cache_key, &src) {
            Ok(prog) => Ok(prog),
            Err(first) if uses_f16_types(&code) => {
                let fallback = rewrite_f16_to_fp32(&code);
                let fallback_src = fragment_src(&fallback, raw_hook, binds_comps, storage, params);
                match gc.program_named(cache_key, &fallback_src) {
                    Ok(prog) => {
                        glsl_diag_once(format!("{}:frag:f16-fallback", p.desc), || {
                            format!(
                                "glsl-f16-fallback: pass='{}' kind=fragment compiled as fp32 because fp16 GLSL was rejected by the driver",
                                p.desc
                            )
                        });
                        Ok(prog)
                    }
                    Err(second) => Err(anyhow!(
                        "GLSL compile failed ({}): {first}; fp32 fallback also failed: {second}",
                        p.desc
                    )),
                }
            }
            Err(e) => {
                dump_failed_source(&p.desc, &src);
                Err(anyhow!("GLSL compile failed ({}): {e}", p.desc))
            }
        }
    }

    fn compute_program_for(
        gc: &mut GlContext,
        cache_key: &str,
        p: &Pass,
        spec: ComputeSpec,
        raw_hook: &str,
        binds_comps: &[(String, u8, bool)],
        storage: &[StorageBind],
        params: &[(String, f32, ParamTy)],
        out_comps: u8,
    ) -> Result<glow::Program> {
        if let Some(program) = gc.cached_program(cache_key) {
            return Ok(program);
        }
        let code = driver_f16_code(gc, compat_shader_code(p, binds_comps));
        let src = compute_src(
            &code,
            spec,
            raw_hook,
            binds_comps,
            storage,
            params,
            out_comps,
        );
        match gc.compute_program_named(cache_key, &src) {
            Ok(prog) => Ok(prog),
            Err(first) if uses_f16_types(&code) => {
                let fallback = rewrite_f16_to_fp32(&code);
                let fallback_src = compute_src(
                    &fallback,
                    spec,
                    raw_hook,
                    binds_comps,
                    storage,
                    params,
                    out_comps,
                );
                match gc.compute_program_named(cache_key, &fallback_src) {
                    Ok(prog) => {
                        glsl_diag_once(format!("{}:compute:f16-fallback", p.desc), || {
                            format!(
                                "glsl-f16-fallback: pass='{}' kind=compute compiled as fp32 because fp16 GLSL was rejected by the driver",
                                p.desc
                            )
                        });
                        Ok(prog)
                    }
                    Err(second) => Err(anyhow!(
                        "GLSL compute compile failed ({}): {first}; fp32 fallback also failed: {second}",
                        p.desc
                    )),
                }
            }
            Err(e) => {
                dump_failed_source(&p.desc, &src);
                Err(anyhow!("GLSL compute compile failed ({}): {e}", p.desc))
            }
        }
    }

    fn render(
        gc: &mut GlContext,
        prog: glow::Program,
        binds: &[(String, GpuTex)],
        storage: &[StorageBind],
        params: &[(String, f32, ParamTy)],
        ow: i32,
        oh: i32,
        out_comps: u8,
    ) -> GpuTex {
        let ow = ow.max(1);
        let oh = oh.max(1);
        let tgt = gc.make_tex(ow, oh, out_comps, Dtype::F16);
        gc.bind_target(tgt);
        let gl = gc.gl.clone();
        let linearized = set_scaled_bind_filters(gc, binds, ow, oh);
        unsafe {
            gl.use_program(Some(prog));
            for (i, sb) in storage.iter().enumerate() {
                gl.bind_image_texture(
                    1 + i as u32,
                    Some(sb.tex.tex),
                    0,
                    false,
                    0,
                    glow::READ_WRITE,
                    sb.gl_format,
                );
            }
            for (unit, (name, tex)) in binds.iter().enumerate() {
                gl.active_texture(glow::TEXTURE0 + unit as u32);
                gl.bind_texture(tex.target(), Some(tex.tex));
                // unused uniforms are optimized out; location None is fine
                if let Some(loc) = gl.get_uniform_location(prog, &format!("{name}_tx")) {
                    gl.uniform_1_i32(Some(&loc), unit as i32);
                }
                if let Some(loc) = gl.get_uniform_location(prog, &format!("{name}_size")) {
                    gl.uniform_2_f32(Some(&loc), tex.w() as f32, tex.h() as f32);
                }
                if let Some(loc) = gl.get_uniform_location(prog, &format!("{name}_pt")) {
                    gl.uniform_2_f32(Some(&loc), 1.0 / tex.w() as f32, 1.0 / tex.h() as f32);
                }
            }
            if let Some(loc) = gl.get_uniform_location(prog, "input_size") {
                let tex = binds.first().map(|(_, tex)| *tex).unwrap_or(tgt);
                gl.uniform_2_f32(Some(&loc), tex.w() as f32, tex.h() as f32);
            }
            if let Some(loc) = gl.get_uniform_location(prog, "target_size") {
                gl.uniform_2_f32(Some(&loc), ow as f32, oh as f32);
            }
            if let Some(loc) = gl.get_uniform_location(prog, "out_size") {
                gl.uniform_2_f32(Some(&loc), ow as f32, oh as f32);
            }
            if let Some(loc) = gl.get_uniform_location(prog, "tex_offset") {
                gl.uniform_2_f32(Some(&loc), 0.0, 0.0);
            }
            if let Some(loc) = gl.get_uniform_location(prog, "random") {
                gl.uniform_1_f32(Some(&loc), current_random());
            }
            if let Some(loc) = gl.get_uniform_location(prog, "frame") {
                gl.uniform_1_i32(Some(&loc), current_frame());
            }
            for (name, v, ty) in params {
                if let Some(loc) = gl.get_uniform_location(prog, name) {
                    match ty {
                        ParamTy::Int => gl.uniform_1_i32(Some(&loc), v.round() as i32),
                        ParamTy::Uint => gl.uniform_1_u32(Some(&loc), v.round().max(0.0) as u32),
                        // DEFINE/CONSTANT params are compiled in, no uniform exists
                        ParamTy::Define
                        | ParamTy::ConstFloat
                        | ParamTy::ConstInt
                        | ParamTy::ConstUint => {}
                        ParamTy::Float => gl.uniform_1_f32(Some(&loc), *v),
                    }
                }
            }
            gl.bind_vertex_array(Some(gc.quad_vao));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_vertex_array(None);
            if !storage.is_empty() {
                // imageStore writes (temporal history) must be visible to the
                // next pass/frame
                gl.memory_barrier(
                    glow::SHADER_IMAGE_ACCESS_BARRIER_BIT | glow::TEXTURE_FETCH_BARRIER_BIT,
                );
            }
        }
        gc.unbind_target();
        restore_source_filters(gc, &linearized);
        tgt
    }

    fn compute(
        gc: &mut GlContext,
        prog: glow::Program,
        spec: ComputeSpec,
        binds: &[(String, GpuTex)],
        storage: &[StorageBind],
        params: &[(String, f32, ParamTy)],
        ow: i32,
        oh: i32,
        out_comps: u8,
    ) -> GpuTex {
        let ow = ow.max(1);
        let oh = oh.max(1);
        let tgt = gc.make_tex(ow, oh, out_comps, Dtype::F16);
        let gl = gc.gl.clone();
        let linearized = set_scaled_bind_filters(gc, binds, ow, oh);
        unsafe {
            gl.use_program(Some(prog));
            for (i, sb) in storage.iter().enumerate() {
                gl.bind_image_texture(
                    1 + i as u32,
                    Some(sb.tex.tex),
                    0,
                    false,
                    0,
                    glow::READ_WRITE,
                    sb.gl_format,
                );
            }
            for (unit, (name, tex)) in binds.iter().enumerate() {
                gl.active_texture(glow::TEXTURE0 + unit as u32);
                gl.bind_texture(tex.target(), Some(tex.tex));
                if let Some(loc) = gl.get_uniform_location(prog, &format!("{name}_tx")) {
                    gl.uniform_1_i32(Some(&loc), unit as i32);
                }
                if let Some(loc) = gl.get_uniform_location(prog, &format!("{name}_size")) {
                    gl.uniform_2_f32(Some(&loc), tex.w() as f32, tex.h() as f32);
                }
                if let Some(loc) = gl.get_uniform_location(prog, &format!("{name}_pt")) {
                    gl.uniform_2_f32(Some(&loc), 1.0 / tex.w() as f32, 1.0 / tex.h() as f32);
                }
            }
            if let Some(loc) = gl.get_uniform_location(prog, "input_size") {
                let tex = binds.first().map(|(_, tex)| *tex).unwrap_or(tgt);
                gl.uniform_2_f32(Some(&loc), tex.w() as f32, tex.h() as f32);
            }
            if let Some(loc) = gl.get_uniform_location(prog, "target_size") {
                gl.uniform_2_f32(Some(&loc), ow as f32, oh as f32);
            }
            if let Some(loc) = gl.get_uniform_location(prog, "out_size") {
                gl.uniform_2_f32(Some(&loc), ow as f32, oh as f32);
            }
            if let Some(loc) = gl.get_uniform_location(prog, "tex_offset") {
                gl.uniform_2_f32(Some(&loc), 0.0, 0.0);
            }
            if let Some(loc) = gl.get_uniform_location(prog, "random") {
                gl.uniform_1_f32(Some(&loc), current_random());
            }
            if let Some(loc) = gl.get_uniform_location(prog, "frame") {
                gl.uniform_1_i32(Some(&loc), current_frame());
            }
            for (name, v, ty) in params {
                if let Some(loc) = gl.get_uniform_location(prog, name) {
                    match ty {
                        ParamTy::Int => gl.uniform_1_i32(Some(&loc), v.round() as i32),
                        ParamTy::Uint => gl.uniform_1_u32(Some(&loc), v.round().max(0.0) as u32),
                        // DEFINE/CONSTANT params are compiled in, no uniform exists
                        ParamTy::Define
                        | ParamTy::ConstFloat
                        | ParamTy::ConstInt
                        | ParamTy::ConstUint => {}
                        ParamTy::Float => gl.uniform_1_f32(Some(&loc), *v),
                    }
                }
            }
            let (_, gl_format) = image_format(out_comps);
            gl.bind_image_texture(0, Some(tgt.tex), 0, false, 0, glow::WRITE_ONLY, gl_format);
            let groups_x = div_ceil(ow as u32, spec.block_w);
            let groups_y = div_ceil(oh as u32, spec.block_h);
            gl.dispatch_compute(groups_x, groups_y, 1);
            gl.memory_barrier(
                glow::SHADER_IMAGE_ACCESS_BARRIER_BIT | glow::TEXTURE_FETCH_BARRIER_BIT,
            );
            gl.bind_image_texture(0, None, 0, false, 0, glow::WRITE_ONLY, gl_format);
        }
        restore_source_filters(gc, &linearized);
        tgt
    }

    fn run_passes(
        gc: &mut GlContext,
        shader: &UserShader,
        textures: &mut HashMap<String, GpuTex>,
        comps: &mut HashMap<String, u8>,
        sizes: &mut Sizes,
        forced_root: Option<&str>,
    ) -> Result<()> {
        // //!TEXTURE embedded LUTs: uploaded once (persistent), then bindable
        // exactly like saved pass textures (crt-royale & other libretro ports)
        for t in &shader.textures {
            let persistent_key =
                persistent_texture_key(&shader.path, &t.name, t.storage, forced_root);
            let tex = gc.persistent_texture(
                &persistent_key,
                t.w,
                t.h,
                t.d,
                t.comps,
                if t.f16 {
                    crate::render::gl::Dtype::F16
                } else {
                    crate::render::gl::Dtype::U8
                },
                t.filter_linear,
                match t.border {
                    crate::render::mpv::TexBorder::Repeat => glow::REPEAT,
                    crate::render::mpv::TexBorder::Mirror => glow::MIRRORED_REPEAT,
                    crate::render::mpv::TexBorder::Clamp => glow::CLAMP_TO_EDGE,
                },
                &t.data,
            );
            textures.insert(t.name.clone(), tex);
            comps.insert(t.name.clone(), t.comps);
            sizes.insert(t.name.clone(), (t.w as f64, t.h as f64));
        }
        let pmap: Params = shader
            .params
            .iter()
            .map(|p| (p.name.clone(), p.value as f64))
            .collect();
        let pvals: Vec<(String, f32, ParamTy)> = shader
            .params
            .iter()
            .map(|p| (p.name.clone(), p.value, p.ty))
            .collect();
        for (idx, p) in shader.passes.iter().enumerate() {
            if let Some(root) = forced_root {
                // A plane invocation starts at LUMA/CHROMA, but subsequent
                // passes commonly hook custom //!SAVE intermediates. Those
                // passes belong to the same plane and must continue running.
                // Only reject a pass when it explicitly hooks another
                // canonical plane (for example CHROMA during the LUMA run).
                let canonical_hooks: Vec<&str> = p
                    .hooks
                    .iter()
                    .map(|h| canonical_tex_name(h))
                    .filter(|h| is_plane_hook(h))
                    .collect();
                if !canonical_hooks.is_empty() && !canonical_hooks.contains(&root) {
                    continue;
                }
                if canonical_hooks.is_empty()
                    && !p.hooks.iter().any(|hook| textures.contains_key(hook))
                {
                    continue;
                }
            }
            let hooked = forced_root
                .map(str::to_string)
                .unwrap_or_else(|| hooked_texture_key(p, textures));
            // skip passes hooking planes we don't carry (e.g. CHROMA-only)
            let Some(&hooked_tex) = textures.get(&hooked) else {
                glsl_diag_once(
                    format!("{}:{idx}:missing-hook:{hooked}", shader.path),
                    || {
                        format!(
                            "glsl-pass skip missing hook: shader={} pass={} desc='{}' hook={} available={:?}",
                            shader.name(),
                            idx,
                            p.desc,
                            hooked,
                            textures.keys().cloned().collect::<Vec<_>>()
                        )
                    },
                );
                continue;
            };
            textures.insert("HOOKED".into(), hooked_tex);
            comps.insert("HOOKED".into(), comps[&hooked]);
            sizes.insert("HOOKED".into(), sizes[&hooked]);

            if let Some(when) = &p.when {
                let when_value = eval_rpn_p(when, sizes, &pmap).unwrap_or(0.0);
                if when_value <= 0.0 {
                    glsl_diag_once(format!("{}:{idx}:when-skip", shader.path), || {
                        format!(
                            "glsl-pass skip WHEN: shader={} pass={} desc='{}' hook={} when='{}' value={:.3}",
                            shader.name(),
                            idx,
                            p.desc,
                            hooked,
                            when,
                            when_value
                        )
                    });
                    continue;
                }
            }

            // TRUNCATE like mpv (C cast), don't round: crt-royale's mask
            // resize buffer must hold an integer number of tiles — rounding
            // 8296*0.0625=518.5 UP to 519 split 2 tiles at 259.5px and the
            // phosphor mask phase-flipped at the tile seam (visible red/green
            // vertical band at screen center).
            let ow = match &p.width {
                Some(e) => eval_rpn_p(e, sizes, &pmap)
                    .ok_or_else(|| anyhow!("bad WIDTH expr: {e}"))?
                    as i32,
                None => sizes[&hooked].0 as i32,
            };
            let oh = match &p.height {
                Some(e) => eval_rpn_p(e, sizes, &pmap)
                    .ok_or_else(|| anyhow!("bad HEIGHT expr: {e}"))?
                    as i32,
                None => sizes[&hooked].1 as i32,
            };

            let mut binds: Vec<(String, GpuTex)> = Vec::with_capacity(p.binds.len());
            let mut binds_comps: Vec<(String, u8, bool)> = Vec::with_capacity(p.binds.len());
            let mut storage: Vec<StorageBind> = Vec::new();
            for n in &p.binds {
                let t = *textures
                    .get(n)
                    .ok_or_else(|| anyhow!("BIND {n} not available in {}", shader.name()))?;
                // //!STORAGE textures bind as read/write image2D, not samplers
                if let Some(st) = shader.textures.iter().find(|t| t.storage && &t.name == n) {
                    let (layout, gl_format) = storage_format(st.comps, st.f16);
                    storage.push(StorageBind {
                        name: n.clone(),
                        tex: t,
                        layout,
                        gl_format,
                    });
                    continue;
                }
                binds.push((n.clone(), t));
                binds_comps.push((n.clone(), comps[n], t.d() > 1));
            }

            let save = p.save.clone().unwrap_or_else(|| hooked.clone());
            let out_comps = match canonical_tex_name(&save) {
                "MAIN" | "RGB" | "NATIVE" | "MAINPRESUB" | "OUTPUT" | "SCALED" => 4,
                "LUMA" => 1,
                "CHROMA" => 2,
                _ => p.components,
            };
            let raw_hook = raw_hook_name(p, &hooked);
            let program_key = pass_program_cache_key(
                shader,
                idx,
                p.compute.is_some(),
                &raw_hook,
                &binds_comps,
                &storage,
                &pvals,
                out_comps,
            );
            glsl_diag_once(format!("{}:{idx}:run", shader.path), || {
                format!(
                    "glsl-pass run: shader={} pass={} desc='{}' hook={} raw_hook={} save={} size={}x{} comps={} binds={:?} compute={}",
                    shader.name(),
                    idx,
                    p.desc,
                    hooked,
                    raw_hook,
                    save,
                    ow,
                    oh,
                    out_comps,
                    p.binds,
                    p.compute.is_some()
                )
            });
            let tgt = if let Some(spec) = p.compute {
                let prog = Self::compute_program_for(
                    gc,
                    &program_key,
                    p,
                    spec,
                    &raw_hook,
                    &binds_comps,
                    &storage,
                    &pvals,
                    out_comps,
                )?;
                Self::compute(gc, prog, spec, &binds, &storage, &pvals, ow, oh, out_comps)
            } else {
                let prog = Self::program_for(
                    gc,
                    &program_key,
                    p,
                    &raw_hook,
                    &binds_comps,
                    &storage,
                    &pvals,
                )?;
                Self::render(gc, prog, &binds, &storage, &pvals, ow, oh, out_comps)
            };
            textures.insert(save.clone(), tgt);
            comps.insert(save.clone(), out_comps);
            sizes.insert(save, (ow as f64, oh as f64));
        }
        Ok(())
    }

    fn base_sizes(w: i32, h: i32, out_size: (i32, i32)) -> Sizes {
        let mut sizes = Sizes::new();
        let wh = (w as f64, h as f64);
        for k in ["MAIN", "RGB", "NATIVE", "MAINPRESUB", "LUMA", "CHROMA"] {
            sizes.insert(k.into(), wh);
        }
        sizes.insert("OUTPUT".into(), (out_size.0 as f64, out_size.1 as f64));
        sizes
    }

    fn run_rgb(
        gc: &mut GlContext,
        shader: &UserShader,
        main: GpuTex,
        out_size: (i32, i32),
    ) -> Result<GpuTex> {
        let mut textures = HashMap::from([
            ("MAIN".to_string(), main),
            ("RGB".to_string(), main),
            ("NATIVE".to_string(), main),
            ("MAINPRESUB".to_string(), main),
        ]);
        let mut comps = HashMap::from([
            ("MAIN".to_string(), 4u8),
            ("RGB".to_string(), 4),
            ("NATIVE".to_string(), 4),
            ("MAINPRESUB".to_string(), 4),
            ("LUMA".to_string(), 1),
            ("CHROMA".to_string(), 2),
        ]);
        let mut sizes = Self::base_sizes(main.w(), main.h(), out_size);
        Self::run_passes(gc, shader, &mut textures, &mut comps, &mut sizes, None)?;
        textures
            .get("MAIN")
            .or_else(|| textures.get("RGB"))
            .copied()
            .ok_or_else(|| anyhow!("shader produced no MAIN/RGB output"))
    }

    /// LUMA-plane shaders (FSRCNNX): extract Y -> run -> luma substitution
    /// (chroma from a LINEAR upscale of the original).
    fn run_luma(
        gc: &mut GlContext,
        shader: &UserShader,
        main: GpuTex,
        out_size: (i32, i32),
    ) -> Result<GpuTex> {
        let luma = Self::extract_luma(gc, main)?;
        let mut textures = HashMap::from([
            ("LUMA".to_string(), luma),
            ("MAIN".to_string(), main),
            ("RGB".to_string(), main),
            ("NATIVE".to_string(), main),
            ("MAINPRESUB".to_string(), main),
        ]);
        let mut comps = HashMap::from([
            ("LUMA".to_string(), 1u8),
            ("MAIN".to_string(), 4),
            ("RGB".to_string(), 4),
            ("NATIVE".to_string(), 4),
            ("MAINPRESUB".to_string(), 4),
            ("CHROMA".to_string(), 2),
        ]);
        let mut sizes = Self::base_sizes(main.w(), main.h(), out_size);
        Self::run_passes(gc, shader, &mut textures, &mut comps, &mut sizes, None)?;
        let yprime = textures["LUMA"];
        Self::luma_substitute(gc, main, yprime)
    }

    /// RGB capture -> mpv-style pseudo YUV planes. This path is used only for
    /// shaders that explicitly hook/bind LUMA or CHROMA; RGB/MAIN shaders never
    /// pay this cost. It lets YUV-plane mpv shaders such as nnedi3 run on
    /// uncompressed/RGB sources instead of silently behaving as no-ops.
    fn run_yuv_emulated(
        gc: &mut GlContext,
        shader: &UserShader,
        main: GpuTex,
        out_size: (i32, i32),
    ) -> Result<GpuTex> {
        glsl_diag_once(format!("{}:yuv-emulation", shader.path), || {
            format!(
                "glsl-yuv-emulation: shader={} input={}x{} output_ref={}x{} chroma=RGB-derived-420",
                shader.name(),
                main.w(),
                main.h(),
                out_size.0,
                out_size.1
            )
        });
        let luma = Self::extract_luma(gc, main)?;
        let chroma = Self::extract_chroma(gc, main)?;
        let base_textures = HashMap::from([
            ("MAIN".to_string(), main),
            ("RGB".to_string(), main),
            ("NATIVE".to_string(), main),
            ("MAINPRESUB".to_string(), main),
            ("LUMA".to_string(), luma),
            ("CHROMA".to_string(), chroma),
        ]);
        let base_comps = HashMap::from([
            ("MAIN".to_string(), 4u8),
            ("RGB".to_string(), 4),
            ("NATIVE".to_string(), 4),
            ("MAINPRESUB".to_string(), 4),
            ("LUMA".to_string(), 1),
            ("CHROMA".to_string(), 2),
        ]);
        // mpv invokes a multi-hook shader once at each hook stage. Keep the
        // saved intermediates (G/GC/etc.) private to each plane; sharing them
        // makes CHROMA consume LUMA guides and produces the characteristic
        // cyan output seen with nlmeans.
        let mut luma_textures = base_textures.clone();
        let mut luma_comps = base_comps.clone();
        let mut luma_sizes = Self::base_sizes(main.w(), main.h(), out_size);
        luma_sizes.insert("CHROMA".into(), (chroma.w() as f64, chroma.h() as f64));
        Self::run_passes(
            gc,
            shader,
            &mut luma_textures,
            &mut luma_comps,
            &mut luma_sizes,
            Some("LUMA"),
        )?;
        let yprime = luma_textures.get("LUMA").copied().unwrap_or(luma);

        let mut chroma_textures = base_textures;
        let mut chroma_comps = base_comps;
        let mut chroma_sizes = Self::base_sizes(main.w(), main.h(), out_size);
        chroma_sizes.insert("CHROMA".into(), (chroma.w() as f64, chroma.h() as f64));
        // A multi-hook shader may explicitly SAVE an analysis result while
        // running at LUMA and BIND it later from a CHROMA pass (StreamClean RT
        // uses SC_TEMP_GUIDE this way). Carry only such missing, non-declared
        // intermediates across. //!STORAGE histories remain plane-private as
        // required by temporal shaders, and canonical LUMA/CHROMA roots retain
        // their own base planes.
        for pass in shader.passes.iter().filter(|pass| {
            pass.hooks
                .iter()
                .any(|hook| canonical_tex_name(hook) == "CHROMA")
        }) {
            for name in &pass.binds {
                let declared_texture = shader.textures.iter().any(|texture| &texture.name == name);
                if declared_texture || chroma_textures.contains_key(name) {
                    continue;
                }
                let (Some(&texture), Some(&components), Some(&size)) = (
                    luma_textures.get(name),
                    luma_comps.get(name),
                    luma_sizes.get(name),
                ) else {
                    continue;
                };
                chroma_textures.insert(name.clone(), texture);
                chroma_comps.insert(name.clone(), components);
                chroma_sizes.insert(name.clone(), size);
            }
        }
        Self::run_passes(
            gc,
            shader,
            &mut chroma_textures,
            &mut chroma_comps,
            &mut chroma_sizes,
            Some("CHROMA"),
        )?;
        let cprime = chroma_textures.get("CHROMA").copied().unwrap_or(chroma);
        Self::yuv_to_rgb(gc, yprime, cprime)
    }

    fn extract_luma(gc: &mut GlContext, main: GpuTex) -> Result<GpuTex> {
        let frag = format!(
            "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D MAIN_tx;\nvoid main(){{\nvec3 rgb = texture(MAIN_tx, v_uv).rgb;\nfrag = vec4(dot(rgb, vec3({KR},{KG},{KB})),0,0,1);\n}}\n"
        );
        let prog = gc.program(&frag).map_err(|e| anyhow!(e))?;
        Ok(Self::render(
            gc,
            prog,
            &[("MAIN".into(), main)],
            &[],
            &[],
            main.w(),
            main.h(),
            1,
        ))
    }

    fn extract_chroma(gc: &mut GlContext, main: GpuTex) -> Result<GpuTex> {
        let frag = format!(
            "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D MAIN_tx;\nvoid main(){{\nvec3 rgb = texture(MAIN_tx, v_uv).rgb;\nfloat y = dot(rgb, vec3({KR},{KG},{KB}));\nfloat cb = (rgb.b - y) / (2.0*(1.0-{KB})) + 0.5;\nfloat cr = (rgb.r - y) / (2.0*(1.0-{KR})) + 0.5;\nfrag = vec4(cb,cr,0.0,1.0);\n}}\n"
        );
        let prog = gc.program(&frag).map_err(|e| anyhow!(e))?;
        Ok(Self::render(
            gc,
            prog,
            &[("MAIN".into(), main)],
            &[],
            &[],
            (main.w() + 1) / 2,
            (main.h() + 1) / 2,
            2,
        ))
    }

    fn luma_substitute(gc: &mut GlContext, main: GpuTex, yprime: GpuTex) -> Result<GpuTex> {
        let frag = format!(
            "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D MAIN_tx; uniform sampler2D Y_tx;\nvoid main(){{\nvec3 rgb = texture(MAIN_tx, v_uv).rgb;\nfloat y = dot(rgb, vec3({KR},{KG},{KB}));\nfloat cb = (rgb.b - y) / (2.0*(1.0-{KB}));\nfloat cr = (rgb.r - y) / (2.0*(1.0-{KR}));\nfloat yp = texture(Y_tx, v_uv).r;\nfloat r = yp + 2.0*(1.0-{KR})*cr;\nfloat b = yp + 2.0*(1.0-{KB})*cb;\nfloat g = (yp - {KR}*r - {KB}*b) / {KG};\nfrag = vec4(r,g,b,1.0);\n}}\n"
        );
        let prog = gc.program(&frag).map_err(|e| anyhow!(e))?;
        gc.set_filter_linear(main, true); // chroma upscale
        let out = Self::render(
            gc,
            prog,
            &[("MAIN".into(), main), ("Y".into(), yprime)],
            &[],
            &[],
            yprime.w(),
            yprime.h(),
            4,
        );
        gc.set_filter_linear(main, false);
        Ok(out)
    }

    fn yuv_to_rgb(gc: &mut GlContext, y: GpuTex, c: GpuTex) -> Result<GpuTex> {
        let frag = format!(
            "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D Y_tx; uniform sampler2D C_tx;\nvoid main(){{\nfloat yp = texture(Y_tx, v_uv).r;\nvec2 cc = texture(C_tx, v_uv).rg - vec2(0.5);\nfloat cb = cc.x;\nfloat cr = cc.y;\nfloat r = yp + 2.0*(1.0-{KR})*cr;\nfloat b = yp + 2.0*(1.0-{KB})*cb;\nfloat g = (yp - {KR}*r - {KB}*b) / {KG};\nfrag = vec4(clamp(vec3(r,g,b), 0.0, 1.0), 1.0);\n}}\n"
        );
        let prog = gc.program(&frag).map_err(|e| anyhow!(e))?;
        gc.set_filter_linear(c, true);
        let out = Self::render(
            gc,
            prog,
            &[("Y".into(), y), ("C".into(), c)],
            &[],
            &[],
            y.w(),
            y.h(),
            4,
        );
        gc.set_filter_linear(c, false);
        Ok(out)
    }
}

fn persistent_texture_key(
    shader_path: &str,
    texture_name: &str,
    storage: bool,
    forced_root: Option<&str>,
) -> String {
    if storage {
        // Temporal shaders run once for LUMA and once for CHROMA. Their writable
        // history must not be shared or the chroma pass overwrites luma history.
        format!(
            "{shader_path}::{texture_name}::{}",
            forced_root.unwrap_or("GLOBAL")
        )
    } else {
        // Embedded LUTs are immutable and can be shared by every plane.
        format!("{shader_path}::{texture_name}")
    }
}

fn canonical_tex_name(name: &str) -> &str {
    match name {
        "NATIVE" | "MAINPRESUB" => "MAIN",
        "SCALED" | "PREKERNEL" | "POSTKERNEL" => "OUTPUT",
        _ => name,
    }
}

fn is_plane_hook(name: &str) -> bool {
    matches!(name, "LUMA" | "CHROMA" | "MAIN" | "RGB" | "OUTPUT")
}

fn set_scaled_bind_filters(
    gc: &GlContext,
    binds: &[(String, GpuTex)],
    ow: i32,
    oh: i32,
) -> Vec<GpuTex> {
    let _ = (ow, oh);
    let mut changed = Vec::new();
    for (_name, tex) in binds {
        // mpv samples EVERY bound texture bilinearly. Exact texel-center
        // reads are identical under LINEAR, but shaders that sample BETWEEN
        // texels on purpose (crt-royale's tex2Dblur*fast) degenerate under
        // NEAREST — at exact texel-edge offsets the rounding even flipped
        // mid-screen and drew a phase-seam line. Persistent //!TEXTURE LUTs
        // keep their declared filter.
        if !gc.is_persistent(*tex) {
            gc.set_filter_linear(*tex, true);
            changed.push(*tex);
        }
    }
    changed
}

fn restore_source_filters(gc: &GlContext, textures: &[GpuTex]) {
    for tex in textures {
        gc.set_filter_linear(*tex, false);
    }
}

fn hooked_texture_key(p: &Pass, textures: &HashMap<String, GpuTex>) -> String {
    for h in &p.hooks {
        let key = canonical_tex_name(h);
        if textures.contains_key(key) {
            return key.to_string();
        }
        if textures.contains_key(h) {
            return h.clone();
        }
    }
    "MAIN".into()
}

fn raw_hook_name(p: &Pass, hooked: &str) -> String {
    p.hooks
        .iter()
        .find(|h| h.as_str() == hooked || canonical_tex_name(h) == hooked)
        .cloned()
        .unwrap_or_else(|| hooked.to_string())
}

fn hook_raw_define(hook: &str, binds_comps: &[(String, u8, bool)]) -> Option<String> {
    let sampler = raw_sampler_name(hook, binds_comps);
    let aliases: &[&str] = match hook {
        "LUMA" => &["LUMA"],
        "CHROMA" => &["CHROMA"],
        "MAIN" => &["MAIN"],
        "RGB" => &["RGB", "MAIN"],
        "NATIVE" => &["NATIVE", "MAIN"],
        "MAINPRESUB" => &["MAINPRESUB", "MAIN"],
        "OUTPUT" | "SCALED" | "PREKERNEL" | "POSTKERNEL" => &["OUTPUT"],
        _ => return None,
    };
    let mut out = String::new();
    for alias in aliases {
        out.push_str(&format!(
            "#ifndef {alias}_raw\n#define {alias}_raw {sampler}_tx\n#endif\n\
#ifndef {alias}_mul\n#define {alias}_mul 1.0\n#endif\n"
        ));
        if *alias != sampler {
            out.push_str(&format!(
                "#ifndef {alias}_size\n#define {alias}_size {sampler}_size\n#endif\n\
#ifndef {alias}_pt\n#define {alias}_pt {sampler}_pt\n#endif\n\
#ifndef {alias}_rot\n#define {alias}_rot {sampler}_rot\n#endif\n\
#ifndef {alias}_pos\n#define {alias}_pos {sampler}_pos\n#endif\n\
#ifndef {alias}_map\n#define {alias}_map(id) {sampler}_map(id)\n#endif\n\
#ifndef {alias}_tex\n#define {alias}_tex(p) {sampler}_tex(p)\n#endif\n\
#ifndef {alias}_texOff\n#define {alias}_texOff(o) {sampler}_texOff(o)\n#endif\n\
#ifndef {alias}_gather\n#define {alias}_gather(p,c) {sampler}_gather(p,c)\n#endif\n\
#ifndef {alias}_off\n#define {alias}_off {sampler}_off\n#endif\n"
            ));
        }
    }
    Some(out)
}

fn raw_sampler_name(hook: &str, binds_comps: &[(String, u8, bool)]) -> String {
    let has = |name: &str| binds_comps.iter().any(|(n, _, _)| n == name);
    let candidates: &[&str] = match hook {
        "LUMA" => &["LUMA", "HOOKED"],
        "CHROMA" => &["CHROMA", "HOOKED"],
        "MAIN" => &["MAIN", "HOOKED"],
        "RGB" => &["RGB", "MAIN", "HOOKED"],
        "NATIVE" => &["NATIVE", "MAIN", "RGB", "HOOKED"],
        "MAINPRESUB" => &["MAINPRESUB", "MAIN", "RGB", "HOOKED"],
        "OUTPUT" | "SCALED" | "PREKERNEL" | "POSTKERNEL" => &["OUTPUT", "SCALED", "HOOKED"],
        _ => &["HOOKED"],
    };
    candidates
        .iter()
        .find(|name| has(name))
        .unwrap_or(&"HOOKED")
        .to_string()
}

fn image_format(comps: u8) -> (&'static str, u32) {
    match comps {
        1 => ("r16f", glow::R16F),
        2 => ("rg16f", glow::RG16F),
        _ => ("rgba16f", glow::RGBA16F),
    }
}

/// A //!STORAGE texture bound to a pass: read/write image2D.
#[derive(Clone)]
struct StorageBind {
    name: String,
    tex: GpuTex,
    /// GLSL layout qualifier matching the texture's internal format
    layout: &'static str,
    /// glBindImageTexture format (must match the texture's internal format)
    gl_format: u32,
}

fn storage_format(comps: u8, f16: bool) -> (&'static str, u32) {
    match (comps, f16) {
        (1, true) => ("r16f", glow::R16F),
        (2, true) => ("rg16f", glow::RG16F),
        (_, true) => ("rgba16f", glow::RGBA16F),
        (1, false) => ("r8", glow::R8),
        (2, false) => ("rg8", glow::RG8),
        _ => ("rgba8", glow::RGBA8),
    }
}

fn fragment_src(
    code: &str,
    raw_hook: &str,
    binds_comps: &[(String, u8, bool)],
    storage: &[StorageBind],
    params: &[(String, f32, ParamTy)],
) -> String {
    let (extensions, body) = split_glsl_extensions(code);
    let mut prelude = format!(
        // #version 440 like mpv composes: slang/libretro ports rely on the
        // 4.2+ relaxation of `const` locals with non-constant initializers
        // (NVIDIA rejects them under 330 with C1059  Ecrt-royale case)
        "#version 440\n{extensions}in vec2 v_uv;\nout vec4 frag;\nuniform vec2 input_size;\nuniform vec2 target_size;\nuniform vec2 out_size;\nuniform vec2 tex_offset;\nvec4 linearize(vec4 c){{ return vec4(pow(max(c.rgb, vec3(0.0)), vec3(2.2)), c.a); }}\nvec4 delinearize(vec4 c){{ return vec4(pow(max(c.rgb, vec3(0.0)), vec3(1.0/2.2)), c.a); }}\n",
    );
    append_common_uniforms(
        &mut prelude,
        raw_hook,
        binds_comps,
        storage,
        params,
        false,
        &body,
    );
    format!("{prelude}\n{body}\nvoid main(){{ frag = hook(); }}\n")
}

#[allow(clippy::too_many_arguments)]
fn compute_src(
    code: &str,
    spec: ComputeSpec,
    raw_hook: &str,
    binds_comps: &[(String, u8, bool)],
    storage: &[StorageBind],
    params: &[(String, f32, ParamTy)],
    out_comps: u8,
) -> String {
    let (extensions, body) = split_glsl_extensions(code);
    let (layout, _) = image_format(out_comps);
    let mut prelude = format!(
        "#version 440\n{extensions}layout(local_size_x = {}, local_size_y = {}, local_size_z = 1) in;\nlayout({layout}, binding = 0) writeonly uniform image2D out_image;\nuniform vec2 input_size;\nuniform vec2 target_size;\nuniform vec2 out_size;\nuniform vec2 tex_offset;\nvec4 linearize(vec4 c){{ return vec4(pow(max(c.rgb, vec3(0.0)), vec3(2.2)), c.a); }}\nvec4 delinearize(vec4 c){{ return vec4(pow(max(c.rgb, vec3(0.0)), vec3(1.0/2.2)), c.a); }}\n",
        spec.threads_w, spec.threads_h
    );
    append_common_uniforms(
        &mut prelude,
        raw_hook,
        binds_comps,
        storage,
        params,
        true,
        &body,
    );
    format!("{prelude}\n{body}\nvoid main(){{ hook(); }}\n")
}

/// True when the shader body itself `#define`s `name` (mpv-libretro ports
/// alias bind names to helper macros; pre-defining the bare name then would
/// be a "macro redefined" error).
fn body_defines(body: &str, name: &str) -> bool {
    body.lines().any(|l| {
        let l = l.trim_start();
        l.strip_prefix("#define")
            .and_then(|r| {
                r.trim_start()
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .next()
                    .map(|t| t == name)
            })
            .unwrap_or(false)
    })
}

#[allow(clippy::too_many_arguments)]
fn append_common_uniforms(
    prelude: &mut String,
    raw_hook: &str,
    binds_comps: &[(String, u8, bool)],
    storage: &[StorageBind],
    params: &[(String, f32, ParamTy)],
    compute: bool,
    body: &str,
) {
    if !params.iter().any(|(name, _, _)| name == "random") {
        prelude.push_str("uniform float random;\n");
    }
    if !params.iter().any(|(name, _, _)| name == "frame") {
        prelude.push_str("uniform int frame;\n");
    }
    for (name, v, ty) in params {
        match ty {
            ParamTy::Int => prelude.push_str(&format!("uniform int {name};\n")),
            ParamTy::Uint => prelude.push_str(&format!("uniform uint {name};\n")),
            ParamTy::Define => {
                // preprocessor constant (usable as loop bound / array size);
                // keep integer-looking values as int literals like mpv does
                if v.fract() == 0.0 && v.abs() < 1e9 {
                    prelude.push_str(&format!("#define {name} {}\n", *v as i64));
                } else {
                    prelude.push_str(&format!("#define {name} {v:?}\n"));
                }
            }
            // CONSTANT params: compiled in as real constants (value changes
            // recompile via the source-keyed program cache)
            ParamTy::ConstFloat => prelude.push_str(&format!("const float {name} = {v:?};\n")),
            ParamTy::ConstInt => {
                prelude.push_str(&format!("const int {name} = {};\n", v.round() as i64))
            }
            ParamTy::ConstUint => prelude.push_str(&format!(
                "const uint {name} = {}u;\n",
                v.round().max(0.0) as u64
            )),
            ParamTy::Float => prelude.push_str(&format!("uniform float {name};\n")),
        }
    }
    if let Some(define) = hook_raw_define(raw_hook, binds_comps) {
        prelude.push_str(&define);
    }
    for (n, _, is_3d) in binds_comps {
        if *is_3d {
            prelude.push_str(&format!(
                "uniform sampler3D {n}_tx;\n#ifndef {n}_raw\n#define {n}_raw {n}_tx\n#endif\n#ifndef {n}_mul\n#define {n}_mul 1.0\n#endif\n"
            ));
            if !body_defines(body, n) {
                prelude.push_str(&format!("#ifndef {n}\n#define {n} {n}_tx\n#endif\n"));
            }
            continue;
        }
        if compute {
            prelude.push_str(&format!(
                "uniform sampler2D {n}_tx;\nuniform vec2 {n}_size;\nuniform vec2 {n}_pt;\n#ifndef {n}_raw\n#define {n}_raw {n}_tx\n#endif\n#ifndef {n}_mul\n#define {n}_mul 1.0\n#endif\n#ifndef {n}_rot\n#define {n}_rot mat2(1.0, 0.0, 0.0, 1.0)\n#endif\n#define {n}_map(id) ((vec2(id) + vec2(0.5)) / out_size)\n#define {n}_pos {n}_map(gl_GlobalInvocationID.xy)\n#define {n}_tex(p) (texture({n}_tx,(p)))\n#define {n}_texOff(o) (texture({n}_tx, {n}_pos + vec2(o)*{n}_pt))\n#define {n}_gather(p,c) (textureGather({n}_tx,(p),(c)))\n#define {n}_off vec2(0.0)\n"
            ));
        } else {
            prelude.push_str(&format!(
                "uniform sampler2D {n}_tx;\nuniform vec2 {n}_size;\nuniform vec2 {n}_pt;\n#ifndef {n}_raw\n#define {n}_raw {n}_tx\n#endif\n#ifndef {n}_mul\n#define {n}_mul 1.0\n#endif\n#ifndef {n}_rot\n#define {n}_rot mat2(1.0, 0.0, 0.0, 1.0)\n#endif\n#define {n}_map(id) ((vec2(id) + vec2(0.5)) / out_size)\n#define {n}_pos v_uv\n#define {n}_tex(p) (texture({n}_tx,(p)))\n#define {n}_texOff(o) (texture({n}_tx, v_uv + vec2(o)*{n}_pt))\n#define {n}_gather(p,c) (textureGather({n}_tx,(p),(c)))\n#define {n}_off vec2(0.0)\n"
            ));
        }
        // mpv/libplacebo also expose the BARE bind name as the sampler itself
        // (libretro ports do `texture(SamplerLUT1, uv)` on //!TEXTURE binds)  E        // unless the body #defines that name itself (would be a redefinition)
        if !body_defines(body, n) {
            prelude.push_str(&format!("#ifndef {n}\n#define {n} {n}_tx\n#endif\n"));
        }
    }
    // //!STORAGE textures: read/write image2D at image units 1.. (unit 0 is
    // the compute out_image), layout format matching the texture storage
    for (i, sb) in storage.iter().enumerate() {
        prelude.push_str(&format!(
            "layout({}, binding = {}) uniform image2D {};\n",
            sb.layout,
            i + 1,
            sb.name
        ));
    }
}

fn div_ceil(v: u32, d: u32) -> u32 {
    (v + d - 1) / d
}

fn uses_f16_types(code: &str) -> bool {
    code.contains("float16_t")
        || code.contains("f16vec")
        || code.contains("f16mat")
        || code.contains("GL_EXT_shader_explicit_arithmetic_types_float16")
}

fn driver_f16_code(gc: &GlContext, code: String) -> String {
    if !uses_f16_types(&code) {
        return code;
    }
    let extensions = gc.gl.supported_extensions();
    let has_ext = extensions.contains("GL_EXT_shader_explicit_arithmetic_types_float16");
    let has_nv = extensions.contains("GL_NV_gpu_shader5");
    let adapted = adapt_f16_extensions(&code, has_ext, has_nv);
    if !has_ext && has_nv && adapted != code {
        glsl_diag_once("f16-extension:nvidia".into(), || {
            "glsl-f16-extension: using GL_NV_gpu_shader5 in place of unsupported GL_EXT float16 arithmetic"
                .to_string()
        });
    }
    adapted
}

fn adapt_f16_extensions(code: &str, has_ext: bool, has_nv: bool) -> String {
    if has_ext || !has_nv {
        return code.to_string();
    }
    let mut body = String::with_capacity(code.len() + 48);
    let mut has_nv_directive = false;
    for line in code.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#extension ")
            && trimmed.contains("GL_EXT_shader_explicit_arithmetic_types_float16")
        {
            continue;
        }
        has_nv_directive |=
            trimmed.starts_with("#extension ") && trimmed.contains("GL_NV_gpu_shader5");
        body.push_str(line);
        body.push('\n');
    }
    if has_nv_directive {
        body
    } else {
        format!("#extension GL_NV_gpu_shader5 : enable\n{body}")
    }
}

fn pass_program_cache_key(
    shader: &UserShader,
    pass_index: usize,
    compute: bool,
    raw_hook: &str,
    binds: &[(String, u8, bool)],
    storage: &[StorageBind],
    params: &[(String, f32, ParamTy)],
    out_comps: u8,
) -> String {
    let kind = if compute { "compute" } else { "fragment" };
    let mut key = format!(
        "mpv-{kind}:{:016x}:{pass_index}:{raw_hook}:{out_comps}",
        shader.source_hash
    );
    for (name, comps, is_3d) in binds {
        key.push_str(&format!(":b={name},{comps},{}", u8::from(*is_3d)));
    }
    for item in storage {
        key.push_str(&format!(
            ":s={},{},{}",
            item.name, item.layout, item.gl_format
        ));
    }
    for (name, value, ty) in params {
        if matches!(
            ty,
            ParamTy::Define | ParamTy::ConstFloat | ParamTy::ConstInt | ParamTy::ConstUint
        ) {
            key.push_str(&format!(":p={name},{value:?},{ty:?}"));
        }
    }
    key
}

fn rewrite_f16_to_fp32(code: &str) -> String {
    let mut body = String::with_capacity(code.len());
    for line in code.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#extension ")
            && (trimmed.contains("GL_EXT_shader_explicit_arithmetic_types_float16")
                || trimmed.contains("GL_NV_gpu_shader5"))
        {
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    body.replace("float16_t", "float")
        .replace("f16vec4", "vec4")
        .replace("f16vec3", "vec3")
        .replace("f16vec2", "vec2")
        .replace("f16vec1", "float")
        .replace("f16mat4x4", "mat4")
        .replace("f16mat3x3", "mat3")
        .replace("f16mat2x2", "mat2")
        .replace("f16mat4", "mat4")
        .replace("f16mat3", "mat3")
        .replace("f16mat2", "mat2")
}

fn split_glsl_extensions(code: &str) -> (String, String) {
    let mut extensions = String::new();
    let mut body = String::with_capacity(code.len());
    for line in code.lines() {
        if line.trim_start().starts_with("#extension ") {
            extensions.push_str(line);
            extensions.push('\n');
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }
    (extensions, body)
}

/// Compat debugging: NEO_GLSL_DUMP=<dir> writes the fully composed source of
/// any pass that fails to compile (error line numbers refer to THIS text, not
/// the .glsl file, so this is the only way to map them).
fn dump_failed_source(desc: &str, src: &str) {
    let Ok(dir) = std::env::var("NEO_GLSL_DUMP") else {
        return;
    };
    let safe: String = desc
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let path = std::path::Path::new(&dir).join(format!("glsl_fail_{safe}.glsl"));
    if std::fs::write(&path, src).is_ok() {
        log::info!("glsl dump: wrote failing source to {}", path.display());
    }
}

fn current_random() -> f32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.subsec_nanos() % 1_000_000) as f32 / 1_000_000.0)
        .unwrap_or(0.5)
}

/// mpv's `frame` builtin uniform: a per-video-frame counter (temporal dither
/// / interlacing effects in CRT shaders key odd/even off it). Advanced ONCE
/// per processed frame by the chain, NOT per pass/shader.
static FRAME_INDEX: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

pub fn advance_frame() {
    FRAME_INDEX.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn current_frame() -> i32 {
    FRAME_INDEX.load(std::sync::atomic::Ordering::Relaxed)
}

fn compat_shader_code(p: &Pass, binds_comps: &[(String, u8, bool)]) -> String {
    let names: Vec<&str> = binds_comps
        .iter()
        .filter_map(|(name, comps, _)| (*comps == 1).then_some(name.as_str()))
        .collect();
    let code = rewrite_float_texture_casts(&p.code(), &names);
    let code = rewrite_mpv_scalar_swizzle_compat(&code);
    rewrite_spirv_dp4_compat(&code)
}

fn rewrite_spirv_dp4_compat(code: &str) -> String {
    if !code.contains("GL_EXT_spirv_intrinsics") || !code.contains("dp4(") {
        return code.to_string();
    }
    let mut body = String::with_capacity(code.len());
    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.contains("GL_EXT_spirv_intrinsics")
            || trimmed.starts_with("spirv_instruction")
            || (trimmed.starts_with("int dp4(") && trimmed.ends_with(';'))
        {
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    let body = body.replace("dp4(", "neo_dp4(");
    format!(
        "int neo_i8(int v, int shift) {{ return (v << (24 - shift)) >> 24; }}\n\
int neo_dp4(int a, int b, int fmt) {{\n\
    return neo_i8(a, 0) * neo_i8(b, 0)\n\
         + neo_i8(a, 8) * neo_i8(b, 8)\n\
         + neo_i8(a, 16) * neo_i8(b, 16)\n\
         + neo_i8(a, 24) * neo_i8(b, 24);\n\
}}\n{body}"
    )
}

fn rewrite_mpv_scalar_swizzle_compat(code: &str) -> String {
    // hdeband and a few mpv shaders use a macro that expands to float.x in
    // LUMA mode. mpv's shader translation accepts it, desktop OpenGL rejects
    // it, so normalize that macro before compilation.
    code.replace(
        "#define unval(v) vec4(v.x, 0, 0, poi_.a)",
        "#define unval(v) vec4((v), 0, 0, poi_.a)",
    )
}

fn rewrite_float_texture_casts(code: &str, one_channel_names: &[&str]) -> String {
    let mut out = String::with_capacity(code.len());
    let mut pos = 0usize;
    while let Some(rel) = code[pos..].find("float(") {
        let start = pos + rel;
        let inner_start = start + "float(".len();
        let rest = &code[inner_start..];
        let matches_one_channel = one_channel_names.iter().any(|name| {
            rest.starts_with(&format!("{name}_texOff("))
                || rest.starts_with(&format!("{name}_tex("))
        });
        if matches_one_channel {
            if let Some(end) = matching_paren(code, start + "float".len()) {
                out.push_str(&code[pos..start]);
                out.push_str(code[inner_start..end].trim());
                out.push_str(".r");
                pos = end + 1;
                continue;
            }
        }
        out.push_str(&code[pos..inner_start]);
        pos = inner_start;
    }
    out.push_str(&code[pos..]);
    out
}

fn matching_paren(s: &str, open_idx: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (off, ch) in s[open_idx..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open_idx + off);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        adapt_f16_extensions, fragment_src, hook_raw_define, persistent_texture_key,
        rewrite_spirv_dp4_compat,
    };

    #[test]
    fn nvidia_f16_extension_replaces_unsupported_ext_directive() {
        let source = "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable\nf16vec4 f(){return f16vec4(1.0);}";
        let adapted = adapt_f16_extensions(source, false, true);
        assert!(!adapted.contains("GL_EXT_shader_explicit_arithmetic_types_float16"));
        assert!(adapted.contains("GL_NV_gpu_shader5"));
        assert!(adapted.contains("f16vec4"));
    }

    #[test]
    fn temporal_storage_is_isolated_per_processing_plane() {
        let luma = persistent_texture_key("temporal.glsl", "PREV1", true, Some("LUMA"));
        let chroma = persistent_texture_key("temporal.glsl", "PREV1", true, Some("CHROMA"));
        assert_ne!(luma, chroma);

        let luma_lut = persistent_texture_key("temporal.glsl", "LUT", false, Some("LUMA"));
        let chroma_lut = persistent_texture_key("temporal.glsl", "LUT", false, Some("CHROMA"));
        assert_eq!(luma_lut, chroma_lut);
    }

    #[test]
    fn mpv_main_aliases_follow_the_actual_bound_texture() {
        let hooked = hook_raw_define("MAIN", &[("HOOKED".into(), 4, false)]).unwrap();
        assert!(hooked.contains("#define MAIN_texOff(o) HOOKED_texOff(o)"));

        let main = hook_raw_define("MAIN", &[("MAIN".into(), 4, false)]).unwrap();
        assert!(!main.contains("#define MAIN_texOff"));
        assert!(main.contains("#define MAIN_raw MAIN_tx"));
    }

    #[test]
    fn mpv_prelude_exposes_scalar_mul_and_input_size() {
        let src = fragment_src(
            "vec4 hook(){ return HOOKED_tex(HOOKED_pos) * HOOKED_mul; }",
            "MAIN",
            &[("HOOKED".into(), 4, false)],
            &[],
            &[],
        );
        assert!(src.contains("uniform vec2 input_size;"));
        assert!(src.contains("#define HOOKED_mul 1.0"));
        assert!(!src.contains("#define HOOKED_mul vec4"));
    }

    #[test]
    fn embedded_3d_lut_uses_sampler3d() {
        let src = fragment_src(
            "vec4 hook(){ return texture(LUT, vec3(0.5)); }",
            "MAIN",
            &[("LUT".into(), 4, true)],
            &[],
            &[],
        );
        assert!(src.contains("uniform sampler3D LUT_tx;"));
        assert!(src.contains("#define LUT LUT_tx"));
    }

    #[test]
    fn spirv_dp4_has_exact_integer_glsl_fallback() {
        let src = "#extension GL_EXT_spirv_intrinsics : require\nspirv_instruction (id = 1)\nint dp4(int a, int b, spirv_literal int fmt);\nint f(){return dp4(1,2,0);}";
        let out = rewrite_spirv_dp4_compat(src);
        assert!(!out.contains("GL_EXT_spirv_intrinsics"));
        assert!(!out.contains("spirv_instruction"));
        assert!(out.contains("neo_i8"));
        assert!(out.contains("neo_dp4(1,2,0)"));
    }
}
