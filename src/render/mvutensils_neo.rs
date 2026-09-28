//! Native MVUtensils-Neo temporal-filter contract.
//!
//! This module deliberately models the complete Super -> AnalyseMany ->
//! Degrain parameter surface before the filter is exposed in the picker.  It
//! is not a one-frame approximation: a radius of N owns N past and N future
//! frames and does not silently reduce the requested radius.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};

pub const FILTER_NAME: &str = "MVUtensils-Neo";
pub const MIN_RADIUS: u8 = 1;
pub const MAX_RADIUS: u8 = 25;
pub const BACKEND_ABI_VERSION: u32 = 1;
pub const BACKEND_MANIFEST: &str = "backend.json";

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BackendManifest {
    pub schema_version: u32,
    pub backend_abi: u32,
    pub name: String,
    pub library: String,
    pub runtime: String,
    pub plugin: String,
    pub license: String,
}

/// Stable C ABI shared with the separately distributed backend pack. All
/// pointers are borrowed only for the duration of the call that receives
/// them; ownership never crosses the DLL boundary.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BackendCreateDescV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub radius: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub overlap_x: u32,
    pub overlap_y: u32,
    pub pel: u32,
    pub search_mode: u32,
    pub search_param: u32,
    pub thsad_y: u32,
    pub thsad_c: u32,
    pub thsad2_y: u32,
    pub thsad2_c: u32,
    pub thscd1: u32,
    pub thscd2: f32,
    pub precision_bits: u32,
    pub pad_x: u32,
    pub pad_y: u32,
    pub sharp: u32,
    pub rfilter: u32,
    pub one_level: u32,
    pub levels: u32,
    pub pel_search: u32,
    pub mv_lambda: u32,
    pub chroma: u32,
    pub delta: u32,
    pub l_sad: u32,
    pub p_level: u32,
    pub global_mv: u32,
    pub p_new: u32,
    pub p_zero: u32,
    pub p_global: u32,
    pub bad_sad: u32,
    pub bad_range: u32,
    pub meander: u32,
    pub try_many: u32,
    pub fields: u32,
    pub tff: u32,
    pub satd: u32,
    pub plane_y: u32,
    pub plane_u: u32,
    pub plane_v: u32,
    pub limit_y: f32,
    pub limit_c: f32,
    pub weight_count: u32,
    pub weights: [u32; 51],
    /// Optional ABI-v1 tail extension. New backends use this to run motion
    /// analysis at a cheaper depth while keeping Degrain at precision_bits.
    /// Old ABI-v1 backends safely ignore this field because they copy only
    /// their older descriptor prefix.
    pub analysis_precision_bits: u32,
    /// Run the CPU backend on its dedicated worker so GPU stages can overlap.
    pub async_pipeline: u32,
    /// Apply a lighter motion-analysis profile at high internal resolutions.
    /// Degrain output precision/radius remain unchanged.
    pub highres_fast: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BackendFrameV1 {
    pub struct_size: u32,
    pub sequence: u64,
    pub width: u32,
    pub height: u32,
    pub stride_bytes: u32,
    /// RGBA8 pixels. The pack converts to its internal planar precision.
    pub data: *const u8,
    pub data_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BackendProfileV1 {
    pub struct_size: u32,
    pub threads: u32,
    pub convert_in_ms: f64,
    pub mvtools_ms: f64,
    pub convert_out_ms: f64,
    pub total_ms: f64,
}

pub const BACKEND_QUERY_SYMBOL: &[u8] = b"mvutensils_neo_query_v1\0";
pub const BACKEND_FEATURES_SYMBOL: &[u8] = b"mvutensils_neo_features_v1\0";
pub const BACKEND_CREATE_SYMBOL: &[u8] = b"mvutensils_neo_create_v1\0";
pub const BACKEND_PUSH_SYMBOL: &[u8] = b"mvutensils_neo_push_v1\0";
pub const BACKEND_DESTROY_SYMBOL: &[u8] = b"mvutensils_neo_destroy_v1\0";
pub const BACKEND_PROFILE_SYMBOL: &[u8] = b"mvutensils_neo_profile_v1\0";
pub const BACKEND_VARIANT_SYMBOL: &[u8] = b"mvutensils_neo_variant_v1\0";

type BackendFeaturesFn = unsafe extern "C" fn() -> u32;
const FEATURE_MIXED_ANALYSIS: u32 = 1 << 0;
const FEATURE_CPU_PIPELINE: u32 = 1 << 1;
const FEATURE_HIGHRES_FAST: u32 = 1 << 2;

type BackendCreateFn =
    unsafe extern "C" fn(*const BackendCreateDescV1, *const u16, *mut c_char, usize) -> *mut c_void;
type BackendPushFn = unsafe extern "C" fn(
    *mut c_void,
    *const BackendFrameV1,
    *mut u8,
    usize,
    *mut usize,
    *mut c_char,
    usize,
) -> i32;
type BackendDestroyFn = unsafe extern "C" fn(*mut c_void);
type BackendProfileFn = unsafe extern "C" fn(*mut c_void, *mut BackendProfileV1) -> i32;
type BackendVariantFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;

pub struct PortableBackendLibrary {
    _library: libloading::Library,
    features: u32,
    create: BackendCreateFn,
    push: BackendPushFn,
    destroy: BackendDestroyFn,
    profile: Option<BackendProfileFn>,
    variant: Option<BackendVariantFn>,
}

impl PortableBackendLibrary {
    /// Load only the absolute DLL declared by the validated portable pack and
    /// verify the complete v1 entry-point set before the filter is advertised.
    pub fn load(layout: &PortableBackendLayout) -> Result<Self, String> {
        layout.validate()?;
        // Plain LoadLibraryW searches the application/current/PATH directories,
        // not necessarily the directory containing an absolute DLL.  That made
        // the pack appear to work when VapourSynth happened to be installed or
        // already loaded on the development PC, while a clean user's machine
        // could not resolve vapoursynth.dll/fftw3f.dll beside the bridge.
        // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR makes the portable pack directory the
        // dependency root without mutating PATH or process-global DLL state.
        let native = unsafe {
            libloading::os::windows::Library::load_with_flags(
                &layout.backend,
                libloading::os::windows::LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
                    | libloading::os::windows::LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        }
        .map_err(|error| format!("load {}: {error}", layout.backend.display()))?;
        let library = libloading::Library::from(native);
        unsafe {
            let query: libloading::Symbol<'_, unsafe extern "C" fn() -> u32> = library
                .get(BACKEND_QUERY_SYMBOL)
                .map_err(|error| format!("missing mvutensils_neo_query_v1: {error}"))?;
            let abi = query();
            if abi != BACKEND_ABI_VERSION {
                return Err(format!(
                    "MVUtensils-Neo DLL ABI mismatch: dll={abi} host={BACKEND_ABI_VERSION}"
                ));
            }
            let features = library
                .get::<BackendFeaturesFn>(BACKEND_FEATURES_SYMBOL)
                .ok()
                .map(|symbol| symbol())
                .unwrap_or(0);
            let create: BackendCreateFn = *library
                .get(BACKEND_CREATE_SYMBOL)
                .map_err(|error| format!("missing mvutensils_neo_create_v1: {error}"))?;
            let push: BackendPushFn = *library
                .get(BACKEND_PUSH_SYMBOL)
                .map_err(|error| format!("missing mvutensils_neo_push_v1: {error}"))?;
            let destroy: BackendDestroyFn = *library
                .get(BACKEND_DESTROY_SYMBOL)
                .map_err(|error| format!("missing mvutensils_neo_destroy_v1: {error}"))?;
            let profile = library
                .get::<BackendProfileFn>(BACKEND_PROFILE_SYMBOL)
                .ok()
                .map(|symbol| *symbol);
            let variant = library
                .get::<BackendVariantFn>(BACKEND_VARIANT_SYMBOL)
                .ok()
                .map(|symbol| *symbol);
            return Ok(Self {
                _library: library,
                features,
                create,
                push,
                destroy,
                profile,
                variant,
            });
        }
    }

    pub fn create_session(
        self,
        layout: &PortableBackendLayout,
        options: &Options,
    ) -> Result<PortableBackendSession, String> {
        options.validate()?;
        if options.degrain.analysis_precision_bits != options.degrain.precision_bits
            && self.features & FEATURE_MIXED_ANALYSIS == 0
        {
            return Err("MVUtensils-Neo Backend Pack is too old for mixed analysis precision; install v847 or newer".into());
        }
        if options.degrain.async_pipeline && self.features & FEATURE_CPU_PIPELINE == 0 {
            return Err("MVUtensils-Neo Backend Pack is too old for CPU Pipeline; install v847 or newer".into());
        }
        if options.degrain.highres_fast && self.features & FEATURE_HIGHRES_FAST == 0 {
            return Err("MVUtensils-Neo Backend Pack is too old for High-res Fast; install v847 or newer".into());
        }
        let desc = options.backend_desc();
        let mut plugin = layout
            .plugin
            .as_os_str()
            .to_string_lossy()
            .encode_utf16()
            .collect::<Vec<_>>();
        plugin.push(0);
        let mut error = vec![0i8; 2048];
        let context =
            unsafe { (self.create)(&desc, plugin.as_ptr(), error.as_mut_ptr(), error.len()) };
        if context.is_null() {
            return Err(ffi_error(&error, "MVUtensils-Neo backend creation failed"));
        }
        Ok(PortableBackendSession {
            library: self,
            context,
            sequence: 0,
            error: vec![0i8; 2048],
        })
    }
}

pub struct PortableBackendSession {
    library: PortableBackendLibrary,
    context: *mut c_void,
    sequence: u64,
    error: Vec<c_char>,
}

// A session is created and called serially by Neo's render thread. The native
// context is never shared and the DLL is kept loaded for its entire lifetime.
unsafe impl Send for PortableBackendSession {}

impl PortableBackendSession {
    /// Push into a caller-owned reusable output buffer. Returns `Ok(false)`
    /// while the exact future-frame window is filling. Keeping both this RGBA
    /// allocation and the FFI error buffer alive removes two per-frame heap
    /// allocations from the live render path.
    pub fn push_rgba8_into(
        &mut self,
        width: u32,
        height: u32,
        stride_bytes: u32,
        pixels: &[u8],
        output: &mut Vec<u8>,
    ) -> Result<bool, String> {
        let frame = BackendFrameV1 {
            struct_size: std::mem::size_of::<BackendFrameV1>() as u32,
            sequence: self.sequence,
            width,
            height,
            stride_bytes,
            data: pixels.as_ptr(),
            data_len: pixels.len(),
        };
        self.sequence = self.sequence.wrapping_add(1);
        let required = width as usize * height as usize * 4;
        // Keep the scratch length stable across warm-up frames. Clearing it on
        // every `not ready yet` result made the next resize zero the full RGBA
        // frame again even though the backend overwrites every output byte.
        output.resize(required, 0);
        if let Some(first) = self.error.first_mut() {
            *first = 0;
        }
        let mut output_len = 0usize;
        let result = unsafe {
            (self.library.push)(
                self.context,
                &frame,
                output.as_mut_ptr(),
                output.len(),
                &mut output_len,
                self.error.as_mut_ptr(),
                self.error.len(),
            )
        };
        match result {
            0 => Ok(false),
            1 if output_len == required => Ok(true),
            1 => Err(format!(
                "MVUtensils-Neo returned an invalid frame size: {output_len}, expected {required}"
            )),
            _ => Err(ffi_error(&self.error, "MVUtensils-Neo processing failed")),
        }
    }

    pub fn profile(&self) -> Option<BackendProfileV1> {
        let query = self.library.profile?;
        let mut profile = BackendProfileV1 {
            struct_size: std::mem::size_of::<BackendProfileV1>() as u32,
            ..BackendProfileV1::default()
        };
        (unsafe { query(self.context, &mut profile) } != 0).then_some(profile)
    }

    pub fn variant(&self) -> Option<String> {
        let query = self.library.variant?;
        let ptr = unsafe { query(self.context) };
        if ptr.is_null() {
            return None;
        }
        let value = unsafe { std::ffi::CStr::from_ptr(ptr) };
        Some(value.to_string_lossy().into_owned())
    }

    /// Convenience wrapper retained for the standalone backend probe.
    pub fn push_rgba8(
        &mut self,
        width: u32,
        height: u32,
        stride_bytes: u32,
        pixels: &[u8],
    ) -> Result<Option<Vec<u8>>, String> {
        let mut output = Vec::new();
        self.push_rgba8_into(width, height, stride_bytes, pixels, &mut output)
            .map(|ready| ready.then_some(output))
    }
}

impl Drop for PortableBackendSession {
    fn drop(&mut self) {
        if !self.context.is_null() {
            unsafe { (self.library.destroy)(self.context) };
            self.context = std::ptr::null_mut();
        }
    }
}

fn ffi_error(buffer: &[c_char], fallback: &str) -> String {
    if buffer.first().copied().unwrap_or_default() == 0 {
        return fallback.into();
    }
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Portable-only runtime layout. We intentionally do not search PATH,
/// registry, Python installations, or the system VapourSynth plugin folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortableBackendLayout {
    pub root: PathBuf,
    pub backend: PathBuf,
    pub runtime: PathBuf,
    pub plugin: PathBuf,
    pub license: PathBuf,
}

impl PortableBackendLayout {
    pub fn under(app_dir: &Path) -> Self {
        let root = app_dir.join("backends").join(FILTER_NAME);
        Self {
            backend: root.join("mvutensils_neo_backend.dll"),
            runtime: root.join("vapoursynth.dll"),
            plugin: root.join("mvutensils.dll"),
            license: root.join("COPYING.GPL-2.0.txt"),
            root,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        for (label, path) in [
            ("library", &self.backend),
            ("runtime", &self.runtime),
            ("plugin", &self.plugin),
            ("license", &self.license),
        ] {
            if !path.is_file() {
                return Err(format!("portable {label} is missing: {}", path.display()));
            }
        }
        Ok(())
    }

    pub fn discover(app_dir: &Path) -> Result<Option<(Self, BackendManifest)>, String> {
        let expected = Self::under(app_dir);
        let manifest_path = expected.root.join(BACKEND_MANIFEST);
        if !manifest_path.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&manifest_path)
            .map_err(|error| format!("read {}: {error}", manifest_path.display()))?;
        let manifest: BackendManifest = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse {}: {error}", manifest_path.display()))?;
        if manifest.schema_version != 1 {
            return Err(format!(
                "unsupported manifest schema {}",
                manifest.schema_version
            ));
        }
        if manifest.backend_abi != BACKEND_ABI_VERSION {
            return Err(format!(
                "MVUtensils-Neo ABI mismatch: pack={} host={}",
                manifest.backend_abi, BACKEND_ABI_VERSION
            ));
        }
        if manifest.name != FILTER_NAME {
            return Err(format!("unexpected backend name: {}", manifest.name));
        }
        let safe_file = |field: &str, value: &str| -> Result<PathBuf, String> {
            let relative = Path::new(value);
            if relative.is_absolute()
                || relative
                    .components()
                    .any(|part| !matches!(part, std::path::Component::Normal(_)))
            {
                return Err(format!("unsafe {field} path: {value}"));
            }
            let path = expected.root.join(relative);
            if !path.is_file() {
                return Err(format!("portable {field} is missing: {}", path.display()));
            }
            Ok(path)
        };
        let library = safe_file("library", &manifest.library)?;
        let layout = Self {
            root: expected.root.clone(),
            backend: library.clone(),
            runtime: safe_file("runtime", &manifest.runtime)?,
            plugin: safe_file("plugin", &manifest.plugin)?,
            license: safe_file("license", &manifest.license)?,
        };
        if library.parent() != Some(layout.root.as_path()) {
            return Err("backend library must be directly inside its pack root".into());
        }
        Ok(Some((layout, manifest)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    LogarithmicDiamond = 0,
    Exhaustive = 1,
    Hexagon = 2,
    Umh = 3,
    Horizontal = 4,
    Vertical = 5,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SuperOptions {
    pub block_size: [u16; 2],
    pub overlap: [u16; 2],
    pub pad: [u16; 2],
    pub pel: u8,
    pub sharp: u8,
    pub rfilter: u8,
    pub one_level: bool,
}

impl Default for SuperOptions {
    fn default() -> Self {
        Self {
            block_size: [16, 16],
            overlap: [8, 8],
            pad: [16, 16],
            pel: 2,
            sharp: 2,
            rfilter: 1,
            one_level: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnalyseOptions {
    pub levels: u8,
    pub search: SearchMode,
    pub search_param: u16,
    pub pel_search: u16,
    pub mv_lambda: u32,
    pub chroma: bool,
    pub delta: u8,
    pub l_sad: u32,
    pub p_level: u8,
    pub global_mv: bool,
    pub p_new: u32,
    pub p_zero: u32,
    pub p_global: u32,
    pub bad_sad: u32,
    pub bad_range: u16,
    pub meander: bool,
    pub try_many: u8,
    pub fields: bool,
    pub tff: bool,
    pub satd: bool,
}

impl Default for AnalyseOptions {
    fn default() -> Self {
        Self {
            levels: 0,
            search: SearchMode::Hexagon,
            search_param: 2,
            pel_search: 2,
            mv_lambda: 1000,
            chroma: true,
            delta: 1,
            l_sad: 400,
            p_level: 1,
            global_mv: true,
            p_new: 25,
            p_zero: 25,
            p_global: 0,
            bad_sad: 10_000,
            bad_range: 24,
            meander: true,
            try_many: 0,
            fields: false,
            tff: false,
            satd: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DegrainOptions {
    pub radius: u8,
    pub th_sad: [u32; 2],
    pub th_sad2: [u32; 2],
    pub planes: [bool; 3],
    pub limit: [f32; 2],
    pub th_scd1: u32,
    /// Percentage of changed blocks, in the current MVUtensils 0..=100 scale.
    pub th_scd2: f32,
    pub weights: Option<Vec<u32>>,
    /// 8 or 16-bit precision used by the final Super/Degrain path.
    pub precision_bits: u8,
    /// 8 or 16-bit precision used only for motion-vector analysis.
    ///
    /// The captured source is RGBA8, so 8-bit analysis does not discard source
    /// precision. MVUtensils explicitly supports applying vectors analysed at
    /// one integer depth to Degrain operating at another depth.
    pub analysis_precision_bits: u8,
    /// Dedicated CPU worker pipeline. Adds a small queueing delay but allows
    /// adjacent GPU stages to overlap with MVTools CPU work.
    pub async_pipeline: bool,
    /// High-resolution motion-analysis acceleration. The backend only activates
    /// it at 1280x720 or larger and leaves final Degrain precision/radius intact.
    pub highres_fast: bool,
}

impl Default for DegrainOptions {
    fn default() -> Self {
        Self {
            radius: 4,
            th_sad: [400, 400],
            th_sad2: [400, 400],
            planes: [true, true, true],
            limit: [f32::INFINITY, f32::INFINITY],
            th_scd1: 400,
            th_scd2: 51.0,
            weights: None,
            precision_bits: 8,
            analysis_precision_bits: 8,
            async_pipeline: true,
            highres_fast: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    pub super_clip: SuperOptions,
    pub analyse: AnalyseOptions,
    pub degrain: DegrainOptions,
}

impl Options {
    pub fn backend_desc(&self) -> BackendCreateDescV1 {
        let mut weights = [0u32; 51];
        let weight_count = self.degrain.weights.as_ref().map_or(0, |values| {
            weights[..values.len()].copy_from_slice(values);
            values.len() as u32
        });
        BackendCreateDescV1 {
            struct_size: std::mem::size_of::<BackendCreateDescV1>() as u32,
            abi_version: BACKEND_ABI_VERSION,
            radius: self.degrain.radius.into(),
            block_x: self.super_clip.block_size[0].into(),
            block_y: self.super_clip.block_size[1].into(),
            overlap_x: self.super_clip.overlap[0].into(),
            overlap_y: self.super_clip.overlap[1].into(),
            pel: self.super_clip.pel.into(),
            search_mode: self.analyse.search as u32,
            search_param: self.analyse.search_param.into(),
            thsad_y: self.degrain.th_sad[0],
            thsad_c: self.degrain.th_sad[1],
            thsad2_y: self.degrain.th_sad2[0],
            thsad2_c: self.degrain.th_sad2[1],
            thscd1: self.degrain.th_scd1,
            thscd2: self.degrain.th_scd2,
            precision_bits: self.degrain.precision_bits.into(),
            pad_x: self.super_clip.pad[0].into(),
            pad_y: self.super_clip.pad[1].into(),
            sharp: self.super_clip.sharp.into(),
            rfilter: self.super_clip.rfilter.into(),
            one_level: self.super_clip.one_level.into(),
            levels: self.analyse.levels.into(),
            pel_search: self.analyse.pel_search.into(),
            mv_lambda: self.analyse.mv_lambda,
            chroma: self.analyse.chroma.into(),
            delta: self.analyse.delta.into(),
            l_sad: self.analyse.l_sad,
            p_level: self.analyse.p_level.into(),
            global_mv: self.analyse.global_mv.into(),
            p_new: self.analyse.p_new,
            p_zero: self.analyse.p_zero,
            p_global: self.analyse.p_global,
            bad_sad: self.analyse.bad_sad,
            bad_range: self.analyse.bad_range.into(),
            meander: self.analyse.meander.into(),
            try_many: self.analyse.try_many.into(),
            fields: self.analyse.fields.into(),
            tff: self.analyse.tff.into(),
            satd: self.analyse.satd.into(),
            plane_y: self.degrain.planes[0].into(),
            plane_u: self.degrain.planes[1].into(),
            plane_v: self.degrain.planes[2].into(),
            limit_y: self.degrain.limit[0],
            limit_c: self.degrain.limit[1],
            weight_count,
            weights,
            analysis_precision_bits: self.degrain.analysis_precision_bits.into(),
            async_pipeline: self.degrain.async_pipeline.into(),
            highres_fast: self.degrain.highres_fast.into(),
        }
    }

    /// Decode the values stored in `StageSpec::params`.  Keeping this mapping
    /// here makes the UI/preset representation independent from the eventual
    /// GPU implementation and, importantly, never drops a supported option.
    pub fn from_stage_params(params: &BTreeMap<String, f32>) -> Result<Self, String> {
        let mut out = Self::default();
        let value = |name: &str, fallback: f32| params.get(name).copied().unwrap_or(fallback);
        let integer = |name: &str, fallback: u32| -> Result<u32, String> {
            let v = value(name, fallback as f32);
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > u32::MAX as f32 {
                return Err(format!("{name} must be a non-negative integer"));
            }
            Ok(v as u32)
        };

        out.degrain.radius = integer("radius", out.degrain.radius.into())?
            .try_into()
            .map_err(|_| "radius is out of range")?;
        out.super_clip.block_size = [
            integer("blksize_x", out.super_clip.block_size[0].into())?
                .try_into()
                .map_err(|_| "blksize_x is out of range")?,
            integer("blksize_y", out.super_clip.block_size[1].into())?
                .try_into()
                .map_err(|_| "blksize_y is out of range")?,
        ];
        out.super_clip.overlap = [
            integer("overlap_x", out.super_clip.overlap[0].into())?
                .try_into()
                .map_err(|_| "overlap_x is out of range")?,
            integer("overlap_y", out.super_clip.overlap[1].into())?
                .try_into()
                .map_err(|_| "overlap_y is out of range")?,
        ];
        out.super_clip.pel = integer("pel", out.super_clip.pel.into())?
            .try_into()
            .map_err(|_| "pel is out of range")?;
        out.super_clip.pad = [
            integer("pad_x", out.super_clip.pad[0].into())?
                .try_into()
                .map_err(|_| "pad_x is out of range")?,
            integer("pad_y", out.super_clip.pad[1].into())?
                .try_into()
                .map_err(|_| "pad_y is out of range")?,
        ];
        out.super_clip.sharp = integer("sharp", out.super_clip.sharp.into())?
            .try_into()
            .map_err(|_| "sharp is out of range")?;
        out.super_clip.rfilter = integer("rfilter", out.super_clip.rfilter.into())?
            .try_into()
            .map_err(|_| "rfilter is out of range")?;
        out.super_clip.one_level = integer("onelevel", out.super_clip.one_level.into())? != 0;
        out.analyse.levels = integer("levels", out.analyse.levels.into())?
            .try_into()
            .map_err(|_| "levels is out of range")?;
        out.analyse.search = match integer("search_mode", out.analyse.search as u32)? {
            0 => SearchMode::LogarithmicDiamond,
            1 => SearchMode::Exhaustive,
            2 => SearchMode::Hexagon,
            3 => SearchMode::Umh,
            4 => SearchMode::Horizontal,
            5 => SearchMode::Vertical,
            _ => return Err("search_mode must be 0..=5".into()),
        };
        out.analyse.search_param = integer("search", out.analyse.search_param.into())?
            .try_into()
            .map_err(|_| "search is out of range")?;
        out.analyse.pel_search = integer("pelsearch", out.analyse.pel_search.into())?
            .try_into()
            .map_err(|_| "pelsearch is out of range")?;
        out.analyse.mv_lambda = integer("lambda", out.analyse.mv_lambda)?;
        out.analyse.chroma = integer("chroma", out.analyse.chroma.into())? != 0;
        out.analyse.delta = integer("delta", out.analyse.delta.into())?
            .try_into()
            .map_err(|_| "delta is out of range")?;
        out.analyse.l_sad = integer("lsad", out.analyse.l_sad)?;
        out.analyse.p_level = integer("plevel", out.analyse.p_level.into())?
            .try_into()
            .map_err(|_| "plevel is out of range")?;
        out.analyse.global_mv = integer("global", out.analyse.global_mv.into())? != 0;
        out.analyse.p_new = integer("pnew", out.analyse.p_new)?;
        out.analyse.p_zero = integer("pzero", out.analyse.p_zero)?;
        out.analyse.p_global = integer("pglobal", out.analyse.p_global)?;
        out.analyse.bad_sad = integer("badsad", out.analyse.bad_sad)?;
        out.analyse.bad_range = integer("badrange", out.analyse.bad_range.into())?
            .try_into()
            .map_err(|_| "badrange is out of range")?;
        out.analyse.meander = integer("meander", out.analyse.meander.into())? != 0;
        out.analyse.try_many = integer("trymany", out.analyse.try_many.into())?
            .try_into()
            .map_err(|_| "trymany is out of range")?;
        out.analyse.fields = integer("fields", out.analyse.fields.into())? != 0;
        out.analyse.tff = integer("tff", out.analyse.tff.into())? != 0;
        out.analyse.satd = integer("satd", out.analyse.satd.into())? != 0;
        out.degrain.th_sad = [
            integer("thsad_y", out.degrain.th_sad[0])?,
            integer("thsad_c", out.degrain.th_sad[1])?,
        ];
        out.degrain.th_sad2 = [
            integer("thsad2_y", out.degrain.th_sad2[0])?,
            integer("thsad2_c", out.degrain.th_sad2[1])?,
        ];
        out.degrain.th_scd1 = integer("thscd1", out.degrain.th_scd1)?;
        out.degrain.th_scd2 = value("thscd2", out.degrain.th_scd2);
        out.degrain.planes = [
            integer("plane_y", out.degrain.planes[0].into())? != 0,
            integer("plane_u", out.degrain.planes[1].into())? != 0,
            integer("plane_v", out.degrain.planes[2].into())? != 0,
        ];
        out.degrain.limit = [
            value("limit_y", out.degrain.limit[0]),
            value("limit_c", out.degrain.limit[1]),
        ];
        out.degrain.precision_bits = integer("precision", out.degrain.precision_bits.into())?
            .try_into()
            .map_err(|_| "precision is out of range")?;
        // Backward compatibility: old presets only carried `precision`.  If an
        // old preset explicitly selected 16-bit and has no new
        // `analysis_precision` key, preserve the legacy all-16-bit analysis
        // path.  New stages default to 8/8, and users can explicitly select
        // 16-bit Degrain + 8-bit analysis for the fast mixed-precision path.
        let analysis_fallback = if params.contains_key("precision")
            && !params.contains_key("analysis_precision")
        {
            out.degrain.precision_bits
        } else {
            out.degrain.analysis_precision_bits
        };
        out.degrain.analysis_precision_bits =
            integer("analysis_precision", analysis_fallback.into())?
                .try_into()
                .map_err(|_| "analysis_precision is out of range")?;
        out.degrain.async_pipeline =
            integer("cpu_pipeline", out.degrain.async_pipeline.into())? != 0;
        out.degrain.highres_fast =
            integer("highres_fast", out.degrain.highres_fast.into())? != 0;
        let weight_count = usize::from(out.degrain.radius) * 2 + 1;
        if (0..weight_count).any(|index| params.contains_key(&format!("weight_{index}"))) {
            let mut weights = Vec::with_capacity(weight_count);
            for index in 0..weight_count {
                weights.push(integer(&format!("weight_{index}"), 1)?);
            }
            out.degrain.weights = Some(weights);
        }
        out.validate()?;
        Ok(out)
    }

    pub fn validate(&self) -> Result<(), String> {
        let radius = self.degrain.radius;
        if !(MIN_RADIUS..=MAX_RADIUS).contains(&radius) {
            return Err(format!("radius must be {MIN_RADIUS}..={MAX_RADIUS}"));
        }
        for axis in 0..2 {
            let block = self.super_clip.block_size[axis];
            let overlap = self.super_clip.overlap[axis];
            if block == 0 {
                return Err("block size must be positive".into());
            }
            if overlap > block / 2 {
                return Err("overlap must not exceed half the block size".into());
            }
        }
        if !matches!(self.super_clip.pel, 1 | 2 | 4) {
            return Err("pel must be 1, 2, or 4".into());
        }
        if !(0.0..=100.0).contains(&self.degrain.th_scd2) || !self.degrain.th_scd2.is_finite() {
            return Err("thscd2 must be a finite percentage in 0..=100".into());
        }
        if !matches!(self.degrain.precision_bits, 8 | 16) {
            return Err("precision must be 8 or 16 bits".into());
        }
        if !matches!(self.degrain.analysis_precision_bits, 8 | 16) {
            return Err("analysis_precision must be 8 or 16 bits".into());
        }
        if !self.degrain.planes.iter().any(|enabled| *enabled) {
            return Err("at least one of the Y, U, or V planes must be enabled".into());
        }
        if let Some(weights) = &self.degrain.weights
            && weights.len() != usize::from(radius) * 2 + 1
        {
            return Err("weights must contain past, centre, and future entries".into());
        }
        Ok(())
    }

    pub fn lookahead_frames(&self) -> usize {
        usize::from(self.degrain.radius)
    }

    pub fn reference_frames(&self) -> usize {
        self.lookahead_frames() * 2
    }

    pub fn minimum_latency_ms(&self, input_fps: f64) -> Option<f64> {
        (input_fps.is_finite() && input_fps > 0.0)
            .then(|| self.lookahead_frames() as f64 * 1000.0 / input_fps)
    }
}

/// Exact MVUtensils reference-weight curve (`DegrainWeight` in Degrains.h).
/// The GPU kernel uses the same equation; keeping a host copy makes shader
/// conformance testable without weakening the algorithm.
pub fn degrain_weight(th_sad: u32, block_sad: u64) -> u16 {
    if th_sad == 0 || block_sad >= u64::from(th_sad) {
        return 0;
    }
    let r = block_sad as f64 / f64::from(th_sad);
    (256.0 * (1.0 - r * r) / (1.0 + r * r)) as u16
}

/// Normalize source/reference weights to exactly 256, matching upstream.
/// `user_weights` is `[source, bw1, fw1, ...]`.
pub fn normalise_weights(
    reference_weights: &mut [u16],
    user_weights: &[u32],
) -> Result<u16, String> {
    if user_weights.len() != reference_weights.len() + 1 {
        return Err("user weights must include source plus every reference".into());
    }
    let mut weighted = Vec::with_capacity(reference_weights.len());
    let mut sum = 256u64
        .saturating_mul(u64::from(user_weights[0]))
        .saturating_add(1);
    for (&weight, &user) in reference_weights.iter().zip(&user_weights[1..]) {
        let w = u64::from(weight).saturating_mul(u64::from(user));
        weighted.push(w);
        sum = sum.saturating_add(w);
    }
    let scale = 256.0 / sum as f64;
    let mut source = 256u16;
    for (dst, weighted) in reference_weights.iter_mut().zip(weighted) {
        let w = (weighted as f64 * scale) as u16;
        *dst = w;
        source = source.saturating_sub(w);
    }
    Ok(source)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionVector {
    /// Sub-pixel displacement in units of 1 / pel.
    pub x: i16,
    pub y: i16,
    pub sad: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockGrid {
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub step_x: u16,
    pub step_y: u16,
}

impl BlockGrid {
    pub fn new(
        width: u32,
        height: u32,
        block: [u16; 2],
        overlap: [u16; 2],
    ) -> Result<Self, String> {
        let step_x = block[0]
            .checked_sub(overlap[0])
            .filter(|&v| v > 0)
            .ok_or("invalid horizontal overlap")?;
        let step_y = block[1]
            .checked_sub(overlap[1])
            .filter(|&v| v > 0)
            .ok_or("invalid vertical overlap")?;
        let count = |extent: u32, size: u16, step: u16| {
            if extent <= u32::from(size) {
                1
            } else {
                (extent - u32::from(size)).div_ceil(u32::from(step)) + 1
            }
        };
        Ok(Self {
            blocks_x: count(width, block[0], step_x),
            blocks_y: count(height, block[1], step_y),
            step_x,
            step_y,
        })
    }

    pub fn len(self) -> usize {
        self.blocks_x as usize * self.blocks_y as usize
    }
}

/// MVTools-compatible scene-change decision. `thscd2=100` intentionally
/// disables the percentage gate without disabling per-block SAD weighting.
pub fn is_scene_change(block_sads: &[u32], th_scd1: u32, th_scd2: f32) -> bool {
    if block_sads.is_empty() {
        return false;
    }
    let changed = block_sads.iter().filter(|&&sad| sad > th_scd1).count();
    changed as f32 * 100.0 / block_sads.len() as f32 > th_scd2
}

/// One normalized Degrain sample. References must already be motion
/// compensated. This is shared by the 8/16-bit conformance path and mirrors
/// the integer upstream kernel's +128 then >>8 rounding.
pub fn degrain_u16_sample(
    source: u16,
    references: &[u16],
    source_weight: u16,
    reference_weights: &[u16],
) -> Result<u16, String> {
    if references.len() != reference_weights.len() {
        return Err("reference pixels and weights must have equal lengths".into());
    }
    let sum = references.iter().zip(reference_weights).fold(
        128u64 + u64::from(source) * u64::from(source_weight),
        |sum, (&pixel, &weight)| sum + u64::from(pixel) * u64::from(weight),
    );
    Ok((sum >> 8).min(u64::from(u16::MAX)) as u16)
}

/// Exact bidirectional ownership window. Once ready, `center()` is the frame
/// to be processed, with precisely radius frames on each side.
pub struct TemporalWindow<T> {
    radius: usize,
    frames: VecDeque<T>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameIdentity {
    pub sequence: u64,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PushResult {
    Accepted,
    /// Resolution, seek, or capture discontinuity invalidated every reference.
    Reset,
}

/// Owns the exact pre-filtered frame window for one chain instance. A queue is
/// never shared by two MVUtensils stages, so reordering/repeating filters cannot
/// leak history between them.
pub struct TemporalStageQueue<T> {
    window: TemporalWindow<(FrameIdentity, T)>,
    last: Option<FrameIdentity>,
}

impl<T> TemporalStageQueue<T> {
    pub fn new(radius: u8) -> Result<Self, String> {
        Ok(Self {
            window: TemporalWindow::new(radius)?,
            last: None,
        })
    }

    pub fn push(&mut self, identity: FrameIdentity, frame: T) -> PushResult {
        let reset = self.last.is_some_and(|last| {
            last.width != identity.width
                || last.height != identity.height
                || identity.sequence <= last.sequence
                || identity.sequence > last.sequence.saturating_add(1)
        });
        if reset {
            self.window.clear();
        }
        self.window.push((identity, frame));
        self.last = Some(identity);
        if reset {
            PushResult::Reset
        } else {
            PushResult::Accepted
        }
    }

    pub fn ready(&self) -> bool {
        self.window.is_ready()
    }

    pub fn center(&self) -> Option<&(FrameIdentity, T)> {
        self.window.center()
    }

    pub fn past(&self) -> Option<impl Iterator<Item = &(FrameIdentity, T)>> {
        self.window.past()
    }

    pub fn future(&self) -> Option<impl Iterator<Item = &(FrameIdentity, T)>> {
        self.window.future()
    }

    pub fn advance(&mut self) -> Option<(FrameIdentity, T)> {
        self.window.advance()
    }

    pub fn clear(&mut self) {
        self.window.clear();
        self.last = None;
    }
}

impl<T> TemporalWindow<T> {
    pub fn new(radius: u8) -> Result<Self, String> {
        if !(MIN_RADIUS..=MAX_RADIUS).contains(&radius) {
            return Err(format!("radius must be {MIN_RADIUS}..={MAX_RADIUS}"));
        }
        Ok(Self {
            radius: usize::from(radius),
            frames: VecDeque::with_capacity(usize::from(radius) * 2 + 1),
        })
    }

    pub fn push(&mut self, frame: T) {
        self.frames.push_back(frame);
    }

    pub fn is_ready(&self) -> bool {
        self.frames.len() >= self.radius * 2 + 1
    }

    pub fn center(&self) -> Option<&T> {
        self.is_ready().then(|| &self.frames[self.radius])
    }

    pub fn past(&self) -> Option<impl Iterator<Item = &T>> {
        self.is_ready()
            .then(|| self.frames.iter().take(self.radius))
    }

    pub fn future(&self) -> Option<impl Iterator<Item = &T>> {
        self.is_ready()
            .then(|| self.frames.iter().skip(self.radius + 1).take(self.radius))
    }

    pub fn advance(&mut self) -> Option<T> {
        self.is_ready().then(|| self.frames.pop_front()).flatten()
    }

    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_realtime_oriented_but_complete() {
        let options = Options::default();
        options.validate().unwrap();
        assert_eq!(options.degrain.radius, 4);
        assert_eq!(options.degrain.precision_bits, 8);
        assert_eq!(options.degrain.analysis_precision_bits, 8);
        assert!(options.degrain.async_pipeline);
        assert!(options.degrain.highres_fast);
        assert_eq!(options.reference_frames(), 8);
        assert_eq!(options.minimum_latency_ms(60.0), Some(1000.0 / 15.0));
    }

    #[test]
    fn radius_twenty_five_really_owns_fifty_references() {
        let mut window = TemporalWindow::new(25).unwrap();
        for frame in 0..51 {
            window.push(frame);
        }
        assert!(window.is_ready());
        assert_eq!(window.center(), Some(&25));
        assert_eq!(window.past().unwrap().count(), 25);
        assert_eq!(window.future().unwrap().count(), 25);
    }

    #[test]
    fn overlap_and_scene_percentage_follow_upstream_ranges() {
        let mut options = Options::default();
        options.super_clip.overlap = [9, 8];
        assert!(options.validate().is_err());
        options.super_clip.overlap = [8, 8];
        options.degrain.th_scd2 = 100.0;
        assert!(options.validate().is_ok());
        options.degrain.th_scd2 = 100.1;
        assert!(options.validate().is_err());
    }

    #[test]
    fn stage_parameters_support_requested_strong_profile() {
        let params = BTreeMap::from([
            ("radius".into(), 4.0),
            ("blksize_x".into(), 16.0),
            ("blksize_y".into(), 16.0),
            ("overlap_x".into(), 8.0),
            ("overlap_y".into(), 8.0),
            ("thsad_y".into(), 4000.0),
            ("thsad_c".into(), 4000.0),
            ("thscd1".into(), 1.0),
            ("thscd2".into(), 100.0),
        ]);
        let options = Options::from_stage_params(&params).unwrap();
        assert_eq!(options.degrain.radius, 4);
        assert_eq!(options.degrain.th_sad, [4000, 4000]);
        assert_eq!(options.degrain.th_scd1, 1);
        assert_eq!(options.degrain.th_scd2, 100.0);
    }

    #[test]
    fn temporal_weights_include_past_centre_and_future() {
        let mut params = BTreeMap::from([("radius".into(), 2.0)]);
        for index in 0..5 {
            params.insert(format!("weight_{index}"), (index + 1) as f32);
        }
        let options = Options::from_stage_params(&params).unwrap();
        assert_eq!(options.degrain.weights, Some(vec![1, 2, 3, 4, 5]));
        let mut invalid = options;
        invalid.degrain.weights = Some(vec![1, 2, 3, 4]);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn upstream_weight_curve_and_normalisation_are_preserved() {
        assert_eq!(degrain_weight(400, 0), 256);
        assert_eq!(degrain_weight(400, 400), 0);
        let mut refs = [256u16, 256, 0, 0];
        let src = normalise_weights(&mut refs, &[1, 1, 1, 1, 1]).unwrap();
        assert_eq!(
            u32::from(src) + refs.iter().map(|&v| u32::from(v)).sum::<u32>(),
            256
        );
        assert_eq!(&refs[2..], &[0, 0]);
    }

    #[test]
    fn block_grid_covers_partial_right_and_bottom_edges() {
        let grid = BlockGrid::new(1921, 1081, [16, 16], [8, 8]).unwrap();
        assert_eq!(grid.step_x, 8);
        assert_eq!(grid.step_y, 8);
        assert!(grid.blocks_x * 8 + 8 >= 1921);
        assert!(grid.blocks_y * 8 + 8 >= 1081);
        assert_eq!(grid.len(), grid.blocks_x as usize * grid.blocks_y as usize);
    }

    #[test]
    fn scene_change_100_percent_keeps_full_temporal_operation() {
        assert!(!is_scene_change(&[9999, 9999, 9999], 1, 100.0));
        assert!(is_scene_change(&[9999, 9999, 0], 1, 50.0));
    }

    #[test]
    fn degrain_sample_uses_all_references() {
        let mut weights = vec![256u16; 8];
        let source_weight = normalise_weights(&mut weights, &[1; 9]).unwrap();
        let out = degrain_u16_sample(1000, &[2000; 8], source_weight, &weights).unwrap();
        assert!(out > 1800, "all eight radius-4 references must contribute");
    }

    #[test]
    fn stage_queue_resets_on_gap_and_never_mixes_resolutions() {
        let mut queue = TemporalStageQueue::new(1).unwrap();
        let id = |sequence, width| FrameIdentity {
            sequence,
            width,
            height: 720,
        };
        assert_eq!(queue.push(id(1, 1280), 1), PushResult::Accepted);
        assert_eq!(queue.push(id(2, 1280), 2), PushResult::Accepted);
        assert_eq!(queue.push(id(3, 1280), 3), PushResult::Accepted);
        assert!(queue.ready());
        assert_eq!(queue.center().map(|frame| frame.1), Some(2));
        assert_eq!(queue.push(id(5, 1280), 5), PushResult::Reset);
        assert!(!queue.ready());
        assert_eq!(queue.push(id(6, 1920), 6), PushResult::Reset);
        assert!(!queue.ready());
    }

    #[test]
    fn portable_layout_never_resolves_from_system_installations() {
        let layout = PortableBackendLayout::under(Path::new("C:/Neo"));
        assert_eq!(
            layout.runtime,
            PathBuf::from("C:/Neo/backends/MVUtensils-Neo/vapoursynth.dll")
        );
        assert_eq!(
            layout.plugin,
            PathBuf::from("C:/Neo/backends/MVUtensils-Neo/mvutensils.dll")
        );
    }

    #[test]
    fn absent_portable_pack_is_a_quiet_none() {
        let root = std::env::temp_dir().join(format!(
            "neo-mvu-absent-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(PortableBackendLayout::discover(&root).unwrap(), None);
    }

    #[test]
    fn manifest_rejects_escape_from_portable_pack() {
        let root = std::env::temp_dir().join(format!("neo-mvu-escape-{}", std::process::id()));
        let pack = root.join("backends").join(FILTER_NAME);
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&pack).unwrap();
        std::fs::write(
            pack.join(BACKEND_MANIFEST),
            r#"{
                "schema_version":1,"backend_abi":1,"name":"MVUtensils-Neo",
                "library":"../evil.dll","runtime":"vapoursynth.dll",
                "plugin":"mvutensils.dll","license":"COPYING.GPL-2.0.txt"
            }"#,
        )
        .unwrap();
        let error = PortableBackendLayout::discover(&root).unwrap_err();
        assert!(error.contains("unsafe library path"));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
