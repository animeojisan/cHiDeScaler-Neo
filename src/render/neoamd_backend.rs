//! Optional NeoAMD Backend Pack boundary.
//!
//! Neo intentionally owns only discovery, ABI validation and fail-safe routing.
//! The external pack owns structural ONNX recognition, packed weights, HIP/WMMA
//! kernels and RDNA4 compatibility.  A model session is resolution-independent:
//! width/height are supplied only to `query_output`/`run_u8`, so enabling
//! NeoAMD never creates or serializes a per-resolution engine.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::{CString, c_char, c_void};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use windows::Win32::{
    Foundation::{FreeLibrary, HANDLE, HMODULE},
    System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        LoadLibraryExW,
    },
};
use windows::core::{PCSTR, PCWSTR};

pub const NEOAMD_BACKEND_API: u32 = 1;
pub const NEOAMD_BRIDGE_ABI: u32 = 1;
pub const NEOAMD_DEFAULT_BRIDGE: &str = "neo_amd_backend.dll";
pub const NEOAMD_VENDOR_AMD: u32 = 0x1002;

pub const NEOAMD_CAP_DYNAMIC_RESOLUTION: u64 = 1 << 0;
pub const NEOAMD_CAP_FP16_NATIVE: u64 = 1 << 1;
pub const NEOAMD_CAP_GFX12_WMMA: u64 = 1 << 2;
pub const NEOAMD_CAP_STRUCTURAL_ROUTING: u64 = 1 << 3;
pub const NEOAMD_CAP_D3D12_SHARED_BUFFER: u64 = 1 << 4;
pub const NEOAMD_CAP_D3D12_SHARED_INPUT: u64 = 1 << 5;
pub const NEOAMD_CAP_TEMPORAL_SHARED_IO: u64 = 1 << 6;
pub const NEOAMD_CAP_INTERP_SHARED_RGBA8: u64 = 1 << 7;

pub const NEOAMD_RUN_ROUTE_GRAPH_FULL: u32 = 1 << 0;
pub const NEOAMD_RUN_ROUTE_GRAPH_COMPUTE: u32 = 1 << 1;
pub const NEOAMD_RUN_ROUTE_DIRECT: u32 = 1 << 2;
pub const NEOAMD_RUN_ROUTE_RTMOSR_FP16: u32 = 1 << 3;
pub const NEOAMD_RUN_ROUTE_RTMOSR_U8_FUSED: u32 = 1 << 4;

pub const NEOAMD_SESSION_IMAGE: u32 = 0;
pub const NEOAMD_SESSION_INTERPOLATION: u32 = 1;
pub const NEOAMD_SESSION_TEMPORAL: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeoAmdManifest {
    pub backend_api: u32,
    pub name: String,
    pub architecture: String,
    pub bridge_abi: u32,
    #[serde(default = "default_bridge_name")]
    pub bridge: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub supported_gpu_architectures: Vec<String>,
    #[serde(default)]
    pub required_files: Vec<String>,
    #[serde(default)]
    pub file_sha256: BTreeMap<String, String>,
    #[serde(default)]
    pub compatibility_tags: Vec<String>,
}

fn default_bridge_name() -> String {
    NEOAMD_DEFAULT_BRIDGE.to_string()
}

#[derive(Clone, Debug, Default)]
pub struct NeoAmdAvailability {
    pub installed: bool,
    pub available: bool,
    pub reason: Option<String>,
    pub gpu_luid: Option<u64>,
    pub gpu_architecture: Option<String>,
    pub manifest: Option<NeoAmdManifest>,
    pub backend_dir: Option<PathBuf>,
}

fn unavailable(
    reason: impl Into<String>,
    installed: bool,
    manifest: Option<NeoAmdManifest>,
    backend_dir: Option<PathBuf>,
) -> NeoAmdAvailability {
    NeoAmdAvailability {
        installed,
        available: false,
        reason: Some(reason.into()),
        manifest,
        backend_dir,
        ..NeoAmdAvailability::default()
    }
}

fn safe_manifest_relative_path(file: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(file);
    if path.as_os_str().is_empty()
        || file.contains(':')
        || path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(format!("unsafe NeoAMD backend file path: {file}"));
    }
    Ok(path)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("{} could not be opened: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("{} could not be hashed: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:X}", hasher.finalize()))
}

fn read_manifest(app_dir: &Path) -> Result<(PathBuf, NeoAmdManifest), NeoAmdAvailability> {
    let backend_dir = app_dir.join("backends").join("neoamd");
    let manifest_path = backend_dir.join("backend.json");
    if !manifest_path.is_file() {
        return Err(unavailable(
            "NeoAMD Backend Pack is not installed",
            false,
            None,
            Some(backend_dir),
        ));
    }
    let bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(unavailable(
                format!("NeoAMD backend.json could not be read: {error}"),
                true,
                None,
                Some(backend_dir),
            ));
        }
    };
    let json = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(&bytes);
    let manifest: NeoAmdManifest = match serde_json::from_slice(json) {
        Ok(manifest) => manifest,
        Err(error) => {
            return Err(unavailable(
                format!("NeoAMD backend.json is invalid: {error}"),
                true,
                None,
                Some(backend_dir),
            ));
        }
    };
    if manifest.backend_api != NEOAMD_BACKEND_API {
        return Err(unavailable(
            format!(
                "NeoAMD backend API {} is incompatible (expected {})",
                manifest.backend_api, NEOAMD_BACKEND_API
            ),
            true,
            Some(manifest),
            Some(backend_dir),
        ));
    }
    if manifest.bridge_abi != NEOAMD_BRIDGE_ABI {
        return Err(unavailable(
            format!(
                "NeoAMD bridge ABI {} is incompatible (expected {})",
                manifest.bridge_abi, NEOAMD_BRIDGE_ABI
            ),
            true,
            Some(manifest),
            Some(backend_dir),
        ));
    }
    if !manifest.architecture.eq_ignore_ascii_case("x86_64") {
        return Err(unavailable(
            format!("unsupported NeoAMD host architecture: {}", manifest.architecture),
            true,
            Some(manifest),
            Some(backend_dir),
        ));
    }
    if !manifest
        .supported_gpu_architectures
        .iter()
        .any(|arch| arch.eq_ignore_ascii_case("gfx1200") || arch.eq_ignore_ascii_case("gfx1201"))
    {
        return Err(unavailable(
            "NeoAMD pack does not declare RDNA4 gfx1200/gfx1201 support",
            true,
            Some(manifest),
            Some(backend_dir),
        ));
    }
    let bridge = match safe_manifest_relative_path(&manifest.bridge) {
        Ok(path) => path,
        Err(error) => {
            return Err(unavailable(error, true, Some(manifest), Some(backend_dir)));
        }
    };
    if !backend_dir.join(&bridge).is_file() {
        return Err(unavailable(
            format!("NeoAMD bridge is missing: {}", manifest.bridge),
            true,
            Some(manifest),
            Some(backend_dir),
        ));
    }
    for file in &manifest.required_files {
        let relative = match safe_manifest_relative_path(file) {
            Ok(path) => path,
            Err(error) => {
                return Err(unavailable(error, true, Some(manifest), Some(backend_dir)));
            }
        };
        if !backend_dir.join(&relative).is_file() {
            return Err(unavailable(
                format!("required NeoAMD pack file is missing: {file}"),
                true,
                Some(manifest),
                Some(backend_dir),
            ));
        }
    }
    Ok((backend_dir, manifest))
}

fn verify_manifest_files(backend_dir: &Path, manifest: &NeoAmdManifest) -> Result<(), String> {
    // The bridge is always pinned.  Runtime-side data can be versioned by the
    // pack and listed in required_files; hashes are optional there so model
    // metadata can evolve without forcing a Neo executable update.
    let expected_bridge = manifest
        .file_sha256
        .get(&manifest.bridge)
        .ok_or_else(|| "NeoAMD backend.json is missing the bridge SHA-256".to_string())?;
    let bridge_path = backend_dir.join(safe_manifest_relative_path(&manifest.bridge)?);
    let actual = sha256_file(&bridge_path)?;
    if !actual.eq_ignore_ascii_case(expected_bridge.trim()) {
        return Err("NeoAMD bridge does not match the Backend Pack manifest".to_string());
    }
    for (file, expected) in &manifest.file_sha256 {
        let relative = safe_manifest_relative_path(file)?;
        let path = backend_dir.join(relative);
        if !path.is_file() {
            return Err(format!("hashed NeoAMD pack file is missing: {file}"));
        }
        let actual = sha256_file(&path)?;
        if !actual.eq_ignore_ascii_case(expected.trim()) {
            return Err(format!("NeoAMD pack file does not match its manifest: {file}"));
        }
    }
    Ok(())
}

fn selected_amd_luid(selected_luid: Option<u64>) -> Result<u64, String> {
    let adapters = crate::platform::gpu::enumerate_adapters_quiet();
    if let Some(luid) = selected_luid {
        let adapter = adapters
            .iter()
            .find(|adapter| adapter.luid == luid)
            .ok_or_else(|| format!("selected GPU LUID {luid:016x} is unavailable"))?;
        if adapter.vendor_id != NEOAMD_VENDOR_AMD {
            return Err(format!("selected GPU '{}' is not an AMD GPU", adapter.name));
        }
        return Ok(luid);
    }
    let mut amd = adapters
        .iter()
        .filter(|adapter| adapter.vendor_id == NEOAMD_VENDOR_AMD);
    let first = amd
        .next()
        .ok_or_else(|| "no AMD GPU is available for NeoAMD".to_string())?;
    if amd.next().is_some() {
        return Err("Auto GPU is ambiguous across multiple AMD adapters".to_string());
    }
    Ok(first.luid)
}

/// Full pack discovery.  Missing packs are normal and leave DirectML untouched.
/// A present pack is loaded only long enough to ask the bridge whether the
/// selected AMD adapter is a supported RDNA4 architecture.
pub fn detect_neoamd_backend(app_dir: &Path, selected_luid: Option<u64>) -> NeoAmdAvailability {
    let (backend_dir, manifest) = match read_manifest(app_dir) {
        Ok(value) => value,
        Err(availability) => return availability,
    };
    if let Err(error) = verify_manifest_files(&backend_dir, &manifest) {
        return unavailable(error, true, Some(manifest), Some(backend_dir));
    }
    let gpu_luid = match selected_amd_luid(selected_luid) {
        Ok(luid) => luid,
        Err(error) => return unavailable(error, true, Some(manifest), Some(backend_dir)),
    };
    let backend = match NeoAmdBackend::load(&backend_dir, &manifest) {
        Ok(backend) => backend,
        Err(error) => return unavailable(error, true, Some(manifest), Some(backend_dir)),
    };
    let info = match backend.query_adapter(gpu_luid) {
        Ok(info) => info,
        Err(error) => return unavailable(error, true, Some(manifest), Some(backend_dir)),
    };
    let architecture = info.architecture_string();
    if info.vendor_id != NEOAMD_VENDOR_AMD
        || !matches!(architecture.as_str(), "gfx1200" | "gfx1201")
    {
        return unavailable(
            format!("selected AMD GPU is not a supported RDNA4 target ({architecture})"),
            true,
            Some(manifest),
            Some(backend_dir),
        );
    }
    let required_caps = NEOAMD_CAP_DYNAMIC_RESOLUTION
        | NEOAMD_CAP_FP16_NATIVE
        | NEOAMD_CAP_GFX12_WMMA
        | NEOAMD_CAP_STRUCTURAL_ROUTING;
    if info.capabilities & required_caps != required_caps {
        return unavailable(
            format!(
                "NeoAMD bridge is missing required RDNA4 production capabilities (reported=0x{:x}, required=0x{:x})",
                info.capabilities, required_caps
            ),
            true,
            Some(manifest),
            Some(backend_dir),
        );
    }
    if !manifest
        .supported_gpu_architectures
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(&architecture))
    {
        return unavailable(
            format!("NeoAMD pack does not list the detected architecture {architecture}"),
            true,
            Some(manifest),
            Some(backend_dir),
        );
    }
    NeoAmdAvailability {
        installed: true,
        available: true,
        reason: None,
        gpu_luid: Some(gpu_luid),
        gpu_architecture: Some(architecture),
        manifest: Some(manifest),
        backend_dir: Some(backend_dir),
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NeoAmdAdapterInfo {
    pub struct_size: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub reserved0: u32,
    pub capabilities: u64,
    pub architecture: [u8; 32],
    pub reserved: [u64; 6],
}

impl Default for NeoAmdAdapterInfo {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            vendor_id: 0,
            device_id: 0,
            reserved0: 0,
            capabilities: 0,
            architecture: [0; 32],
            reserved: [0; 6],
        }
    }
}

impl NeoAmdAdapterInfo {
    pub fn architecture_string(&self) -> String {
        let end = self
            .architecture
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(self.architecture.len());
        String::from_utf8_lossy(&self.architecture[..end])
            .trim()
            .to_ascii_lowercase()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NeoAmdSessionInfo {
    pub struct_size: u32,
    pub kind: u32,
    pub temporal_frames: u32,
    pub min_inputs: u32,
    pub max_inputs: u32,
    pub scale_num: u32,
    pub scale_den: u32,
    pub reserved0: u32,
    pub capabilities: u64,
    pub reserved: [u64; 6],
}

#[repr(C)]
struct NeoAmdCreateDesc {
    struct_size: u32,
    model_path_utf16: *const u16,
    adapter_luid: u64,
    flags: u64,
    reserved: [u64; 6],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NeoAmdRunDesc {
    struct_size: u32,
    width: u32,
    height: u32,
    input_count: u32,
    input_pixel_stride: u32,
    output_pixel_stride: u32,
    phase: f32,
    flags: u32,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NeoAmdOutputDesc {
    struct_size: u32,
    width: u32,
    height: u32,
    required_bytes: u64,
    reserved: [u64; 4],
}

pub const NEOAMD_SHARED_NCHW_FP16: u32 = 1;
pub const NEOAMD_SHARED_RGBA8: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NeoAmdSharedOutputDesc {
    struct_size: u32,
    width: u32,
    height: u32,
    format: u32,
    resource_key: u64,
    byte_len: u64,
    allocation_byte_len: u64,
    reserved: [u64; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct NeoAmdSharedOutput {
    pub width: i32,
    pub height: i32,
    pub resource_key: u64,
    pub byte_len: usize,
    pub allocation_byte_len: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct NeoAmdSharedIo {
    pub input: NeoAmdSharedOutput,
    pub output: NeoAmdSharedOutput,
}

type GetApiVersionFn = unsafe extern "C" fn() -> u32;
type QueryAdapterFn = unsafe extern "C" fn(u64, *mut NeoAmdAdapterInfo) -> i32;
type CreateFn =
    unsafe extern "C" fn(*const NeoAmdCreateDesc, *mut *mut c_void, *mut NeoAmdSessionInfo) -> i32;
type QueryOutputFn =
    unsafe extern "C" fn(*mut c_void, *const NeoAmdRunDesc, *mut NeoAmdOutputDesc) -> i32;
type RunU8Fn = unsafe extern "C" fn(
    *mut c_void,
    *const NeoAmdRunDesc,
    *const *const u8,
    *mut u8,
    u64,
) -> i32;
type RunInterpManyU8Fn = unsafe extern "C" fn(
    *mut c_void,
    *const NeoAmdRunDesc,
    *const *const u8,
    *const f32,
    u32,
    *mut u8,
    u64,
    u64,
) -> i32;
type PrepareInterpStreamU8Fn = unsafe extern "C" fn(
    *mut c_void,
    *const NeoAmdRunDesc,
    *const *const u8,
) -> i32;
type RunInterpStreamPhaseU8Fn = unsafe extern "C" fn(
    *mut c_void,
    *const NeoAmdRunDesc,
    f32,
    *mut u8,
    u64,
) -> i32;
type PrepareInterpStreamSharedRgba8Fn = unsafe extern "C" fn(
    *mut c_void,
    *const NeoAmdRunDesc,
    *const *const u8,
    *mut NeoAmdSharedOutputDesc,
) -> i32;
type RunInterpStreamPhaseSharedRgba8Fn =
    unsafe extern "C" fn(*mut c_void, *const NeoAmdRunDesc, f32, u64) -> i32;
type PrepareSharedFp16Fn = unsafe extern "C" fn(
    *mut c_void,
    *const NeoAmdRunDesc,
    *mut NeoAmdSharedOutputDesc,
) -> i32;
type GetSharedOutputHandleFn = unsafe extern "C" fn(*mut c_void, u64, *mut u64) -> i32;
type RunU8SharedFp16Fn = unsafe extern "C" fn(
    *mut c_void,
    *const NeoAmdRunDesc,
    *const *const u8,
    u64,
) -> i32;
type PrepareSharedIoFp16Fn = unsafe extern "C" fn(
    *mut c_void,
    *const NeoAmdRunDesc,
    *mut NeoAmdSharedOutputDesc,
    *mut NeoAmdSharedOutputDesc,
) -> i32;
type GetSharedInputHandleFn = unsafe extern "C" fn(*mut c_void, u64, *mut u64) -> i32;
type RunSharedIoFp16Fn =
    unsafe extern "C" fn(*mut c_void, *const NeoAmdRunDesc, u64, u64) -> i32;
type PrepareTemporalSharedIoFp16Fn = unsafe extern "C" fn(
    *mut c_void,
    *const NeoAmdRunDesc,
    *mut NeoAmdSharedOutputDesc,
    *mut NeoAmdSharedOutputDesc,
) -> i32;
type RunTemporalSharedIoFp16Fn =
    unsafe extern "C" fn(*mut c_void, *const NeoAmdRunDesc, u64, u64) -> i32;
type RunPeerSharedIoFp16Fn =
    unsafe extern "C" fn(*mut c_void, *const NeoAmdRunDesc, u64, u64) -> i32;
type ResetTemporalHistoryFn = unsafe extern "C" fn(*mut c_void) -> i32;
type ReleaseSharedOutputFn = unsafe extern "C" fn(*mut c_void);
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type LastErrorFn = unsafe extern "C" fn(*mut c_void, *mut c_char, u32) -> u32;
type DescribeBackendFn = unsafe extern "C" fn(*mut c_char, u32) -> u32;
type GetLastRunRouteFn = unsafe extern "C" fn(*mut c_void) -> u32;
type GetLastImageGpuMsFn = unsafe extern "C" fn(*mut c_void) -> f32;
type GetLastImageCopyGpuMsFn = unsafe extern "C" fn(*mut c_void) -> f32;
type GetLastImageComputeGpuMsFn = unsafe extern "C" fn(*mut c_void) -> f32;

unsafe fn symbol<T: Copy>(module: HMODULE, name: &str) -> Result<T, String> {
    let printable = name.to_string();
    let name = CString::new(name).map_err(|_| format!("invalid export name: {printable}"))?;
    let raw = unsafe { GetProcAddress(module, PCSTR(name.as_ptr().cast())) }
        .ok_or_else(|| format!("required NeoAMD bridge export is missing: {printable}"))?;
    Ok(unsafe { std::mem::transmute_copy(&raw) })
}

unsafe fn optional_symbol<T: Copy>(module: HMODULE, name: &str) -> Option<T> {
    let name = CString::new(name).ok()?;
    let raw = unsafe { GetProcAddress(module, PCSTR(name.as_ptr().cast())) }?;
    Some(unsafe { std::mem::transmute_copy(&raw) })
}

fn wide(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

pub struct NeoAmdBackend {
    module: HMODULE,
    query_adapter: QueryAdapterFn,
    create: CreateFn,
    query_output: QueryOutputFn,
    run_u8: RunU8Fn,
    run_interp_many_u8: Option<RunInterpManyU8Fn>,
    prepare_interp_stream_u8: Option<PrepareInterpStreamU8Fn>,
    run_interp_stream_phase_u8: Option<RunInterpStreamPhaseU8Fn>,
    prepare_interp_stream_shared_rgba8: Option<PrepareInterpStreamSharedRgba8Fn>,
    run_interp_stream_phase_shared_rgba8: Option<RunInterpStreamPhaseSharedRgba8Fn>,
    prepare_shared_fp16: Option<PrepareSharedFp16Fn>,
    get_shared_output_handle: Option<GetSharedOutputHandleFn>,
    run_u8_shared_fp16: Option<RunU8SharedFp16Fn>,
    prepare_shared_io_fp16: Option<PrepareSharedIoFp16Fn>,
    get_shared_input_handle: Option<GetSharedInputHandleFn>,
    run_shared_io_fp16: Option<RunSharedIoFp16Fn>,
    prepare_temporal_shared_io_fp16: Option<PrepareTemporalSharedIoFp16Fn>,
    run_temporal_shared_io_fp16: Option<RunTemporalSharedIoFp16Fn>,
    run_peer_shared_io_fp16: Option<RunPeerSharedIoFp16Fn>,
    reset_temporal_history: Option<ResetTemporalHistoryFn>,
    release_shared_output: Option<ReleaseSharedOutputFn>,
    destroy: DestroyFn,
    last_error: LastErrorFn,
    describe_backend: Option<DescribeBackendFn>,
    get_last_run_route: Option<GetLastRunRouteFn>,
    get_last_image_gpu_ms: Option<GetLastImageGpuMsFn>,
    get_last_image_copy_gpu_ms: Option<GetLastImageCopyGpuMsFn>,
    get_last_image_compute_gpu_ms: Option<GetLastImageComputeGpuMsFn>,
}

// The bridge API is immutable after load. A session itself remains serialized
// by OnnxStage's existing Mutex; sharing the loaded module/function table is safe.
unsafe impl Send for NeoAmdBackend {}
unsafe impl Sync for NeoAmdBackend {}

impl NeoAmdBackend {
    fn load(backend_dir: &Path, manifest: &NeoAmdManifest) -> Result<Arc<Self>, String> {
        // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR rejects a relative path with
        // ERROR_INVALID_PARAMETER. Developer probes and portable launchers may
        // supply a relative app directory, so normalize it without broadening
        // the DLL search path beyond this verified pack directory.
        let bridge_path = backend_dir
            .join(safe_manifest_relative_path(&manifest.bridge)?)
            .canonicalize()
            .map_err(|error| format!("NeoAMD bridge path could not be resolved: {error}"))?;
        let encoded = wide(&bridge_path);
        let module = unsafe {
            LoadLibraryExW(
                PCWSTR(encoded.as_ptr()),
                None,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        }
        .map_err(|error| format!("{} could not be loaded: {error}", bridge_path.display()))?;
        let loaded = (|| unsafe {
            let get_api_version: GetApiVersionFn = symbol(module, "neoamd_get_api_version")?;
            let actual = get_api_version();
            if actual != NEOAMD_BRIDGE_ABI {
                return Err(format!(
                    "NeoAMD bridge ABI {actual} is incompatible (expected {NEOAMD_BRIDGE_ABI})"
                ));
            }
            Ok(Arc::new(Self {
                module,
                query_adapter: symbol(module, "neoamd_query_adapter")?,
                create: symbol(module, "neoamd_create_session")?,
                query_output: symbol(module, "neoamd_query_output")?,
                run_u8: symbol(module, "neoamd_run_u8")?,
                run_interp_many_u8: optional_symbol(module, "neoamd_run_interp_many_u8"),
                prepare_interp_stream_u8: optional_symbol(module, "neoamd_prepare_interp_stream_u8"),
                run_interp_stream_phase_u8: optional_symbol(module, "neoamd_run_interp_stream_phase_u8"),
                prepare_interp_stream_shared_rgba8: optional_symbol(module, "neoamd_prepare_interp_stream_shared_rgba8"),
                run_interp_stream_phase_shared_rgba8: optional_symbol(module, "neoamd_run_interp_stream_phase_shared_rgba8"),
                prepare_shared_fp16: optional_symbol(module, "neoamd_prepare_shared_fp16"),
                get_shared_output_handle: optional_symbol(module, "neoamd_get_shared_output_handle"),
                run_u8_shared_fp16: optional_symbol(module, "neoamd_run_u8_shared_fp16"),
                prepare_shared_io_fp16: optional_symbol(module, "neoamd_prepare_shared_io_fp16"),
                get_shared_input_handle: optional_symbol(module, "neoamd_get_shared_input_handle"),
                run_shared_io_fp16: optional_symbol(module, "neoamd_run_shared_io_fp16"),
                prepare_temporal_shared_io_fp16: optional_symbol(module, "neoamd_prepare_temporal_shared_io_fp16"),
                run_temporal_shared_io_fp16: optional_symbol(module, "neoamd_run_temporal_shared_io_fp16"),
                run_peer_shared_io_fp16: optional_symbol(module, "neoamd_run_peer_shared_io_fp16"),
                reset_temporal_history: optional_symbol(module, "neoamd_reset_temporal_history"),
                release_shared_output: optional_symbol(module, "neoamd_release_shared_output"),
                destroy: symbol(module, "neoamd_destroy_session")?,
                last_error: symbol(module, "neoamd_last_error")?,
                describe_backend: optional_symbol(module, "neoamd_describe_backend"),
                get_last_run_route: optional_symbol(module, "neoamd_get_last_run_route"),
                get_last_image_gpu_ms: optional_symbol(module, "neoamd_get_last_image_gpu_ms"),
                get_last_image_copy_gpu_ms: optional_symbol(module, "neoamd_get_last_image_copy_gpu_ms"),
                get_last_image_compute_gpu_ms: optional_symbol(module, "neoamd_get_last_image_compute_gpu_ms"),
            }))
        })();
        if loaded.is_err() {
            unsafe {
                let _ = FreeLibrary(module);
            }
        }
        loaded
    }

    pub fn load_from_app_dir(app_dir: &Path) -> Result<Arc<Self>, String> {
        let (backend_dir, manifest) = read_manifest(app_dir).map_err(|a| {
            a.reason
                .unwrap_or_else(|| "NeoAMD Backend Pack is unavailable".to_string())
        })?;
        verify_manifest_files(&backend_dir, &manifest)?;
        Self::load(&backend_dir, &manifest)
    }

    pub fn query_adapter(&self, luid: u64) -> Result<NeoAmdAdapterInfo, String> {
        let mut info = NeoAmdAdapterInfo::default();
        let status = unsafe { (self.query_adapter)(luid, &mut info) };
        if status == 0 {
            Ok(info)
        } else {
            Err(self.error_string(std::ptr::null_mut(), status, "query-adapter"))
        }
    }

    pub fn description(&self) -> Option<String> {
        let function = self.describe_backend?;
        let mut buffer = [0u8; 2048];
        let written = unsafe { function(buffer.as_mut_ptr().cast(), buffer.len() as u32) } as usize;
        if written == 0 {
            return None;
        }
        let len = written.min(buffer.len()).min(
            buffer
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(buffer.len()),
        );
        let value = String::from_utf8_lossy(&buffer[..len]).trim().to_string();
        (!value.is_empty()).then_some(value)
    }

    pub fn create_session(
        self: &Arc<Self>,
        model_path: &Path,
        adapter_luid: u64,
    ) -> Result<NeoAmdSession, String> {
        let model_path_utf16 = wide(model_path);
        let desc = NeoAmdCreateDesc {
            struct_size: std::mem::size_of::<NeoAmdCreateDesc>() as u32,
            model_path_utf16: model_path_utf16.as_ptr(),
            adapter_luid,
            flags: 0,
            reserved: [0; 6],
        };
        let mut context = std::ptr::null_mut();
        let mut info = NeoAmdSessionInfo {
            struct_size: std::mem::size_of::<NeoAmdSessionInfo>() as u32,
            ..NeoAmdSessionInfo::default()
        };
        let status = unsafe { (self.create)(&desc, &mut context, &mut info) };
        if status != 0 || context.is_null() {
            let error = self.error_string(context, status, "create-session");
            if !context.is_null() {
                unsafe { (self.destroy)(context) };
            }
            return Err(error);
        }
        if info.capabilities & NEOAMD_CAP_DYNAMIC_RESOLUTION == 0 {
            unsafe { (self.destroy)(context) };
            return Err("NeoAMD model session is not dynamic-resolution capable".to_string());
        }
        Ok(NeoAmdSession {
            backend: self.clone(),
            context,
            info,
            adapter_luid,
        })
    }

    fn error_string(&self, context: *mut c_void, status: i32, operation: &str) -> String {
        let mut buffer = [0u8; 1024];
        let written = unsafe {
            (self.last_error)(
                context,
                buffer.as_mut_ptr().cast::<c_char>(),
                buffer.len() as u32,
            )
        } as usize;
        let message = if written == 0 {
            String::new()
        } else {
            let len = written.min(buffer.len()).min(
                buffer
                    .iter()
                    .position(|byte| *byte == 0)
                    .unwrap_or(buffer.len()),
            );
            String::from_utf8_lossy(&buffer[..len]).trim().to_string()
        };
        if message.is_empty() {
            format!("NeoAMD {operation} failed with bridge status {status}")
        } else {
            format!("NeoAMD {operation} failed with bridge status {status}: {message}")
        }
    }
}

impl Drop for NeoAmdBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = FreeLibrary(self.module);
        }
    }
}

pub struct NeoAmdSession {
    backend: Arc<NeoAmdBackend>,
    context: *mut c_void,
    info: NeoAmdSessionInfo,
    adapter_luid: u64,
}

unsafe impl Send for NeoAmdSession {}

impl NeoAmdSession {
    pub fn info(&self) -> NeoAmdSessionInfo {
        self.info
    }

    pub fn adapter_luid(&self) -> u64 {
        self.adapter_luid
    }

    pub fn last_run_route(&self) -> Option<u32> {
        self.backend
            .get_last_run_route
            .map(|function| unsafe { function(self.context) })
    }

    pub fn last_image_gpu_ms(&self) -> Option<f32> {
        self.backend
            .get_last_image_gpu_ms
            .map(|function| unsafe { function(self.context) })
            .filter(|value| value.is_finite() && *value > 0.0)
    }

    pub fn last_image_copy_gpu_ms(&self) -> Option<f32> {
        self.backend
            .get_last_image_copy_gpu_ms
            .map(|function| unsafe { function(self.context) })
            .filter(|value| value.is_finite() && *value > 0.0)
    }

    pub fn last_image_compute_gpu_ms(&self) -> Option<f32> {
        self.backend
            .get_last_image_compute_gpu_ms
            .map(|function| unsafe { function(self.context) })
            .filter(|value| value.is_finite() && *value > 0.0)
    }

    pub fn supports_shared_fp16(&self) -> bool {
        self.info.capabilities & NEOAMD_CAP_D3D12_SHARED_BUFFER != 0
            && self.backend.prepare_shared_fp16.is_some()
            && self.backend.get_shared_output_handle.is_some()
            && self.backend.run_u8_shared_fp16.is_some()
    }

    pub fn supports_shared_io_fp16(&self) -> bool {
        self.supports_shared_fp16()
            && self.info.capabilities & NEOAMD_CAP_D3D12_SHARED_INPUT != 0
            && self.backend.prepare_shared_io_fp16.is_some()
            && self.backend.get_shared_input_handle.is_some()
            && self.backend.run_shared_io_fp16.is_some()
    }

    pub fn supports_peer_shared_io_fp16(&self) -> bool {
        self.info.capabilities & NEOAMD_CAP_D3D12_SHARED_BUFFER != 0
            && self.backend.run_peer_shared_io_fp16.is_some()
    }

    pub fn supports_temporal_shared_io_fp16(&self) -> bool {
        self.info.kind == NEOAMD_SESSION_TEMPORAL
            && self.info.capabilities & NEOAMD_CAP_D3D12_SHARED_BUFFER != 0
            && self.info.capabilities & NEOAMD_CAP_D3D12_SHARED_INPUT != 0
            && self.info.capabilities & NEOAMD_CAP_TEMPORAL_SHARED_IO != 0
            && self.backend.prepare_temporal_shared_io_fp16.is_some()
            && self.backend.get_shared_output_handle.is_some()
            && self.backend.get_shared_input_handle.is_some()
            && self.backend.run_temporal_shared_io_fp16.is_some()
    }

    fn shared_surface(desc: NeoAmdSharedOutputDesc, label: &str) -> Result<NeoAmdSharedOutput, String> {
        if desc.format != NEOAMD_SHARED_NCHW_FP16 || desc.width == 0 || desc.height == 0 {
            return Err(format!("NeoAMD bridge returned an invalid {label} FP16 surface"));
        }
        let minimum = (desc.width as u64)
            .checked_mul(desc.height as u64)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| format!("NeoAMD {label} shared surface size overflow"))?;
        if desc.byte_len < minimum || desc.allocation_byte_len < desc.byte_len || desc.resource_key == 0 {
            return Err(format!("NeoAMD {label} shared surface allocation is invalid"));
        }
        Ok(NeoAmdSharedOutput {
            width: desc.width as i32,
            height: desc.height as i32,
            resource_key: desc.resource_key,
            byte_len: usize::try_from(desc.byte_len)
                .map_err(|_| format!("NeoAMD {label} shared surface is too large"))?,
            allocation_byte_len: desc.allocation_byte_len,
        })
    }

    fn run_desc(
        width: i32,
        height: i32,
        input_count: usize,
        input_pixel_stride: usize,
        output_pixel_stride: usize,
        phase: f32,
    ) -> Result<NeoAmdRunDesc, String> {
        if width <= 0 || height <= 0 || input_count == 0 || input_pixel_stride < 3 {
            return Err("NeoAMD shared run descriptor is invalid".to_string());
        }
        Ok(NeoAmdRunDesc {
            struct_size: std::mem::size_of::<NeoAmdRunDesc>() as u32,
            width: width as u32,
            height: height as u32,
            input_count: input_count as u32,
            input_pixel_stride: input_pixel_stride as u32,
            output_pixel_stride: output_pixel_stride as u32,
            phase: phase.clamp(0.0, 1.0),
            flags: 0,
            reserved: [0; 4],
        })
    }

    pub fn prepare_shared_fp16(
        &mut self,
        width: i32,
        height: i32,
        input_count: usize,
        input_pixel_stride: usize,
        phase: f32,
    ) -> Result<NeoAmdSharedOutput, String> {
        if !self.supports_shared_fp16() {
            return Err("NeoAMD shared FP16 output is unavailable".to_string());
        }
        let desc = Self::run_desc(width, height, input_count, input_pixel_stride, 3, phase)?;
        let mut shared = NeoAmdSharedOutputDesc {
            struct_size: std::mem::size_of::<NeoAmdSharedOutputDesc>() as u32,
            ..NeoAmdSharedOutputDesc::default()
        };
        let status = unsafe {
            self.backend.prepare_shared_fp16.unwrap()(self.context, &desc, &mut shared)
        };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "prepare-shared-fp16"));
        }
        if shared.format != NEOAMD_SHARED_NCHW_FP16 || shared.width == 0 || shared.height == 0 {
            return Err("NeoAMD bridge returned an invalid shared FP16 surface".to_string());
        }
        let minimum = (shared.width as u64)
            .checked_mul(shared.height as u64)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| "NeoAMD shared output size overflow".to_string())?;
        if shared.byte_len < minimum || shared.allocation_byte_len < shared.byte_len {
            return Err("NeoAMD shared output allocation is too small".to_string());
        }
        Ok(NeoAmdSharedOutput {
            width: shared.width as i32,
            height: shared.height as i32,
            resource_key: shared.resource_key,
            byte_len: usize::try_from(shared.byte_len)
                .map_err(|_| "NeoAMD shared output is too large".to_string())?,
            allocation_byte_len: shared.allocation_byte_len,
        })
    }

    pub fn shared_output_handle(&mut self, resource_key: u64) -> Result<HANDLE, String> {
        let function = self
            .backend
            .get_shared_output_handle
            .ok_or_else(|| "NeoAMD shared output handle export is unavailable".to_string())?;
        let mut raw = 0u64;
        let status = unsafe { function(self.context, resource_key, &mut raw) };
        if status != 0 || raw == 0 {
            return Err(self.backend.error_string(self.context, status, "shared-output-handle"));
        }
        Ok(HANDLE(raw as *mut c_void))
    }

    pub fn prepare_shared_io_fp16(
        &mut self,
        width: i32,
        height: i32,
        phase: f32,
    ) -> Result<NeoAmdSharedIo, String> {
        if !self.supports_shared_io_fp16() {
            return Err("NeoAMD shared FP16 input/output is unavailable".to_string());
        }
        let desc = Self::run_desc(width, height, 1, 4, 3, phase)?;
        let mut input = NeoAmdSharedOutputDesc {
            struct_size: std::mem::size_of::<NeoAmdSharedOutputDesc>() as u32,
            ..NeoAmdSharedOutputDesc::default()
        };
        let mut output = NeoAmdSharedOutputDesc {
            struct_size: std::mem::size_of::<NeoAmdSharedOutputDesc>() as u32,
            ..NeoAmdSharedOutputDesc::default()
        };
        let status = unsafe {
            self.backend.prepare_shared_io_fp16.unwrap()(
                self.context,
                &desc,
                &mut input,
                &mut output,
            )
        };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "prepare-shared-io-fp16"));
        }
        let input = Self::shared_surface(input, "input")?;
        let output = Self::shared_surface(output, "output")?;
        if input.width != width || input.height != height {
            return Err("NeoAMD shared input geometry changed unexpectedly".to_string());
        }
        Ok(NeoAmdSharedIo { input, output })
    }

    pub fn prepare_temporal_shared_io_fp16(
        &mut self,
        width: i32,
        height: i32,
    ) -> Result<NeoAmdSharedIo, String> {
        if !self.supports_temporal_shared_io_fp16() {
            return Err("NeoAMD TemporalFix shared FP16 input/output is unavailable".to_string());
        }
        let desc = Self::run_desc(width, height, 1, 4, 3, 0.5)?;
        let mut input = NeoAmdSharedOutputDesc {
            struct_size: std::mem::size_of::<NeoAmdSharedOutputDesc>() as u32,
            ..NeoAmdSharedOutputDesc::default()
        };
        let mut output = NeoAmdSharedOutputDesc {
            struct_size: std::mem::size_of::<NeoAmdSharedOutputDesc>() as u32,
            ..NeoAmdSharedOutputDesc::default()
        };
        let status = unsafe {
            self.backend.prepare_temporal_shared_io_fp16.unwrap()(
                self.context,
                &desc,
                &mut input,
                &mut output,
            )
        };
        if status != 0 {
            return Err(self.backend.error_string(
                self.context,
                status,
                "prepare-temporal-shared-io-fp16",
            ));
        }
        let input = Self::shared_surface(input, "temporal-input")?;
        let output = Self::shared_surface(output, "temporal-output")?;
        if input.width != width
            || input.height != height
            || output.width != width
            || output.height != height
        {
            return Err("NeoAMD TemporalFix shared IO geometry changed unexpectedly".to_string());
        }
        Ok(NeoAmdSharedIo { input, output })
    }

    pub fn shared_input_handle(&mut self, resource_key: u64) -> Result<HANDLE, String> {
        let function = self
            .backend
            .get_shared_input_handle
            .ok_or_else(|| "NeoAMD shared input handle export is unavailable".to_string())?;
        let mut raw = 0u64;
        let status = unsafe { function(self.context, resource_key, &mut raw) };
        if status != 0 || raw == 0 {
            return Err(self.backend.error_string(self.context, status, "shared-input-handle"));
        }
        Ok(HANDLE(raw as *mut c_void))
    }

    pub fn run_shared_io_fp16(
        &mut self,
        width: i32,
        height: i32,
        phase: f32,
        input_resource_key: u64,
        output_resource_key: u64,
    ) -> Result<(), String> {
        let function = self
            .backend
            .run_shared_io_fp16
            .ok_or_else(|| "NeoAMD shared IO run export is unavailable".to_string())?;
        let desc = Self::run_desc(width, height, 1, 4, 3, phase)?;
        let status = unsafe {
            function(
                self.context,
                &desc,
                input_resource_key,
                output_resource_key,
            )
        };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "run-shared-io-fp16"));
        }
        Ok(())
    }

    pub fn run_temporal_shared_io_fp16(
        &mut self,
        width: i32,
        height: i32,
        input_resource_key: u64,
        output_resource_key: u64,
    ) -> Result<(), String> {
        let function = self
            .backend
            .run_temporal_shared_io_fp16
            .ok_or_else(|| "NeoAMD TemporalFix shared IO run export is unavailable".to_string())?;
        let desc = Self::run_desc(width, height, 1, 4, 3, 0.5)?;
        let status = unsafe {
            function(
                self.context,
                &desc,
                input_resource_key,
                output_resource_key,
            )
        };
        if status != 0 {
            return Err(self.backend.error_string(
                self.context,
                status,
                "run-temporal-shared-io-fp16",
            ));
        }
        Ok(())
    }

    pub fn run_peer_shared_io_fp16(
        &mut self,
        width: i32,
        height: i32,
        source_resource_key: u64,
        output_resource_key: u64,
    ) -> Result<(), String> {
        let function = self
            .backend
            .run_peer_shared_io_fp16
            .ok_or_else(|| "NeoAMD peer shared IO run export is unavailable".to_string())?;
        let desc = Self::run_desc(width, height, 1, 4, 3, 0.5)?;
        let status = unsafe {
            function(self.context, &desc, source_resource_key, output_resource_key)
        };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "run-peer-shared-io-fp16"));
        }
        Ok(())
    }

    pub fn reset_temporal_history(&mut self) -> Result<(), String> {
        let Some(function) = self.backend.reset_temporal_history else {
            // Optional ABI-1 v054 aid. Older packs can still run through the
            // CPU-visible path; they simply cannot preserve a warmed shared
            // Graph while clearing the Backend-owned TRUE-7 ring.
            return Ok(());
        };
        let status = unsafe { function(self.context) };
        if status != 0 {
            return Err(self.backend.error_string(
                self.context,
                status,
                "reset-temporal-history",
            ));
        }
        Ok(())
    }

    pub fn release_shared_output(&mut self) {
        if let Some(function) = self.backend.release_shared_output {
            unsafe { function(self.context) };
        }
    }

    pub fn run_u8_shared_fp16(
        &mut self,
        width: i32,
        height: i32,
        inputs: &[&[u8]],
        input_pixel_stride: usize,
        phase: f32,
        resource_key: u64,
    ) -> Result<(), String> {
        let function = self
            .backend
            .run_u8_shared_fp16
            .ok_or_else(|| "NeoAMD shared FP16 run export is unavailable".to_string())?;
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "NeoAMD input size overflow".to_string())?;
        let required = pixels
            .checked_mul(input_pixel_stride)
            .ok_or_else(|| "NeoAMD input byte size overflow".to_string())?;
        for input in inputs {
            if input.len() < required {
                return Err("NeoAMD shared input buffer is too small".to_string());
            }
        }
        let desc = Self::run_desc(width, height, inputs.len(), input_pixel_stride, 3, phase)?;
        let pointers: Vec<*const u8> = inputs.iter().map(|input| input.as_ptr()).collect();
        let status = unsafe { function(self.context, &desc, pointers.as_ptr(), resource_key) };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "run-shared-fp16"));
        }
        Ok(())
    }

    fn shared_rgba8_surface(
        desc: NeoAmdSharedOutputDesc,
        label: &str,
    ) -> Result<NeoAmdSharedOutput, String> {
        if desc.format != NEOAMD_SHARED_RGBA8 || desc.width == 0 || desc.height == 0 {
            return Err(format!("NeoAMD bridge returned an invalid {label} RGBA8 surface"));
        }
        let minimum = (desc.width as u64)
            .checked_mul(desc.height as u64)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| format!("NeoAMD {label} RGBA8 surface size overflow"))?;
        if desc.byte_len < minimum
            || desc.allocation_byte_len < desc.byte_len
            || desc.resource_key == 0
        {
            return Err(format!("NeoAMD {label} RGBA8 surface allocation is invalid"));
        }
        Ok(NeoAmdSharedOutput {
            width: desc.width as i32,
            height: desc.height as i32,
            resource_key: desc.resource_key,
            byte_len: usize::try_from(desc.byte_len)
                .map_err(|_| format!("NeoAMD {label} RGBA8 byte length overflow"))?,
            allocation_byte_len: desc.allocation_byte_len,
        })
    }

    pub fn supports_interp_stream_shared_rgba8(&self) -> bool {
        self.info.kind == NEOAMD_SESSION_INTERPOLATION
            && self.info.capabilities & NEOAMD_CAP_INTERP_SHARED_RGBA8 != 0
            && self.backend.prepare_interp_stream_shared_rgba8.is_some()
            && self.backend.run_interp_stream_phase_shared_rgba8.is_some()
            && self.backend.get_shared_output_handle.is_some()
    }

    pub fn prepare_interp_stream_shared_rgba8(
        &mut self,
        width: i32,
        height: i32,
        inputs: &[&[u8]],
    ) -> Result<Option<NeoAmdSharedOutput>, String> {
        if !self.supports_interp_stream_shared_rgba8() {
            return Ok(None);
        }
        let function = self
            .backend
            .prepare_interp_stream_shared_rgba8
            .ok_or_else(|| "NeoAMD shared interpolation prepare export is unavailable".to_string())?;
        if width <= 0 || height <= 0 || !(inputs.len() == 2 || inputs.len() == 4) {
            return Err("NeoAMD shared interpolation arguments are invalid".to_string());
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "NeoAMD shared interpolation input size overflow".to_string())?;
        let required = pixels
            .checked_mul(4)
            .ok_or_else(|| "NeoAMD shared interpolation input byte size overflow".to_string())?;
        if inputs.iter().any(|input| input.len() < required) {
            return Err("NeoAMD shared interpolation input frame is too small".to_string());
        }
        let desc = Self::run_desc(width, height, inputs.len(), 4, 4, 0.5)?;
        let pointers: Vec<*const u8> = inputs.iter().map(|input| input.as_ptr()).collect();
        let mut output = NeoAmdSharedOutputDesc {
            struct_size: std::mem::size_of::<NeoAmdSharedOutputDesc>() as u32,
            ..NeoAmdSharedOutputDesc::default()
        };
        let status = unsafe { function(self.context, &desc, pointers.as_ptr(), &mut output) };
        if status != 0 {
            return Err(self
                .backend
                .error_string(self.context, status, "prepare-interp-stream-shared-rgba8"));
        }
        Self::shared_rgba8_surface(output, "interpolation output").map(Some)
    }

    pub fn run_interp_stream_phase_shared_rgba8(
        &mut self,
        width: i32,
        height: i32,
        input_count: usize,
        phase: f32,
        resource_key: u64,
    ) -> Result<(), String> {
        let function = self
            .backend
            .run_interp_stream_phase_shared_rgba8
            .ok_or_else(|| "NeoAMD shared interpolation phase export is unavailable".to_string())?;
        if !(input_count == 2 || input_count == 4)
            || !phase.is_finite()
            || !(0.0..=1.0).contains(&phase)
        {
            return Err("NeoAMD shared interpolation phase is invalid".to_string());
        }
        let desc = Self::run_desc(width, height, input_count, 4, 4, phase)?;
        let status = unsafe { function(self.context, &desc, phase, resource_key) };
        if status != 0 {
            return Err(self
                .backend
                .error_string(self.context, status, "run-interp-stream-phase-shared-rgba8"));
        }
        Ok(())
    }

    pub fn supports_interp_stream_u8(&self) -> bool {
        self.backend.prepare_interp_stream_u8.is_some()
            && self.backend.run_interp_stream_phase_u8.is_some()
    }

    pub fn prepare_interp_stream_u8(
        &mut self,
        width: i32,
        height: i32,
        inputs: &[&[u8]],
        input_pixel_stride: usize,
        output_pixel_stride: usize,
    ) -> Result<bool, String> {
        let Some(function) = self.backend.prepare_interp_stream_u8 else {
            return Ok(false);
        };
        if width <= 0 || height <= 0 || inputs.is_empty() {
            return Err("NeoAMD interpolation stream arguments are invalid".to_string());
        }
        if input_pixel_stride < 3 || !(output_pixel_stride == 3 || output_pixel_stride == 4) {
            return Err("NeoAMD interpolation stream pixel stride is invalid".to_string());
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "NeoAMD interpolation stream input size overflow".to_string())?;
        let required = pixels
            .checked_mul(input_pixel_stride)
            .ok_or_else(|| "NeoAMD interpolation stream input byte size overflow".to_string())?;
        if inputs.iter().any(|input| input.len() < required) {
            return Err("NeoAMD interpolation stream input frame is too small".to_string());
        }
        let desc = Self::run_desc(
            width,
            height,
            inputs.len(),
            input_pixel_stride,
            output_pixel_stride,
            0.5,
        )?;
        let pointers: Vec<*const u8> = inputs.iter().map(|input| input.as_ptr()).collect();
        let status = unsafe { function(self.context, &desc, pointers.as_ptr()) };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "prepare-interp-stream"));
        }
        Ok(true)
    }

    pub fn run_interp_stream_phase_u8(
        &mut self,
        width: i32,
        height: i32,
        input_count: usize,
        input_pixel_stride: usize,
        output_pixel_stride: usize,
        phase: f32,
    ) -> Result<(i32, i32, Vec<u8>), String> {
        let function = self
            .backend
            .run_interp_stream_phase_u8
            .ok_or_else(|| "NeoAMD interpolation stream phase export is unavailable".to_string())?;
        if !phase.is_finite() || !(0.0..=1.0).contains(&phase) {
            return Err("NeoAMD interpolation stream timestep is invalid".to_string());
        }
        let desc = Self::run_desc(
            width,
            height,
            input_count,
            input_pixel_stride,
            output_pixel_stride,
            phase,
        )?;
        let mut output = NeoAmdOutputDesc {
            struct_size: std::mem::size_of::<NeoAmdOutputDesc>() as u32,
            ..NeoAmdOutputDesc::default()
        };
        let status = unsafe { (self.backend.query_output)(self.context, &desc, &mut output) };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "query-interp-stream-output"));
        }
        let bytes_len = (output.width as usize)
            .checked_mul(output.height as usize)
            .and_then(|n| n.checked_mul(output_pixel_stride))
            .ok_or_else(|| "NeoAMD interpolation stream output size overflow".to_string())?;
        let mut bytes = vec![0u8; bytes_len];
        let status = unsafe {
            function(
                self.context,
                &desc,
                phase,
                bytes.as_mut_ptr(),
                bytes.len() as u64,
            )
        };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "run-interp-stream-phase"));
        }
        Ok((output.width as i32, output.height as i32, bytes))
    }

    pub fn run_interp_many_u8(
        &mut self,
        width: i32,
        height: i32,
        inputs: &[&[u8]],
        input_pixel_stride: usize,
        output_pixel_stride: usize,
        phases: &[f32],
    ) -> Result<Option<Vec<(i32, i32, Vec<u8>)>>, String> {
        let Some(function) = self.backend.run_interp_many_u8 else {
            return Ok(None);
        };
        if width <= 0 || height <= 0 || inputs.is_empty() || phases.is_empty() || phases.len() > 4 {
            return Err("NeoAMD multi-phase interpolation arguments are invalid".to_string());
        }
        if input_pixel_stride < 3 || !(output_pixel_stride == 3 || output_pixel_stride == 4) {
            return Err("NeoAMD multi-phase pixel stride is invalid".to_string());
        }
        if phases.iter().any(|phase| !phase.is_finite() || !(0.0..=1.0).contains(phase)) {
            return Err("NeoAMD multi-phase timestep is invalid".to_string());
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "NeoAMD multi-phase input size overflow".to_string())?;
        let required_input = pixels
            .checked_mul(input_pixel_stride)
            .ok_or_else(|| "NeoAMD multi-phase input byte size overflow".to_string())?;
        if inputs.iter().any(|input| input.len() < required_input) {
            return Err("NeoAMD multi-phase input frame is too small".to_string());
        }
        let desc = Self::run_desc(
            width,
            height,
            inputs.len(),
            input_pixel_stride,
            output_pixel_stride,
            phases[0],
        )?;
        let mut output = NeoAmdOutputDesc {
            struct_size: std::mem::size_of::<NeoAmdOutputDesc>() as u32,
            ..NeoAmdOutputDesc::default()
        };
        let status = unsafe { (self.backend.query_output)(self.context, &desc, &mut output) };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "query-multi-output"));
        }
        let logical = (output.width as usize)
            .checked_mul(output.height as usize)
            .and_then(|value| value.checked_mul(output_pixel_stride))
            .ok_or_else(|| "NeoAMD multi-phase output size overflow".to_string())?;
        let stride = usize::try_from(output.required_bytes)
            .unwrap_or(logical)
            .max(logical);
        let capacity = stride
            .checked_mul(phases.len())
            .ok_or_else(|| "NeoAMD multi-phase output capacity overflow".to_string())?;
        let mut bytes = vec![0u8; capacity];
        let pointers: Vec<*const u8> = inputs.iter().map(|input| input.as_ptr()).collect();
        let status = unsafe {
            function(
                self.context,
                &desc,
                pointers.as_ptr(),
                phases.as_ptr(),
                phases.len() as u32,
                bytes.as_mut_ptr(),
                stride as u64,
                bytes.len() as u64,
            )
        };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "run-multi-interp"));
        }
        let mut results = Vec::with_capacity(phases.len());
        for index in 0..phases.len() {
            let start = index * stride;
            results.push((
                output.width as i32,
                output.height as i32,
                bytes[start..start + logical].to_vec(),
            ));
        }
        Ok(Some(results))
    }

    pub fn run_u8(
        &mut self,
        width: i32,
        height: i32,
        inputs: &[&[u8]],
        input_pixel_stride: usize,
        output_pixel_stride: usize,
        phase: f32,
    ) -> Result<(i32, i32, Vec<u8>), String> {
        if width <= 0 || height <= 0 {
            return Err("NeoAMD input size is empty".to_string());
        }
        if inputs.is_empty() {
            return Err("NeoAMD received no input frames".to_string());
        }
        if input_pixel_stride < 3 || !(output_pixel_stride == 3 || output_pixel_stride == 4) {
            return Err("NeoAMD pixel stride is invalid".to_string());
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "NeoAMD input size overflow".to_string())?;
        let required_input = pixels
            .checked_mul(input_pixel_stride)
            .ok_or_else(|| "NeoAMD input byte size overflow".to_string())?;
        if inputs.iter().any(|input| input.len() < required_input) {
            return Err("NeoAMD input frame is too small".to_string());
        }
        let desc = NeoAmdRunDesc {
            struct_size: std::mem::size_of::<NeoAmdRunDesc>() as u32,
            width: width as u32,
            height: height as u32,
            input_count: inputs.len() as u32,
            input_pixel_stride: input_pixel_stride as u32,
            output_pixel_stride: output_pixel_stride as u32,
            phase: phase.clamp(0.0, 1.0),
            flags: 0,
            reserved: [0; 4],
        };
        let mut output = NeoAmdOutputDesc {
            struct_size: std::mem::size_of::<NeoAmdOutputDesc>() as u32,
            ..NeoAmdOutputDesc::default()
        };
        let status = unsafe { (self.backend.query_output)(self.context, &desc, &mut output) };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "query-output"));
        }
        if output.width == 0 || output.height == 0 {
            return Err("NeoAMD bridge returned an empty output size".to_string());
        }
        let minimum = (output.width as u64)
            .checked_mul(output.height as u64)
            .and_then(|value| value.checked_mul(output_pixel_stride as u64))
            .ok_or_else(|| "NeoAMD output byte size overflow".to_string())?;
        let required = output.required_bytes.max(minimum);
        let capacity = usize::try_from(required)
            .map_err(|_| "NeoAMD output is too large for this process".to_string())?;
        let mut bytes = vec![0u8; capacity];
        let pointers: Vec<*const u8> = inputs.iter().map(|input| input.as_ptr()).collect();
        let status = unsafe {
            (self.backend.run_u8)(
                self.context,
                &desc,
                pointers.as_ptr(),
                bytes.as_mut_ptr(),
                bytes.len() as u64,
            )
        };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "run"));
        }
        bytes.truncate(minimum as usize);
        Ok((output.width as i32, output.height as i32, bytes))
    }
}

impl Drop for NeoAmdSession {
    fn drop(&mut self) {
        if !self.context.is_null() {
            unsafe { (self.backend.destroy)(self.context) };
            self.context = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_defaults_to_canonical_bridge_name() {
        let manifest: NeoAmdManifest = serde_json::from_str(&format!(
            r#"{{"backend_api":{NEOAMD_BACKEND_API},"name":"test","architecture":"x86_64","bridge_abi":{NEOAMD_BRIDGE_ABI},"supported_gpu_architectures":["gfx1200","gfx1201"]}}"#
        ))
        .unwrap();
        assert_eq!(manifest.bridge, NEOAMD_DEFAULT_BRIDGE);
    }

    #[test]
    fn backend_paths_cannot_escape_private_directory() {
        assert!(safe_manifest_relative_path("neo_amd_backend.dll").is_ok());
        assert!(safe_manifest_relative_path("kernels/gfx1200.bin").is_ok());
        assert!(safe_manifest_relative_path("../neo_amd_backend.dll").is_err());
        assert!(safe_manifest_relative_path("C:/Windows/System32/a.dll").is_err());
    }

    #[test]
    fn session_create_contract_is_resolution_independent() {
        // Width/height belong to NeoAmdRunDesc, never NeoAmdCreateDesc. Keep this
        // guard because per-resolution engine/session creation is explicitly
        // outside the NeoAMD ABI contract.
        assert!(std::mem::size_of::<NeoAmdCreateDesc>() < std::mem::size_of::<NeoAmdRunDesc>() * 2);
        let source = include_str!("neoamd_backend.rs");
        let create = source
            .split("struct NeoAmdCreateDesc")
            .nth(1)
            .unwrap()
            .split("struct NeoAmdRunDesc")
            .next()
            .unwrap();
        assert!(!create.contains("width:"));
        assert!(!create.contains("height:"));
    }
}
