//! Optional DLSS Neural Rendering Backend Pack boundary.
//!
//! Stability contract:
//! - A normal Neo installation only performs a cheap manifest/path probe.
//! - No backend DLL is loaded and no large runtime file is hashed until an
//!   explicit preflight/load request is made.
//! - The bridge ABI is deliberately CPU RGBA8 first.  This is not the final
//!   performance path; it is a narrow compatibility boundary that lets the
//!   NVIDIA/D3D12 implementation evolve outside the main executable.
//! - Callers must treat every bridge error as fail-open and continue with the
//!   original frame. The render stage uses this module only in an isolated worker.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::{CString, c_char, c_void};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use windows::Win32::{
    Foundation::{FreeLibrary, HMODULE},
    System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        LoadLibraryExW,
    },
};
use windows::core::{PCSTR, PCWSTR};

pub const DLSSNR_BACKEND_API: u32 = 1;
pub const DLSSNR_BRIDGE_ABI: u32 = 1;
pub const DLSSNR_DEFAULT_BRIDGE: &str = "neo_dlssnr_backend.dll";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DlssNrManifest {
    pub backend_api: u32,
    pub name: String,
    pub architecture: String,
    pub bridge_abi: u32,
    #[serde(default = "default_bridge_name")]
    pub bridge: String,
    /// Informational only.  The bridge owns runtime discovery inside its
    /// private directory and Neo never loads this file directly.
    #[serde(default)]
    pub runtime: String,
    #[serde(default)]
    pub required_files: Vec<String>,
    /// Hashes are verified only when the pack is explicitly preflighted.
    /// This avoids hashing a very large DLSSNR runtime during normal startup.
    #[serde(default)]
    pub file_sha256: BTreeMap<String, String>,
    /// Optional ordered runtime locations understood by the bridge. Neo only
    /// checks that at least one candidate exists; it never loads a runtime
    /// itself. This allows official/community/mod runtimes to evolve without
    /// changing the host ABI.
    #[serde(default)]
    pub runtime_candidates: Vec<String>,
    /// Explicit opt-in for user-replaceable runtime binaries. The bridge DLL
    /// itself can never be made replaceable through this switch.
    #[serde(default)]
    pub allow_user_runtime_replacement: bool,
    /// Runtime files for which presence is required but a fixed SHA-256 is not.
    /// Entries are accepted only below runtime/ and only when the opt-in above
    /// is true. This is intended for user-supplied/modded DLSSNR runtimes.
    #[serde(default)]
    pub replaceable_runtime_files: Vec<String>,
    /// Informational compatibility labels for diagnostics/UI. The host does
    /// not gate behavior on these strings.
    #[serde(default)]
    pub compatibility_tags: Vec<String>,
}

fn default_bridge_name() -> String {
    DLSSNR_DEFAULT_BRIDGE.to_string()
}

#[derive(Clone, Debug, Default)]
pub struct DlssNrAvailability {
    pub installed: bool,
    pub reason: Option<String>,
    pub manifest: Option<DlssNrManifest>,
    pub backend_dir: Option<PathBuf>,
}

fn unavailable(
    reason: impl Into<String>,
    manifest: Option<DlssNrManifest>,
    backend_dir: Option<PathBuf>,
) -> DlssNrAvailability {
    DlssNrAvailability {
        installed: false,
        reason: Some(reason.into()),
        manifest,
        backend_dir,
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
        return Err(format!("unsafe backend file path: {file}"));
    }
    Ok(path)
}

fn is_runtime_relative_path(path: &Path) -> bool {
    matches!(
        path.components().next(),
        Some(Component::Normal(first)) if first.to_string_lossy().eq_ignore_ascii_case("runtime")
    )
}

fn confined_file(root: &Path, relative: &Path) -> Result<PathBuf, String> {
    let canonical_root = root.canonicalize().map_err(|e| e.to_string())?;
    let target = root
        .join(relative)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if !target.starts_with(&canonical_root) {
        return Err("DLSSNR file escapes its private directory through a link".into());
    }
    Ok(target)
}

fn validate_replaceable_runtime_files(manifest: &DlssNrManifest) -> Result<(), String> {
    if !manifest.allow_user_runtime_replacement && !manifest.replaceable_runtime_files.is_empty() {
        return Err(
            "replaceable_runtime_files requires allow_user_runtime_replacement=true".to_string(),
        );
    }
    for file in &manifest.replaceable_runtime_files {
        let relative = safe_manifest_relative_path(file)?;
        if !is_runtime_relative_path(&relative) {
            return Err(format!(
                "user-replaceable DLSSNR files must stay below runtime/: {file}"
            ));
        }
        if file.eq_ignore_ascii_case(&manifest.bridge) {
            return Err("the DLSSNR bridge DLL can never be user-replaceable".to_string());
        }
    }
    Ok(())
}

fn is_replaceable_runtime_file(manifest: &DlssNrManifest, file: &str) -> bool {
    manifest.allow_user_runtime_replacement
        && manifest
            .replaceable_runtime_files
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(file))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path).map_err(|error| {
        format!(
            "{} could not be opened for hashing: {error}",
            path.display()
        )
    })?;
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

/// Cheap startup discovery only.  It intentionally does not hash or load any
/// DLL.  Missing packs are a normal condition and are not logged as errors.
pub fn detect_dlssnr_backend_pack(app_dir: &Path) -> DlssNrAvailability {
    let backend_dir = app_dir.join("backends").join("dlssnr");
    let manifest_path = backend_dir.join("backend.json");
    if !manifest_path.is_file() {
        return unavailable(
            "Neo DLSSNR Backend Pack is not installed",
            None,
            Some(backend_dir),
        );
    }

    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            return unavailable(
                format!("backend.json could not be read: {error}"),
                None,
                Some(backend_dir),
            );
        }
    };
    // Windows PowerShell 5.1 writes UTF-8 with a BOM for `-Encoding UTF8`.
    // Accept that form defensively so a valid locally-generated manifest cannot
    // disable the optional backend solely because of its text encoding.
    let manifest_json = manifest_bytes
        .strip_prefix(&[0xEF, 0xBB, 0xBF])
        .unwrap_or(&manifest_bytes);
    let manifest: DlssNrManifest = match serde_json::from_slice(manifest_json) {
        Ok(manifest) => manifest,
        Err(error) => {
            return unavailable(
                format!("backend.json is invalid: {error}"),
                None,
                Some(backend_dir),
            );
        }
    };

    if manifest.backend_api != DLSSNR_BACKEND_API {
        return unavailable(
            format!(
                "backend API {} is incompatible (expected {})",
                manifest.backend_api, DLSSNR_BACKEND_API
            ),
            Some(manifest),
            Some(backend_dir),
        );
    }
    if manifest.bridge_abi != DLSSNR_BRIDGE_ABI {
        return unavailable(
            format!(
                "bridge ABI {} is incompatible (expected {})",
                manifest.bridge_abi, DLSSNR_BRIDGE_ABI
            ),
            Some(manifest),
            Some(backend_dir),
        );
    }
    if !manifest.architecture.eq_ignore_ascii_case("x86_64") {
        return unavailable(
            format!(
                "unsupported backend architecture: {}",
                manifest.architecture
            ),
            Some(manifest),
            Some(backend_dir),
        );
    }
    if let Err(error) = validate_replaceable_runtime_files(&manifest) {
        return unavailable(error, Some(manifest), Some(backend_dir));
    }
    let bridge_relative = match safe_manifest_relative_path(&manifest.bridge) {
        Ok(path) => path,
        Err(error) => {
            return unavailable(error, Some(manifest), Some(backend_dir));
        }
    };
    if !backend_dir.join(&bridge_relative).is_file() {
        return unavailable(
            format!("DLSSNR bridge is missing: {}", manifest.bridge),
            Some(manifest),
            Some(backend_dir),
        );
    }
    for file in &manifest.required_files {
        let relative = match safe_manifest_relative_path(file) {
            Ok(path) => path,
            Err(error) => {
                return unavailable(error, Some(manifest), Some(backend_dir));
            }
        };
        if !backend_dir.join(relative).is_file() {
            return unavailable(
                format!("required DLSSNR pack file is missing: {file}"),
                Some(manifest),
                Some(backend_dir),
            );
        }
    }

    if !manifest.runtime_candidates.is_empty() {
        let mut any_runtime = false;
        for file in &manifest.runtime_candidates {
            let relative = match safe_manifest_relative_path(file) {
                Ok(path) => path,
                Err(error) => {
                    return unavailable(error, Some(manifest), Some(backend_dir));
                }
            };
            if !is_runtime_relative_path(&relative) {
                return unavailable(
                    format!("DLSSNR runtime candidates must stay below runtime/: {file}"),
                    Some(manifest),
                    Some(backend_dir),
                );
            }
            any_runtime |= backend_dir.join(relative).is_file();
        }
        if !any_runtime {
            return unavailable(
                "none of the declared DLSSNR runtime candidates are installed",
                Some(manifest),
                Some(backend_dir),
            );
        }
    }

    DlssNrAvailability {
        installed: true,
        reason: None,
        manifest: Some(manifest),
        backend_dir: Some(backend_dir),
    }
}

/// Expensive integrity validation.  This is deliberately separate from
/// startup discovery and is intended to run only when DLSSNR is explicitly
/// enabled by the render stage or explicitly preflighted by the probe.
pub fn verify_dlssnr_backend_pack(availability: &DlssNrAvailability) -> Result<(), String> {
    if !availability.installed {
        return Err(availability
            .reason
            .clone()
            .unwrap_or_else(|| "DLSSNR Backend Pack is unavailable".to_string()));
    }
    let manifest = availability
        .manifest
        .as_ref()
        .ok_or_else(|| "DLSSNR manifest is unavailable".to_string())?;
    let backend_dir = availability
        .backend_dir
        .as_ref()
        .ok_or_else(|| "DLSSNR backend directory is unavailable".to_string())?;

    validate_replaceable_runtime_files(manifest)?;

    let mut files = manifest.required_files.clone();
    if !files
        .iter()
        .any(|file| file.eq_ignore_ascii_case(&manifest.bridge))
    {
        files.push(manifest.bridge.clone());
    }
    for file in files {
        let relative = safe_manifest_relative_path(&file)?;
        confined_file(backend_dir, &relative)?;
        if is_replaceable_runtime_file(manifest, &file) {
            if !backend_dir.join(relative).is_file() {
                return Err(format!(
                    "user-replaceable DLSSNR runtime is missing: {file}"
                ));
            }
            continue;
        }

        let expected = manifest
            .file_sha256
            .get(&file)
            .ok_or_else(|| format!("backend.json is missing the SHA-256 for {file}"))?;
        if expected.trim().is_empty() {
            return Err(format!("backend.json has an empty SHA-256 for {file}"));
        }
        let actual = sha256_file(&backend_dir.join(relative))?;
        if !actual.eq_ignore_ascii_case(expected.trim()) {
            return Err(format!("{file} does not match the Backend Pack manifest"));
        }
    }

    // Runtime candidates are also verified when they are not explicitly
    // user-replaceable. This prevents an accidentally loose manifest from
    // turning a strict pack into an unhashed runtime path.
    for file in &manifest.runtime_candidates {
        let relative = safe_manifest_relative_path(file)?;
        if backend_dir.join(&relative).is_file() {
            confined_file(
                &backend_dir.join("runtime"),
                relative
                    .strip_prefix("runtime")
                    .map_err(|_| "runtime root must be runtime/")?,
            )?;
        }
        if !backend_dir.join(&relative).is_file() || is_replaceable_runtime_file(manifest, file) {
            continue;
        }
        let expected = manifest.file_sha256.get(file).ok_or_else(|| {
            format!("backend.json is missing the SHA-256 for runtime candidate {file}")
        })?;
        if expected.trim().is_empty() {
            return Err(format!(
                "backend.json has an empty SHA-256 for runtime candidate {file}"
            ));
        }
        let actual = sha256_file(&backend_dir.join(relative))?;
        if !actual.eq_ignore_ascii_case(expected.trim()) {
            return Err(format!("{file} does not match the Backend Pack manifest"));
        }
    }

    // The bridge is the trusted executable boundary and is always pinned,
    // even when runtime replacement is explicitly allowed.
    let bridge_hash = manifest
        .file_sha256
        .get(&manifest.bridge)
        .ok_or_else(|| "backend.json must pin the DLSSNR bridge SHA-256".to_string())?;
    if bridge_hash.trim().is_empty() {
        return Err("backend.json has an empty SHA-256 for the DLSSNR bridge".to_string());
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NeoDlssNrCreateDesc {
    pub struct_size: u32,
    pub api_version: u32,
    /// DXGI adapter LUID. Zero means the bridge may choose its default NVIDIA
    /// adapter. Neo's render worker always passes the exact selected GPU.
    pub adapter_luid: u64,
    pub width: u32,
    pub height: u32,
    /// Feature 18 render preset. 0 = runtime default, 1..=3 = explicit presets.
    pub preset: u32,
    pub reserved: [u32; 7],
}

impl NeoDlssNrCreateDesc {
    const OPTIONS_V2_MAGIC: u32 = 0x3256_524E; // ASCII "NRV2" little-endian.

    pub fn new(adapter_luid: Option<u64>, width: u32, height: u32, preset: u32) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            api_version: DLSSNR_BRIDGE_ABI,
            adapter_luid: adapter_luid.unwrap_or(0),
            width,
            height,
            preset: preset.min(3),
            reserved: [0; 7],
        }
    }

    pub fn with_options(
        adapter_luid: Option<u64>,
        width: u32,
        height: u32,
        options: crate::core::dlssnr::DlssNrOptions,
    ) -> Self {
        let options = crate::core::dlssnr::sanitize_options(options);
        let mut desc = Self::new(adapter_luid, width, height, options.preset);
        // Keep ABI v1 and consume only formerly-reserved words. Bridges that
        // predate v710 ignore these values; v710+ bridges key off NRV2 magic.
        desc.reserved[0] = options.style;
        desc.reserved[1] = (if options.auto_mask { 1 } else { 0 }) | ((if options.ui_correction { 1 } else { 0 }) << 1);
        desc.reserved[2] = options.intensity.to_bits();
        desc.reserved[3] = options.local_tone.to_bits();
        desc.reserved[4] = options.local_structure.to_bits();
        desc.reserved[5] = options.skin_structure.to_bits();
        desc.reserved[6] = Self::OPTIONS_V2_MAGIC;
        desc
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NeoDlssNrFrameDesc {
    pub struct_size: u32,
    pub width: u32,
    pub height: u32,
    /// Nonzero asks the temporal backend to discard prior history.
    pub reset_history: u32,
    pub frame_index: u64,
    pub reserved: [u32; 6],
}

impl NeoDlssNrFrameDesc {
    pub fn new(width: u32, height: u32, reset_history: bool, frame_index: u64) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            width,
            height,
            reset_history: u32::from(reset_history),
            frame_index,
            reserved: [0; 6],
        }
    }
}

type GetApiVersionFn = unsafe extern "C" fn() -> u32;
type CreateFn = unsafe extern "C" fn(*const NeoDlssNrCreateDesc, *mut *mut c_void) -> i32;
type ProcessRgba8Fn = unsafe extern "C" fn(
    *mut c_void,
    *const NeoDlssNrFrameDesc,
    *const u8,
    u32,
    *mut u8,
    u32,
) -> i32;
type ResetHistoryFn = unsafe extern "C" fn(*mut c_void) -> i32;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type LastErrorFn = unsafe extern "C" fn(*mut c_void, *mut c_char, u32) -> u32;
type GetCapabilitiesFn = unsafe extern "C" fn() -> u64;
type DescribeBackendFn = unsafe extern "C" fn(*mut c_char, u32) -> u32;
type SetOptionsFn = unsafe extern "C" fn(*mut c_void, *const NeoDlssNrEvalOptions) -> i32;

#[repr(C)]
pub struct NeoDlssNrEvalOptions {
    pub struct_size: u32,
    pub intensity: f32,
    pub local_tone: f32,
    pub local_structure: f32,
    pub skin_structure: f32,
    // ABI-v1 optional tail. Old bridges receive struct_size=20 and ignore it.
    pub style: u32,
    pub use_auto_mask: u32,
    pub ui_correction: u32,
    pub reserved: u32,
}

pub const DLSSNR_EVAL_OPTIONS_V1_SIZE: u32 = 20;
pub const DLSSNR_CAP_ZERO_GUIDANCE: u64 = 1 << 0;
pub const DLSSNR_CAP_OPTICAL_FLOW_MOTION: u64 = 1 << 1;
pub const DLSSNR_CAP_DEPTH_GUIDANCE: u64 = 1 << 2;
pub const DLSSNR_CAP_D3D12_SHARED_TEXTURE: u64 = 1 << 3;
pub const DLSSNR_CAP_USER_RUNTIME_SELECTION: u64 = 1 << 4;
pub const DLSSNR_CAP_EVAL_OPTIONS: u64 = 1 << 5;
pub const DLSSNR_CAP_ADVANCED_OPTIONS: u64 = 1 << 6;

unsafe fn symbol<T: Copy>(module: HMODULE, name: &str) -> Result<T, String> {
    let export_name = name.to_string();
    let name = CString::new(name).map_err(|_| format!("invalid export name: {export_name}"))?;
    let raw = unsafe { GetProcAddress(module, PCSTR(name.as_ptr().cast())) }
        .ok_or_else(|| format!("required bridge export is missing: {export_name}"))?;
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

/// Loaded bridge DLL.  Creation/evaluation remains entirely inside the pack.
/// Keeping this type private to the optional path prevents NVIDIA headers or
/// SDK libraries from becoming Neo dependencies.
pub struct DlssNrBackend {
    module: HMODULE,
    create: CreateFn,
    process_rgba8: ProcessRgba8Fn,
    reset_history: ResetHistoryFn,
    destroy: DestroyFn,
    last_error: LastErrorFn,
    get_capabilities: Option<GetCapabilitiesFn>,
    describe_backend: Option<DescribeBackendFn>,
    set_options: Option<SetOptionsFn>,
}

impl DlssNrBackend {
    pub fn load_verified(availability: &DlssNrAvailability) -> Result<Self, String> {
        verify_dlssnr_backend_pack(availability)?;
        let manifest = availability
            .manifest
            .as_ref()
            .ok_or_else(|| "DLSSNR manifest is unavailable".to_string())?;
        let backend_dir = availability
            .backend_dir
            .as_ref()
            .ok_or_else(|| "DLSSNR backend directory is unavailable".to_string())?;
        let bridge_path = backend_dir.join(safe_manifest_relative_path(&manifest.bridge)?);
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
            let get_api_version: GetApiVersionFn = symbol(module, "neo_dlssnr_get_api_version")?;
            let actual_abi = get_api_version();
            if actual_abi != DLSSNR_BRIDGE_ABI {
                return Err(format!(
                    "DLSSNR bridge ABI {actual_abi} is incompatible (expected {DLSSNR_BRIDGE_ABI})"
                ));
            }
            type SelectRuntime = unsafe extern "C" fn(*const u16) -> i32;
            if let Some(select) =
                optional_symbol::<SelectRuntime>(module, "neo_dlssnr_select_runtime")
            {
                let selected = manifest
                    .runtime_candidates
                    .iter()
                    .find(|name| backend_dir.join(name).is_file())
                    .ok_or("no installed runtime candidate")?;
                let relative = safe_manifest_relative_path(selected)?;
                let runtime = confined_file(backend_dir, &relative)?;
                log::info!("dlssnr-runtime-selected: {}", runtime.display());
                eprintln!("DLSSNR selected runtime: {}", runtime.display());
                if select(wide(&runtime).as_ptr()) != 0 {
                    return Err("bridge rejected runtime selection".into());
                }
            }
            Ok(Self {
                module,
                create: symbol(module, "neo_dlssnr_create")?,
                process_rgba8: symbol(module, "neo_dlssnr_process_rgba8")?,
                reset_history: symbol(module, "neo_dlssnr_reset_history")?,
                destroy: symbol(module, "neo_dlssnr_destroy")?,
                last_error: symbol(module, "neo_dlssnr_last_error")?,
                get_capabilities: optional_symbol(module, "neo_dlssnr_get_capabilities"),
                describe_backend: optional_symbol(module, "neo_dlssnr_describe_backend"),
                set_options: optional_symbol(module, "neo_dlssnr_set_options"),
            })
        })();

        if loaded.is_err() {
            unsafe {
                let _ = FreeLibrary(module);
            }
        }
        loaded
    }

    /// Optional ABI-v1 extension. Older bridges simply report no declared
    /// capabilities and remain fully compatible.
    pub fn capabilities(&self) -> u64 {
        self.get_capabilities
            .map(|function| unsafe { function() })
            .unwrap_or(0)
    }

    /// Optional UTF-8 diagnostic description supplied by the bridge, e.g. the
    /// active official/community/mod runtime build. Neo never parses this to
    /// select behavior; it is intentionally informational.
    pub fn backend_description(&self) -> Option<String> {
        let function = self.describe_backend?;
        let mut buffer = [0u8; 2048];
        let written =
            unsafe { function(buffer.as_mut_ptr().cast::<c_char>(), buffer.len() as u32) } as usize;
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
        &self,
        adapter_luid: Option<u64>,
        width: u32,
        height: u32,
        preset: u32,
    ) -> Result<DlssNrSession<'_>, String> {
        let options = crate::core::dlssnr::DlssNrOptions {
            preset: preset.min(3),
            ..crate::core::dlssnr::DlssNrOptions::default()
        };
        self.create_session_with_options(adapter_luid, width, height, options)
    }

    pub fn create_session_with_options(
        &self,
        adapter_luid: Option<u64>,
        width: u32,
        height: u32,
        options: crate::core::dlssnr::DlssNrOptions,
    ) -> Result<DlssNrSession<'_>, String> {
        if width == 0 || height == 0 {
            return Err("DLSSNR cannot create a zero-sized session".to_string());
        }
        let options = crate::core::dlssnr::sanitize_options(options);
        let desc = NeoDlssNrCreateDesc::with_options(adapter_luid, width, height, options);
        let mut context = std::ptr::null_mut();
        let status = unsafe { (self.create)(&desc, &mut context) };
        if status != 0 || context.is_null() {
            return Err(self.error_string(context, status, "create"));
        }
        Ok(DlssNrSession {
            backend: self,
            context,
            width,
            height,
            next_frame_index: 0,
        })
    }

    fn error_string(&self, context: *mut c_void, status: i32, operation: &str) -> String {
        let mut buffer = [0u8; 1024];
        // ABI-v1 permits a bridge to expose its most recent global create/load
        // error when context is null. This keeps Feature 18 bring-up failures
        // diagnosable without weakening the fail-open host contract.
        let written = (unsafe {
            (self.last_error)(
                context,
                buffer.as_mut_ptr().cast::<c_char>(),
                buffer.len() as u32,
            )
        }) as usize;
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
            format!("DLSSNR {operation} failed with bridge status {status}")
        } else {
            format!("DLSSNR {operation} failed with bridge status {status}: {message}")
        }
    }
}

impl Drop for DlssNrBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = FreeLibrary(self.module);
        }
    }
}

pub struct DlssNrSession<'a> {
    backend: &'a DlssNrBackend,
    context: *mut c_void,
    width: u32,
    height: u32,
    next_frame_index: u64,
}

impl DlssNrSession<'_> {
    pub fn set_options(
        &mut self,
        values: crate::core::dlssnr::DlssNrOptions,
    ) -> Result<(), String> {
        let values = crate::core::dlssnr::sanitize_options(values);
        let Some(set) = self.backend.set_options else {
            return if values == crate::core::dlssnr::DlssNrOptions::default() {
                Ok(())
            } else {
                Err("DLSSNR backend pack does not support evaluation options".into())
            };
        };
        let advanced = self.backend.capabilities() & DLSSNR_CAP_ADVANCED_OPTIONS != 0;
        let legacy_strength = |v: f32| v.clamp(0.0, 1.0);
        let options = NeoDlssNrEvalOptions {
            struct_size: if advanced {
                std::mem::size_of::<NeoDlssNrEvalOptions>() as u32
            } else {
                DLSSNR_EVAL_OPTIONS_V1_SIZE
            },
            intensity: if advanced { values.intensity } else { legacy_strength(values.intensity) },
            local_tone: if advanced { values.local_tone } else { legacy_strength(values.local_tone) },
            local_structure: if advanced { values.local_structure } else { legacy_strength(values.local_structure) },
            skin_structure: if advanced { values.skin_structure } else { legacy_strength(values.skin_structure) },
            style: values.style,
            use_auto_mask: if values.auto_mask { 1 } else { 0 },
            ui_correction: if values.ui_correction { 1 } else { 0 },
            reserved: 0,
        };
        let status = unsafe { set(self.context, &options) };
        if status == 0 {
            Ok(())
        } else {
            Err(self
                .backend
                .error_string(self.context, status, "set-options"))
        }
    }

    pub fn reset_history(&mut self) -> Result<(), String> {
        let status = unsafe { (self.backend.reset_history)(self.context) };
        if status == 0 {
            Ok(())
        } else {
            Err(self
                .backend
                .error_string(self.context, status, "reset-history"))
        }
    }

    /// Evaluate one tightly-packed or explicitly-strided RGBA8 frame.
    /// On error, no output is returned; the render caller must present
    /// the original input unchanged (fail-open policy).
    pub fn process_rgba8(
        &mut self,
        input: &[u8],
        input_stride: u32,
        reset_history: bool,
    ) -> Result<Vec<u8>, String> {
        let min_stride = self
            .width
            .checked_mul(4)
            .ok_or_else(|| "DLSSNR row size overflow".to_string())?;
        if input_stride < min_stride {
            return Err(format!(
                "DLSSNR input stride {input_stride} is smaller than {min_stride}"
            ));
        }
        let input_bytes = (input_stride as usize)
            .checked_mul(self.height as usize)
            .ok_or_else(|| "DLSSNR input size overflow".to_string())?;
        if input.len() < input_bytes {
            return Err(format!(
                "DLSSNR input buffer is too small: {} < {}",
                input.len(),
                input_bytes
            ));
        }

        let output_stride = min_stride;
        let output_bytes = (output_stride as usize)
            .checked_mul(self.height as usize)
            .ok_or_else(|| "DLSSNR output size overflow".to_string())?;
        let mut output = vec![0u8; output_bytes];
        let desc = NeoDlssNrFrameDesc::new(
            self.width,
            self.height,
            reset_history,
            self.next_frame_index,
        );
        let status = unsafe {
            (self.backend.process_rgba8)(
                self.context,
                &desc,
                input.as_ptr(),
                input_stride,
                output.as_mut_ptr(),
                output_stride,
            )
        };
        if status != 0 {
            return Err(self.backend.error_string(self.context, status, "evaluate"));
        }
        self.next_frame_index = self.next_frame_index.wrapping_add(1);
        Ok(output)
    }
}

impl Drop for DlssNrSession<'_> {
    fn drop(&mut self) {
        if !self.context.is_null() {
            unsafe {
                (self.backend.destroy)(self.context);
            }
            self.context = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        DLSSNR_BACKEND_API, DLSSNR_BRIDGE_ABI, DlssNrManifest, safe_manifest_relative_path,
        validate_replaceable_runtime_files,
    };

    #[test]
    fn manifest_defaults_to_canonical_bridge_name() {
        let json = format!(
            r#"{{
                "backend_api":{DLSSNR_BACKEND_API},
                "name":"test",
                "architecture":"x86_64",
                "bridge_abi":{DLSSNR_BRIDGE_ABI}
            }}"#
        );
        let parsed: DlssNrManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.bridge, "neo_dlssnr_backend.dll");
    }

    #[test]
    fn manifest_paths_cannot_escape_backend_directory() {
        assert!(safe_manifest_relative_path("neo_dlssnr_backend.dll").is_ok());
        assert!(safe_manifest_relative_path("runtime/nvngx_dlssnr.dll").is_ok());
        assert!(safe_manifest_relative_path("../nvngx_dlssnr.dll").is_err());
        assert!(safe_manifest_relative_path("C:/Windows/System32/foo.dll").is_err());
        assert!(safe_manifest_relative_path("").is_err());
        assert!(safe_manifest_relative_path("runtime/a.dll:payload").is_err());
        assert!(safe_manifest_relative_path("runtime/../../a.dll").is_err());
    }
    #[test]
    fn missing_pack_is_unavailable_without_loading_a_dll() {
        let root = std::env::current_dir()
            .unwrap()
            .join("nonexistent-dlssnr-test-root");
        let availability = super::detect_dlssnr_backend_pack(&root);
        assert!(!availability.installed);
        assert!(super::verify_dlssnr_backend_pack(&availability).is_err());
    }
    #[test]
    fn dlssnr_bridge_hash_mismatch_is_rejected_before_load() {
        let root = std::env::current_dir()
            .unwrap()
            .join(format!(".dlssnr-test-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("neo_dlssnr_backend.dll");
        std::fs::write(&file, b"not a DLL").unwrap();
        let mut manifest: DlssNrManifest = serde_json::from_str(&format!(
            r#"{{"backend_api":{DLSSNR_BACKEND_API},"name":"test","architecture":"x86_64","bridge_abi":{DLSSNR_BRIDGE_ABI}}}"#
        )).unwrap();
        manifest
            .file_sha256
            .insert(manifest.bridge.clone(), "0".repeat(64));
        let availability = super::DlssNrAvailability {
            installed: true,
            reason: None,
            manifest: Some(manifest),
            backend_dir: Some(root.clone()),
        };
        let result = super::verify_dlssnr_backend_pack(&availability);
        std::fs::remove_file(file).unwrap();
        std::fs::remove_dir(root).unwrap();
        assert!(result.unwrap_err().contains("does not match"));
    }
    #[test]
    fn replaceable_runtime_is_explicit_and_confined() {
        let mut manifest = DlssNrManifest {
            backend_api: DLSSNR_BACKEND_API,
            name: "test".to_string(),
            architecture: "x86_64".to_string(),
            bridge_abi: DLSSNR_BRIDGE_ABI,
            bridge: "neo_dlssnr_backend.dll".to_string(),
            runtime: String::new(),
            required_files: Vec::new(),
            file_sha256: BTreeMap::new(),
            runtime_candidates: Vec::new(),
            allow_user_runtime_replacement: false,
            replaceable_runtime_files: vec!["runtime/nvngx_dlssnr.dll".to_string()],
            compatibility_tags: Vec::new(),
        };
        assert!(validate_replaceable_runtime_files(&manifest).is_err());
        manifest.allow_user_runtime_replacement = true;
        assert!(validate_replaceable_runtime_files(&manifest).is_ok());
        manifest.replaceable_runtime_files = vec!["neo_dlssnr_backend.dll".to_string()];
        assert!(validate_replaceable_runtime_files(&manifest).is_err());
        manifest.replaceable_runtime_files = vec!["../nvngx_dlssnr.dll".to_string()];
        assert!(validate_replaceable_runtime_files(&manifest).is_err());
    }
}
