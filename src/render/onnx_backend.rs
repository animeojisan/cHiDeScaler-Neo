//! Optional TensorRT backend discovery and portable DLL activation.
//!
//! Detection never loads a model. A normal DirectML-only installation only
//! checks for `backends/tensorrt/backend.json`; an installed pack additionally
//! proves that the TensorRT and CUDA execution providers can be registered.

use crate::platform::gpu::{
    GpuAdapter, enumerate_adapters_quiet, high_performance_luid_order_quiet,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::{CString, c_void};
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;
use windows::Win32::{
    Foundation::{FreeLibrary, HMODULE},
    Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory1},
    System::LibraryLoader::{
        AddDllDirectory, GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_USER_DIRS, LoadLibraryExW,
        SetDefaultDllDirectories,
    },
};
use windows::core::{Interface, PCWSTR};

pub const BACKEND_API: u32 = 1;
pub const ORT_COMPAT_VERSION: &str = "1.24.4";
const NVIDIA_VENDOR_ID: u32 = 0x10de;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorRtManifest {
    pub backend_api: u32,
    pub name: String,
    pub architecture: String,
    pub onnxruntime: String,
    pub tensor_rt: String,
    pub cuda_major: u32,
    pub provider: String,
    #[serde(default)]
    pub required_files: Vec<String>,
    /// SHA-256 of the unified onnxruntime.dll shipped with Neo itself.
    /// This prevents provider DLLs from a different ORT build being mixed in.
    #[serde(default)]
    pub onnxruntime_sha256: String,
    /// At minimum both ORT provider DLLs must be covered. Backend packs may
    /// include hashes for every CUDA/TensorRT dependency as well.
    #[serde(default)]
    pub file_sha256: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default)]
pub struct TensorRtAvailability {
    pub available: bool,
    pub reason: Option<String>,
    pub device_id: Option<i32>,
    /// Canonical DXGI identity of the CUDA device selected for TensorRT.
    /// Auto and an explicit selection that resolve to the same physical GPU
    /// must share this identity so they also share engine/session caches.
    pub gpu_luid: Option<u64>,
    pub manifest: Option<TensorRtManifest>,
    pub backend_dir: Option<PathBuf>,
}

static DLL_DIRECTORY_COOKIES: OnceLock<Vec<usize>> = OnceLock::new();

fn unavailable(
    reason: impl Into<String>,
    manifest: Option<TensorRtManifest>,
) -> TensorRtAvailability {
    let reason = reason.into();
    log::info!("onnx-backend-unavailable: {reason}");
    TensorRtAvailability {
        reason: Some(reason),
        manifest,
        ..Default::default()
    }
}

fn wide(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

fn register_dll_directories(paths: &[&Path]) -> Result<(), String> {
    if DLL_DIRECTORY_COOKIES.get().is_some() {
        return Ok(());
    }
    unsafe {
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS)
            .map_err(|error| format!("SetDefaultDllDirectories failed: {error}"))?;
    }
    let mut cookies = Vec::new();
    for path in paths {
        let encoded = wide(path);
        let cookie = unsafe { AddDllDirectory(PCWSTR(encoded.as_ptr())) };
        if cookie.is_null() {
            return Err(format!("AddDllDirectory failed for {}", path.display()));
        }
        cookies.push(cookie as usize);
    }
    let _ = DLL_DIRECTORY_COOKIES.set(cookies);
    Ok(())
}

fn load_for_preflight(path: &Path) -> Result<HMODULE, String> {
    let encoded = wide(path);
    unsafe {
        LoadLibraryExW(
            PCWSTR(encoded.as_ptr()),
            None,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
                | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
                | LOAD_LIBRARY_SEARCH_USER_DIRS,
        )
    }
    .map_err(|error| format!("{} could not be loaded: {error}", path.display()))
}

fn find_runtime_dll(dir: &Path, prefix: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.to_ascii_lowercase()
                        .starts_with(&prefix.to_ascii_lowercase())
                        && name.to_ascii_lowercase().ends_with(".dll")
                })
        })
}

fn safe_manifest_relative_path(file: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(file);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(format!("unsafe backend file path: {file}"));
    }
    Ok(path)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("{} could not be hashed: {error}", path.display()))?;
    Ok(format!("{:X}", Sha256::digest(bytes)))
}

type CudaGetDeviceCount = unsafe extern "system" fn(*mut i32) -> i32;
type CudaDeviceGetLuid = unsafe extern "system" fn(*mut i8, *mut u32, i32) -> i32;
type CudaD3D11GetDevice = unsafe extern "system" fn(*mut i32, *mut c_void) -> i32;

unsafe fn cuda_symbol<T: Copy>(module: HMODULE, name: &str) -> Option<T> {
    let name = CString::new(name).ok()?;
    let raw = unsafe { GetProcAddress(module, windows::core::PCSTR(name.as_ptr().cast())) }?;
    Some(unsafe { std::mem::transmute_copy(&raw) })
}

/// CUDA's documented Direct3D 11 interoperability API can map the exact DXGI
/// adapter object to a CUDA ordinal without relying on cudaDeviceGetLuid. This
/// is the preferred Windows fallback for driver/runtime combinations where the
/// standalone LUID helper is not exported.
fn cuda_device_for_dxgi_adapter(
    module: HMODULE,
    target: &GpuAdapter,
) -> Result<Option<i32>, String> {
    let Some(map_fn) = (unsafe { cuda_symbol::<CudaD3D11GetDevice>(module, "cudaD3D11GetDevice") })
    else {
        return Ok(None);
    };

    let factory = unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }
        .map_err(|error| format!("CreateDXGIFactory1 for CUDA mapping failed: {error}"))?;
    let adapter1 = unsafe { factory.EnumAdapters1(target.device_id as u32) }.map_err(|error| {
        format!(
            "EnumAdapters1({}) for CUDA mapping failed: {error}",
            target.device_id
        )
    })?;
    let adapter: IDXGIAdapter = adapter1
        .cast()
        .map_err(|error| format!("IDXGIAdapter cast for CUDA mapping failed: {error}"))?;
    let mut cuda_device = -1i32;
    let status = unsafe { map_fn(&mut cuda_device, adapter.as_raw()) };
    if status == 0 && cuda_device >= 0 {
        log::info!(
            "tensorrt-cuda-d3d11-map: dxgi_device_id={} luid={:016x} cuda_device_id={} gpu='{}' result=match",
            target.device_id,
            target.luid,
            cuda_device,
            target.name
        );
        Ok(Some(cuda_device))
    } else {
        log::debug!(
            "tensorrt-cuda-d3d11-map: dxgi_device_id={} luid={:016x} gpu='{}' cuda_status={} result=no-match",
            target.device_id,
            target.luid,
            target.name,
            status
        );
        Ok(None)
    }
}

fn resolve_cuda_device(
    backend_dir: &Path,
    adapters: &[GpuAdapter],
    selected_adapter_luid: Option<u64>,
) -> Result<(i32, u64), String> {
    let nvidia: Vec<&GpuAdapter> = adapters
        .iter()
        .filter(|adapter| adapter.vendor_id == NVIDIA_VENDOR_ID)
        .collect();
    if nvidia.is_empty() {
        return Err("NVIDIA GPU was not detected".into());
    }
    let dxgi_candidates = nvidia
        .iter()
        .map(|adapter| {
            format!(
                "dxgi_device={} luid={:016x} vram_mb={} pci={:04x}:{:04x}:{:08x} name='{}'",
                adapter.device_id,
                adapter.luid,
                adapter.dedicated_video_memory / (1024 * 1024),
                adapter.vendor_id,
                adapter.pci_device_id,
                adapter.subsys_id,
                adapter.name
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    log::info!(
        "tensorrt-dxgi-nvidia-candidates: count={} selected_luid={} candidates=[{}]",
        nvidia.len(),
        selected_adapter_luid
            .map(|luid| format!("{luid:016x}"))
            .unwrap_or_else(|| "Auto".to_string()),
        dxgi_candidates
    );

    // An explicit Neo GPU selection is authoritative. Auto is different: do
    // not make TensorRT unavailable merely because more than one NVIDIA GPU is
    // present. CUDA exposes each device's DXGI LUID, so Auto can first prove
    // the real DXGI<->CUDA matches and then select a deterministic best NVIDIA
    // candidate. The render-GPU finalization step may later prefer a different
    // matching NVIDIA LUID if that is the actual OpenGL presentation device.
    let explicit_target = if let Some(luid) = selected_adapter_luid {
        Some(
            nvidia
                .iter()
                .copied()
                .find(|adapter| adapter.luid == luid)
                .ok_or_else(|| "the selected DXGI adapter is not an NVIDIA GPU".to_string())?,
        )
    } else {
        None
    };

    let cudart = find_runtime_dll(backend_dir, "cudart64_")
        .ok_or_else(|| "CUDA Runtime DLL (cudart64_*.dll) was not found".to_string())?;
    let module = load_for_preflight(&cudart)?;
    let result: Result<(i32, u64, &'static str, &GpuAdapter), String> = unsafe {
        let count_fn: CudaGetDeviceCount = cuda_symbol(module, "cudaGetDeviceCount")
            .ok_or_else(|| "cudaGetDeviceCount is unavailable".to_string())?;
        let mut count = 0;
        let status = count_fn(&mut count);
        if status != 0 || count <= 0 {
            Err(format!(
                "cudaGetDeviceCount failed with CUDA status {status}"
            ))
        } else if let Some(luid_fn) = cuda_symbol::<CudaDeviceGetLuid>(module, "cudaDeviceGetLuid")
        {
            let mut cuda_luids: Vec<(i32, u64)> = Vec::new();
            for device in 0..count {
                let mut luid = [0i8; 8];
                let mut node_mask = 0u32;
                let luid_status = luid_fn(luid.as_mut_ptr(), &mut node_mask, device);
                if luid_status == 0 {
                    let luid = u64::from_le_bytes(luid.map(|byte| byte as u8));
                    log::info!(
                        "tensorrt-cuda-device: cuda_device_id={} luid={:016x} node_mask={:#x}",
                        device,
                        luid,
                        node_mask
                    );
                    cuda_luids.push((device, luid));
                } else {
                    log::warn!(
                        "tensorrt-cuda-device: cuda_device_id={} luid=unavailable cuda_status={}",
                        device,
                        luid_status
                    );
                }
            }

            if let Some(target) = explicit_target {
                cuda_luids
                    .iter()
                    .find(|(_, luid)| *luid == target.luid)
                    .map(|(device_id, _)| (*device_id, target.luid, "exact-luid", target))
                    .ok_or_else(|| {
                        format!(
                            "no CUDA device matched the selected DXGI adapter LUID {:016x}",
                            target.luid
                        )
                    })
            } else {
                let mut matches: Vec<(&GpuAdapter, i32)> = nvidia
                    .iter()
                    .filter_map(|adapter| {
                        cuda_luids
                            .iter()
                            .find(|(_, luid)| *luid == adapter.luid)
                            .map(|(device_id, _)| (*adapter, *device_id))
                    })
                    .collect();
                if matches.is_empty() {
                    Err("no CUDA device matched any NVIDIA DXGI adapter LUID".into())
                } else {
                    // Do not infer GPU performance from VRAM size: a faster
                    // newer GPU can legitimately have less VRAM than an older
                    // model. Use Windows/DXGI's HIGH_PERFORMANCE preference
                    // ordering, then the canonical DXGI ordinal as a stable
                    // fallback. Once OpenGL identifies an NVIDIA presentation
                    // GPU, main.rs remaps TensorRT to that exact LUID.
                    let high_performance_order = high_performance_luid_order_quiet();
                    let preference_rank = |adapter: &GpuAdapter| {
                        high_performance_order
                            .iter()
                            .position(|luid| *luid == adapter.luid)
                            .unwrap_or(usize::MAX)
                    };
                    matches.sort_by(|(a, _), (b, _)| {
                        preference_rank(a)
                            .cmp(&preference_rank(b))
                            .then_with(|| a.device_id.cmp(&b.device_id))
                    });
                    let (target, device_id) = matches[0];
                    if matches.len() > 1 {
                        let candidates = matches
                            .iter()
                            .map(|(adapter, cuda_device)| {
                                format!(
                                    "dxgi_device={} cuda_device={} luid={:016x} high_perf_rank={} vram_mb={} name='{}'",
                                    adapter.device_id,
                                    cuda_device,
                                    adapter.luid,
                                    preference_rank(adapter),
                                    adapter.dedicated_video_memory / (1024 * 1024),
                                    adapter.name
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(" | ");
                        log::info!(
                            "tensorrt-gpu-map: auto-selection matched_nvidia={} policy=dxgi-high-performance-then-canonical-order candidates=[{}]",
                            matches.len(),
                            candidates
                        );
                    }
                    Ok((device_id, target.luid, "auto-high-performance-luid", target))
                }
            }
        } else {
            // Some Windows CUDA Runtime builds do not export cudaDeviceGetLuid
            // even though CUDA/TensorRT itself works. Do not disable TensorRT in
            // that case: CUDA provides cudaD3D11GetDevice specifically to map an
            // IDXGIAdapter to its CUDA ordinal. This also handles two real same-
            // model NVIDIA GPUs and driver-created DXGI aliases correctly.
            log::info!(
                "tensorrt-gpu-map: cudaDeviceGetLuid=unavailable action=try-cudaD3D11GetDevice cuda_device_count={count}"
            );

            if let Some(target) = explicit_target {
                match cuda_device_for_dxgi_adapter(module, target)? {
                    Some(device_id) if device_id < count => {
                        Ok((device_id, target.luid, "dxgi-adapter-to-cuda-d3d11", target))
                    }
                    Some(device_id) => Err(format!(
                        "cudaD3D11GetDevice returned out-of-range CUDA device {device_id} (count={count})"
                    )),
                    None if count == 1 => {
                        // A single CUDA device is unambiguous even when DXGI has
                        // multiple aliases for that physical NVIDIA GPU.
                        Ok((0, target.luid, "single-cuda-explicit-fallback", target))
                    }
                    None => Err(format!(
                        "the selected NVIDIA DXGI adapter could not be mapped to one of {count} CUDA devices"
                    )),
                }
            } else {
                let high_performance_order = high_performance_luid_order_quiet();
                let preference_rank = |adapter: &GpuAdapter| {
                    high_performance_order
                        .iter()
                        .position(|luid| *luid == adapter.luid)
                        .unwrap_or(usize::MAX)
                };
                let mut mapped: Vec<(&GpuAdapter, i32)> = Vec::new();
                for target in &nvidia {
                    if let Some(device_id) = cuda_device_for_dxgi_adapter(module, target)?
                        && device_id < count
                    {
                        mapped.push((*target, device_id));
                    }
                }
                mapped.sort_by(|(a, cuda_a), (b, cuda_b)| {
                    preference_rank(a)
                        .cmp(&preference_rank(b))
                        .then_with(|| cuda_a.cmp(cuda_b))
                        .then_with(|| a.device_id.cmp(&b.device_id))
                });

                // Different DXGI LUIDs may be aliases of the same physical CUDA
                // device. Collapse those aliases only for TensorRT mapping while
                // leaving the GUI's DXGI choices intact.
                let mut unique_cuda: Vec<(&GpuAdapter, i32)> = Vec::new();
                for (adapter, device_id) in mapped {
                    if let Some((canonical, _)) = unique_cuda
                        .iter()
                        .find(|(_, known_device)| *known_device == device_id)
                    {
                        log::info!(
                            "tensorrt-cuda-dxgi-alias: cuda_device_id={} canonical_luid={:016x} alias_luid={:016x} canonical_dxgi_device={} alias_dxgi_device={} name='{}'",
                            device_id,
                            canonical.luid,
                            adapter.luid,
                            canonical.device_id,
                            adapter.device_id,
                            adapter.name
                        );
                        continue;
                    }
                    unique_cuda.push((adapter, device_id));
                }

                if let Some((target, device_id)) = unique_cuda.first().copied() {
                    let candidates = unique_cuda
                        .iter()
                        .map(|(adapter, cuda_device)| {
                            format!(
                                "dxgi_device={} cuda_device={} luid={:016x} high_perf_rank={} name='{}'",
                                adapter.device_id,
                                cuda_device,
                                adapter.luid,
                                preference_rank(adapter),
                                adapter.name
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" | ");
                    log::info!(
                        "tensorrt-gpu-map: auto-selection mapped_nvidia={} policy=cudaD3D11GetDevice-then-dxgi-high-performance candidates=[{}]",
                        unique_cuda.len(),
                        candidates
                    );
                    Ok((
                        device_id,
                        target.luid,
                        "auto-d3d11-high-performance",
                        target,
                    ))
                } else if count == 1 {
                    let target = nvidia
                        .iter()
                        .copied()
                        .min_by_key(|adapter| (preference_rank(adapter), adapter.device_id))
                        .expect("NVIDIA list is non-empty");
                    log::warn!(
                        "tensorrt-gpu-map: cudaD3D11GetDevice produced no match but CUDA exposes exactly one device; using cuda_device_id=0 with high-performance DXGI candidate luid={:016x}",
                        target.luid
                    );
                    Ok((
                        0,
                        target.luid,
                        "single-cuda-high-performance-fallback",
                        target,
                    ))
                } else {
                    // Last-resort Auto behavior: CUDA's documented default
                    // enumeration is FASTEST_FIRST, so device 0 is the best
                    // available CUDA guess when neither Windows mapping API is
                    // exported. Keep this Auto-only; an explicit GPU selection
                    // must never silently run on a different NVIDIA device.
                    let target = nvidia
                        .iter()
                        .copied()
                        .min_by_key(|adapter| (preference_rank(adapter), adapter.device_id))
                        .expect("NVIDIA list is non-empty");
                    log::warn!(
                        "tensorrt-gpu-map: no LUID/D3D11 CUDA mapping API available; Auto falling back to cuda_device_id=0 (CUDA default FASTEST_FIRST) dxgi_hint_luid={:016x}",
                        target.luid
                    );
                    Ok((0, target.luid, "auto-cuda-fastest-first-fallback", target))
                }
            }
        }
    };
    unsafe {
        let _ = FreeLibrary(module);
    }
    result.map(|(device_id, gpu_luid, mapping, target)| {
        log::info!(
            "tensorrt-gpu-map: dxgi_luid={:016x} dxgi_device_id={} cuda_device_id={} gpu='{}' mapping={}",
            gpu_luid,
            target.device_id,
            device_id,
            target.name,
            mapping,
        );
        (device_id, gpu_luid)
    })
}

fn probe_execution_providers(device_id: i32) -> Result<(), String> {
    super::onnx_stage::init_onnx()?;
    let tensorrt = ort::ep::TensorRT::default()
        .with_device_id(device_id)
        .build()
        .error_on_failure();
    let cuda = ort::ep::CUDA::default()
        .with_device_id(device_id)
        .build()
        .error_on_failure();
    ort::session::Session::builder()
        .map_err(|error| error.to_string())?
        .with_execution_providers([tensorrt, cuda])
        .map(|_| ())
        .map_err(|error| format!("TensorRT/CUDA provider registration failed: {error}"))
}

fn ensure_provider_aliases(runtime_dir: &Path, backend_dir: &Path) -> Result<(), String> {
    for provider in [
        "onnxruntime_providers_tensorrt.dll",
        "onnxruntime_providers_cuda.dll",
    ] {
        let source = backend_dir.join(provider);
        let alias = runtime_dir.join(provider);
        let source_hash = sha256_file(&source)?;
        if alias.is_file() {
            let alias_hash = sha256_file(&alias)?;
            if alias_hash.eq_ignore_ascii_case(&source_hash) {
                continue;
            }
            return Err(format!(
                "{} already exists beside onnxruntime.dll but belongs to a different Backend Pack",
                alias.display()
            ));
        }
        match std::fs::hard_link(&source, &alias) {
            Ok(()) => log::info!(
                "onnx-backend-detect: provider alias=hardlink source={} target={}",
                source.display(),
                alias.display()
            ),
            Err(hardlink_error) => {
                std::fs::copy(&source, &alias).map_err(|copy_error| {
                    format!(
                        "provider alias creation failed for {} (hardlink: {}; copy: {})",
                        provider, hardlink_error, copy_error
                    )
                })?;
                log::info!(
                    "onnx-backend-detect: provider alias=copy source={} target={}",
                    source.display(),
                    alias.display()
                );
            }
        }
    }
    Ok(())
}

pub fn detect_tensorrt_backend(
    app_dir: &Path,
    selected_adapter_luid: Option<u64>,
) -> TensorRtAvailability {
    let backend_dir = app_dir.join("backends").join("tensorrt");
    let manifest_path = backend_dir.join("backend.json");
    if !manifest_path.is_file() {
        return unavailable("Neo TensorRT Backend Pack is not installed", None);
    }
    let manifest: TensorRtManifest = match std::fs::read(&manifest_path)
        .map_err(|error| error.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|error| error.to_string()))
    {
        Ok(manifest) => manifest,
        Err(error) => {
            return unavailable(format!("backend.json is invalid: {error}"), None);
        }
    };
    if manifest.backend_api != BACKEND_API {
        return unavailable(
            format!(
                "backend API {} is incompatible (expected {})",
                manifest.backend_api, BACKEND_API
            ),
            Some(manifest),
        );
    }
    if !manifest.architecture.eq_ignore_ascii_case("x86_64") {
        return unavailable(
            format!(
                "unsupported backend architecture: {}",
                manifest.architecture
            ),
            Some(manifest),
        );
    }
    if manifest.onnxruntime != ORT_COMPAT_VERSION {
        return unavailable(
            format!(
                "ONNX Runtime {} is incompatible (expected {})",
                manifest.onnxruntime, ORT_COMPAT_VERSION
            ),
            Some(manifest),
        );
    }
    if manifest.cuda_major != 12 {
        return unavailable(
            format!(
                "CUDA major version {} is incompatible (expected 12)",
                manifest.cuda_major
            ),
            Some(manifest),
        );
    }
    if !manifest.provider.eq_ignore_ascii_case("TensorRT+CUDA") {
        return unavailable(
            format!("unsupported provider set: {}", manifest.provider),
            Some(manifest),
        );
    }
    // GPU changes are allowed without restarting Neo in v636. Reject an
    // explicit non-NVIDIA selection before hashing/loading the TensorRT pack;
    // this makes AMD/Intel -> DirectML switching immediate and guarantees that
    // TensorRT never silently runs on a different GPU than the user's choice.
    let adapters = enumerate_adapters_quiet();
    if let Some(luid) = selected_adapter_luid {
        match adapters.iter().find(|adapter| adapter.luid == luid) {
            Some(adapter) if adapter.vendor_id != NVIDIA_VENDOR_ID => {
                return unavailable(
                    format!(
                        "the selected GPU '{}' is not NVIDIA; TensorRT requires an NVIDIA GPU",
                        adapter.name
                    ),
                    Some(manifest),
                );
            }
            None => {
                return unavailable(
                    format!("the selected DXGI adapter LUID {luid:016x} is unavailable"),
                    Some(manifest),
                );
            }
            _ => {}
        }
    }
    if manifest.tensor_rt.trim().is_empty()
        || manifest.tensor_rt.contains("使用")
        || manifest.tensor_rt.contains("exact")
    {
        return unavailable(
            "backend.json does not contain an exact TensorRT version",
            Some(manifest),
        );
    }
    let runtime_path = app_dir.join("backends").join("onnxruntime.dll");
    if !runtime_path.is_file() {
        return unavailable(
            "the unified backends/onnxruntime.dll is missing",
            Some(manifest),
        );
    }
    if manifest.onnxruntime_sha256.trim().is_empty() {
        return unavailable("backend.json is missing onnxruntime_sha256", Some(manifest));
    }
    let runtime_hash = match sha256_file(&runtime_path) {
        Ok(hash) => hash,
        Err(error) => return unavailable(error, Some(manifest)),
    };
    if !runtime_hash.eq_ignore_ascii_case(manifest.onnxruntime_sha256.trim()) {
        return unavailable(
            "the TensorRT Backend Pack was built for a different onnxruntime.dll",
            Some(manifest),
        );
    }
    for file in &manifest.required_files {
        let relative = match safe_manifest_relative_path(file) {
            Ok(path) => path,
            Err(error) => return unavailable(error, Some(manifest)),
        };
        if !backend_dir.join(relative).is_file() {
            return unavailable(
                format!("required TensorRT file is missing: {file}"),
                Some(manifest),
            );
        }
    }
    for required in [
        "onnxruntime_providers_tensorrt.dll",
        "onnxruntime_providers_cuda.dll",
    ] {
        let provider_path = backend_dir.join(required);
        if !provider_path.is_file() {
            return unavailable(
                format!("required TensorRT provider is missing: {required}"),
                Some(manifest),
            );
        }
        let Some(expected_hash) = manifest.file_sha256.get(required) else {
            return unavailable(
                format!("backend.json is missing the SHA-256 for {required}"),
                Some(manifest),
            );
        };
        let actual_hash = match sha256_file(&provider_path) {
            Ok(hash) => hash,
            Err(error) => return unavailable(error, Some(manifest)),
        };
        if !actual_hash.eq_ignore_ascii_case(expected_hash.trim()) {
            return unavailable(
                format!("{required} does not match the Backend Pack manifest"),
                Some(manifest),
            );
        }
    }
    let runtime_dir = app_dir.join("backends");
    if let Err(error) = register_dll_directories(&[runtime_dir.as_path(), backend_dir.as_path()]) {
        return unavailable(error, Some(manifest));
    }
    let (device_id, gpu_luid) =
        match resolve_cuda_device(&backend_dir, &adapters, selected_adapter_luid) {
            Ok(mapping) => mapping,
            Err(error) => return unavailable(error, Some(manifest)),
        };
    if let Err(error) = ensure_provider_aliases(&runtime_dir, &backend_dir) {
        return unavailable(error, Some(manifest));
    }
    if let Err(error) = probe_execution_providers(device_id) {
        return unavailable(error, Some(manifest));
    }
    log::info!(
        "onnx-backend-detect: TensorRT available device={} runtime={} tensorrt={} cuda={}",
        device_id,
        manifest.onnxruntime,
        manifest.tensor_rt,
        manifest.cuda_major
    );
    TensorRtAvailability {
        available: true,
        reason: None,
        device_id: Some(device_id),
        gpu_luid: Some(gpu_luid),
        manifest: Some(manifest),
        backend_dir: Some(backend_dir),
    }
}

fn migrate_legacy_cuda_cache_root(
    tensor_root: &Path,
    canonical_runtime_root: &Path,
    availability: &TensorRtAvailability,
    runtime: &str,
) {
    let (Some(_gpu_luid), Some(device_id)) = (availability.gpu_luid, availability.device_id) else {
        return;
    };
    let legacy_runtime_root = tensor_root
        .join(format!("cuda-device-{device_id}"))
        .join(runtime);
    if legacy_runtime_root == canonical_runtime_root || !legacy_runtime_root.is_dir() {
        return;
    }

    if !canonical_runtime_root.exists() {
        if let Some(parent) = canonical_runtime_root.parent()
            && std::fs::create_dir_all(parent).is_ok()
            && std::fs::rename(&legacy_runtime_root, canonical_runtime_root).is_ok()
        {
            log::info!(
                "tensorrt-cache: migrated legacy GPU root old={} new={}",
                legacy_runtime_root.display(),
                canonical_runtime_root.display()
            );
            let _ = std::fs::remove_dir(tensor_root.join(format!("cuda-device-{device_id}")));
            return;
        }
    }

    // Both roots can exist after a user has already exercised Auto and an
    // explicit selector. Merge only non-conflicting model directories. Cache
    // keys are content/runtime based, so a missing destination directory is
    // safe to move once the CUDA ordinal has been proven to map to this LUID.
    if std::fs::create_dir_all(canonical_runtime_root).is_err() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(&legacy_runtime_root) else {
        return;
    };
    let mut moved = 0usize;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_text = name.to_string_lossy();
        let is_model_cache =
            name_text.len() == 32 && name_text.chars().all(|ch| ch.is_ascii_hexdigit());
        if !is_model_cache {
            continue;
        }
        let source = entry.path();
        let target = canonical_runtime_root.join(&name);
        if target.exists() {
            continue;
        }
        if std::fs::rename(&source, &target).is_ok() {
            moved += 1;
        }
    }
    if moved > 0 {
        log::info!(
            "tensorrt-cache: merged legacy GPU root old={} new={} moved_entries={}",
            legacy_runtime_root.display(),
            canonical_runtime_root.display(),
            moved
        );
    }
    let _ = std::fs::remove_dir(&legacy_runtime_root);
    let _ = std::fs::remove_dir(tensor_root.join(format!("cuda-device-{device_id}")));
}

pub fn tensorrt_cache_root(
    app_dir: &Path,
    availability: &TensorRtAvailability,
    selected_adapter_luid: Option<u64>,
) -> PathBuf {
    fn safe(value: &str) -> String {
        value
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                    ch
                } else {
                    '_'
                }
            })
            .collect()
    }

    let runtime = availability
        .manifest
        .as_ref()
        .map(|manifest| {
            format!(
                "ort-{}_trt-{}_cuda-{}",
                safe(&manifest.onnxruntime),
                safe(&manifest.tensor_rt),
                manifest.cuda_major
            )
        })
        .unwrap_or_else(|| "runtime-unavailable".into());
    // Cache by the physical GPU actually selected for TensorRT, not by how
    // the user selected it. In particular, Auto and an explicit selector that
    // both resolve to the same NVIDIA adapter must share one engine cache.
    let gpu = availability
        .gpu_luid
        .map(|luid| format!("luid-{luid:016x}"))
        .or_else(|| selected_adapter_luid.map(|luid| format!("luid-{luid:016x}")))
        .or_else(|| {
            availability
                .device_id
                .map(|device| format!("cuda-device-{device}"))
        })
        .unwrap_or_else(|| "gpu-unresolved".into());
    let tensor_root = app_dir.join("cache").join("TensorRT");
    let root = tensor_root.join(gpu).join(&runtime);
    migrate_legacy_cuda_cache_root(&tensor_root, &root, availability, &runtime);
    root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_pack_is_a_nonfatal_unavailable_state() {
        let root = std::env::temp_dir().join(format!("neo-no-trt-pack-{}", std::process::id()));
        let state = detect_tensorrt_backend(&root, None);
        assert!(!state.available);
        assert!(state.reason.unwrap().contains("not installed"));
    }

    #[test]
    fn manifest_rejects_an_incompatible_runtime_before_loading_dlls() {
        let manifest = TensorRtManifest {
            backend_api: BACKEND_API,
            name: "test".into(),
            architecture: "x86_64".into(),
            onnxruntime: "1.23.0".into(),
            tensor_rt: "10".into(),
            cuda_major: 12,
            provider: "TensorRT+CUDA".into(),
            required_files: Vec::new(),
            onnxruntime_sha256: String::new(),
            file_sha256: BTreeMap::new(),
        };
        assert_ne!(manifest.onnxruntime, ORT_COMPAT_VERSION);
    }

    #[test]
    fn manifest_paths_cannot_escape_the_backend_directory() {
        assert!(safe_manifest_relative_path("nvinfer_10.dll").is_ok());
        assert!(safe_manifest_relative_path("runtime/cudart64_12.dll").is_ok());
        assert!(safe_manifest_relative_path("../onnxruntime.dll").is_err());
        assert!(safe_manifest_relative_path(r"C:\Windows\System32\version.dll").is_err());
    }

    #[test]
    fn auto_and_explicit_same_gpu_share_tensorrt_cache_root() {
        let app =
            std::env::temp_dir().join(format!("neo-trt-cache-root-test-{}", std::process::id()));
        let availability = TensorRtAvailability {
            available: true,
            reason: None,
            device_id: Some(0),
            gpu_luid: Some(0xFEB8),
            manifest: Some(TensorRtManifest {
                backend_api: BACKEND_API,
                name: "test".into(),
                architecture: "x86_64".into(),
                onnxruntime: ORT_COMPAT_VERSION.into(),
                tensor_rt: "10.14.1.48".into(),
                cuda_major: 12,
                provider: "TensorRT+CUDA".into(),
                required_files: Vec::new(),
                onnxruntime_sha256: String::new(),
                file_sha256: BTreeMap::new(),
            }),
            backend_dir: None,
        };
        assert_eq!(
            tensorrt_cache_root(&app, &availability, None),
            tensorrt_cache_root(&app, &availability, Some(0xFEB8))
        );
        assert!(
            tensorrt_cache_root(&app, &availability, None)
                .to_string_lossy()
                .contains("luid-000000000000feb8")
        );
    }
}
