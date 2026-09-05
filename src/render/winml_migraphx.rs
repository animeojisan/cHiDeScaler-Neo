//! Optional Windows ML MIGraphX provider used only by NeoAccel experiments.
//!
//! The stable DirectML path does not depend on this module.  MIGraphX is
//! dynamically registered only when a Windows ML runtime has explicitly been
//! prepared next to the application (or CHIDE_WINML_RUNTIME_DIR points to it).

use anyhow::{Context, Result, anyhow};
use ort::{
    environment::Environment,
    memory::DeviceType,
    session::{Session, builder::GraphOptimizationLevel},
};
use std::{
    ffi::{CString, c_char, c_void},
    path::{Path, PathBuf},
    ptr,
    sync::OnceLock,
};
use windows::{
    Win32::{
        Foundation::HMODULE,
        System::LibraryLoader::{GetProcAddress, LoadLibraryW},
    },
    core::{HRESULT, PCSTR, PCWSTR},
};

pub const MIGRAPHX_EP_NAME: &str = "MIGraphXExecutionProvider";
const WINML_DLL_NAME: &str = "Microsoft.Windows.AI.MachineLearning.dll";
static MIGRAPHX_REGISTERED: OnceLock<std::result::Result<(), String>> = OnceLock::new();

type CatalogHandle = *mut c_void;
type EpHandle = *mut c_void;
type CatalogCreate = unsafe extern "system" fn(*mut CatalogHandle) -> HRESULT;
type CatalogRelease = unsafe extern "system" fn(CatalogHandle);
type CatalogFindProvider = unsafe extern "system" fn(
    CatalogHandle,
    *const c_char,
    *const c_char,
    *mut EpHandle,
) -> HRESULT;
type EpEnsureReady = unsafe extern "system" fn(EpHandle) -> HRESULT;
type EpGetLibraryPathSize = unsafe extern "system" fn(EpHandle, *mut usize) -> HRESULT;
type EpGetLibraryPath =
    unsafe extern "system" fn(EpHandle, usize, *mut c_char, *mut usize) -> HRESULT;

struct WinMlCatalog {
    catalog: CatalogHandle,
    release: CatalogRelease,
    _module: HMODULE,
    ep_library_path: PathBuf,
}

impl Drop for WinMlCatalog {
    fn drop(&mut self) {
        if !self.catalog.is_null() {
            // SAFETY: catalog was returned by WinMLEpCatalogCreate and remains
            // valid while Microsoft.Windows.AI.MachineLearning.dll is loaded.
            unsafe { (self.release)(self.catalog) };
        }
    }
}

impl WinMlCatalog {
    fn prepare_migraphx(runtime_dir: &Path) -> Result<Self> {
        let winml_dll = runtime_dir.join(WINML_DLL_NAME);
        anyhow::ensure!(
            winml_dll.is_file(),
            "Windows ML runtime not found: {}",
            winml_dll.display()
        );
        let module = load_library(&winml_dll)
            .with_context(|| format!("failed to load {}", winml_dll.display()))?;

        // SAFETY: signatures match the public WinMLEpCatalog.h API. The module
        // is intentionally kept loaded for the lifetime of the process.
        let create: CatalogCreate = unsafe { load_symbol(module, "WinMLEpCatalogCreate")? };
        let release: CatalogRelease = unsafe { load_symbol(module, "WinMLEpCatalogRelease")? };
        let find_provider: CatalogFindProvider =
            unsafe { load_symbol(module, "WinMLEpCatalogFindProvider")? };
        let ensure_ready: EpEnsureReady = unsafe { load_symbol(module, "WinMLEpEnsureReady")? };
        let get_path_size: EpGetLibraryPathSize =
            unsafe { load_symbol(module, "WinMLEpGetLibraryPathSize")? };
        let get_path: EpGetLibraryPath = unsafe { load_symbol(module, "WinMLEpGetLibraryPath")? };

        let mut catalog: CatalogHandle = ptr::null_mut();
        // SAFETY: points to writable storage for a single catalog handle.
        check_hr(unsafe { create(&mut catalog) }, "WinMLEpCatalogCreate")?;
        anyhow::ensure!(!catalog.is_null(), "Windows ML returned a null catalog");

        let result = (|| {
            let mut provider: EpHandle = ptr::null_mut();
            let mut last_find_hr = HRESULT(0);
            for candidate in [MIGRAPHX_EP_NAME, "MIGraphX"] {
                let provider_name = CString::new(candidate).expect("provider name has no NUL");
                // SAFETY: all strings are NUL terminated and handles are valid.
                last_find_hr = unsafe {
                    find_provider(catalog, provider_name.as_ptr(), ptr::null(), &mut provider)
                };
                if !last_find_hr.is_err() && !provider.is_null() {
                    break;
                }
                provider = ptr::null_mut();
            }
            if provider.is_null() {
                check_hr(
                    last_find_hr,
                    "WinMLEpCatalogFindProvider(MIGraphXExecutionProvider)",
                )?;
                return Err(anyhow!("MIGraphX is unavailable for this PC/driver"));
            }

            // EnsureReady may install/update the compatible AMD EP package and
            // adds it to this process package graph. This must happen in-process.
            check_hr(
                unsafe { ensure_ready(provider) },
                "WinMLEpEnsureReady(MIGraphX)",
            )?;

            let mut path_size = 0usize;
            check_hr(
                unsafe { get_path_size(provider, &mut path_size) },
                "WinMLEpGetLibraryPathSize(MIGraphX)",
            )?;
            anyhow::ensure!(path_size > 1, "MIGraphX returned an empty library path");
            let mut path_utf8 = vec![0u8; path_size];
            check_hr(
                unsafe {
                    get_path(
                        provider,
                        path_utf8.len(),
                        path_utf8.as_mut_ptr().cast::<c_char>(),
                        ptr::null_mut(),
                    )
                },
                "WinMLEpGetLibraryPath(MIGraphX)",
            )?;
            if let Some(nul) = path_utf8.iter().position(|byte| *byte == 0) {
                path_utf8.truncate(nul);
            }
            let path = PathBuf::from(
                String::from_utf8(path_utf8).context("MIGraphX DLL path is not valid UTF-8")?,
            );
            anyhow::ensure!(
                path.is_file(),
                "MIGraphX library path does not exist: {}",
                path.display()
            );
            Ok(path)
        })();

        match result {
            Ok(ep_library_path) => Ok(Self {
                catalog,
                release,
                _module: module,
                ep_library_path,
            }),
            Err(error) => {
                // SAFETY: catalog was created above and has not been released.
                unsafe { release(catalog) };
                Err(error)
            }
        }
    }
}

pub fn runtime_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("CHIDE_WINML_RUNTIME_DIR") {
        let path = PathBuf::from(path);
        if path.join(WINML_DLL_NAME).is_file() {
            return Some(path);
        }
    }
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let candidate = exe_dir.join("backends_winml_test");
    candidate
        .join(WINML_DLL_NAME)
        .is_file()
        .then_some(candidate)
}

pub fn available_runtime() -> bool {
    runtime_dir().is_some()
}

fn ensure_registered() -> Result<()> {
    let result = MIGRAPHX_REGISTERED.get_or_init(|| {
        (|| -> Result<()> {
            let runtime_dir = runtime_dir().ok_or_else(|| {
                anyhow!(
                    "Windows ML test runtime is not prepared (expected backends_winml_test or CHIDE_WINML_RUNTIME_DIR)"
                )
            })?;
            let env = Environment::current().context("ONNX Runtime environment unavailable")?;

            if env.devices().any(|device| {
                device.ty() == DeviceType::GPU
                    && device.ep().is_ok_and(|name| name == MIGRAPHX_EP_NAME)
            }) {
                log::info!("neoaccel-migraphx-register: already-registered=true");
                return Ok(());
            }

            let catalog = WinMlCatalog::prepare_migraphx(&runtime_dir)?;
            let ep_path = catalog.ep_library_path.clone();
            let registration = env
                .register_ep_library(MIGRAPHX_EP_NAME, &ep_path)
                .with_context(|| {
                    format!(
                        "failed to register Windows ML MIGraphX EP library: {}",
                        ep_path.display()
                    )
                })?;

            let devices = env
                .devices()
                .filter(|device| {
                    device.ty() == DeviceType::GPU
                        && device.ep().is_ok_and(|name| name == MIGRAPHX_EP_NAME)
                })
                .count();
            anyhow::ensure!(devices > 0, "MIGraphX registered but exposed no GPU device");
            log::info!(
                "neoaccel-migraphx-register: runtime={} ep={} gpu_devices={} result=ready",
                runtime_dir.display(),
                ep_path.display(),
                devices
            );

            // Both objects must outlive every session using the plugin. They are
            // intentionally process-lifetime resources in this optional backend.
            std::mem::forget(registration);
            std::mem::forget(catalog);
            Ok(())
        })()
        .map_err(|error| format!("{error:#}"))
    });
    match result {
        Ok(()) => Ok(()),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

pub fn create_session(model: &Path) -> Result<Session> {
    ensure_registered()?;
    let env = Environment::current().context("ONNX Runtime environment unavailable")?;
    let devices = env.devices().filter(|device| {
        device.ty() == DeviceType::GPU && device.ep().is_ok_and(|name| name == MIGRAPHX_EP_NAME)
    });
    Session::builder()
        .map_err(|error| anyhow!(error.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|error| anyhow!(error.to_string()))?
        .with_devices(devices, None)
        .map_err(|error| anyhow!(error.to_string()))?
        .with_disable_cpu_fallback()
        .map_err(|error| anyhow!(error.to_string()))?
        .with_memory_pattern(false)
        .map_err(|error| anyhow!(error.to_string()))?
        .commit_from_file(model)
        .map_err(|error| anyhow!(error.to_string()))
}

fn load_library(path: &Path) -> Result<HMODULE> {
    let wide: Vec<u16> = path
        .as_os_str()
        .to_string_lossy()
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    // SAFETY: wide is a valid NUL-terminated UTF-16 path for this call.
    unsafe { LoadLibraryW(PCWSTR(wide.as_ptr())) }
        .with_context(|| format!("LoadLibraryW failed: {}", path.display()))
}

unsafe fn load_symbol<T: Copy>(module: HMODULE, name: &str) -> Result<T> {
    let c_name = CString::new(name).context("invalid symbol name")?;
    // SAFETY: the module is live and the symbol name is NUL-terminated.
    let proc = unsafe { GetProcAddress(module, PCSTR(c_name.as_ptr().cast::<u8>())) }
        .ok_or_else(|| anyhow!("Windows ML export not found: {name}"))?;
    // SAFETY: caller chooses T to exactly match WinMLEpCatalog.h.
    Ok(unsafe { std::mem::transmute_copy(&proc) })
}

fn check_hr(hr: HRESULT, operation: &str) -> Result<()> {
    if hr.is_err() {
        Err(anyhow!(
            "{operation} failed with HRESULT 0x{:08X}",
            hr.0 as u32
        ))
    } else {
        Ok(())
    }
}
