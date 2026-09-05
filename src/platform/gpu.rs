//! DXGI adapter discovery plus the legacy v635 restart-time handoff helper.
//!
//! v636 selects DirectML/TensorRT/Vulkan compute devices directly and no longer
//! restarts Neo for an ordinary GPU change. The older temporary Windows Graphics
//! Settings handoff remains here only for compatibility with a v635 parent that
//! may already have launched this executable with a ready marker.

use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use windows::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
    IDXGIAdapter1, IDXGIFactory1, IDXGIFactory6,
};
use windows::Win32::System::Registry::{
    HKEY, HKEY_CURRENT_USER, REG_SZ, RRF_RT_REG_SZ, RegCloseKey, RegCreateKeyW, RegDeleteKeyValueW,
    RegGetValueW, RegSetKeyValueW,
};
use windows::core::PCWSTR;

const USER_GPU_PREFERENCES_KEY: &str = r"Software\Microsoft\DirectX\UserGpuPreferences";
// Windows 11 writes this flag for the "Specific GPU" picker. It is intentionally
// confined to the restart handoff and restored immediately after Neo's WGPU/WGL
// devices exist. Generic Auto/high-performance policy remains untouched.
const WINDOWS_SPECIFIC_GPU_PREFERENCE: u32 = 0x4000_0000;
const GPU_HANDOFF_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuAdapter {
    /// DirectML's device id follows IDXGIFactory1::EnumAdapters1 order.
    pub device_id: i32,
    pub luid: u64,
    pub name: String,
    pub dedicated_video_memory: u64,
    pub vendor_id: u32,
    pub pci_device_id: u32,
    pub subsys_id: u32,
    pub revision: u32,
}

fn luid_u64(luid: windows::Win32::Foundation::LUID) -> u64 {
    ((luid.HighPart as u32 as u64) << 32) | luid.LowPart as u64
}

fn wide_name(value: &[u16]) -> String {
    let len = value.iter().position(|c| *c == 0).unwrap_or(value.len());
    String::from_utf16_lossy(&value[..len]).trim().to_string()
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn normalize_adapters(raw: Vec<GpuAdapter>, log_detected: bool) -> Vec<GpuAdapter> {
    // One physical DXGI adapter is identified by its LUID. Some driver / upgrade
    // combinations can expose the same LUID more than once in EnumAdapters1.
    // Keep the first (lowest DXGI / DirectML ordinal) and never let a duplicate
    // row become a fake second GPU in the GUI or TensorRT ambiguity logic.
    let mut adapters: Vec<GpuAdapter> = Vec::with_capacity(raw.len());
    for adapter in raw {
        if let Some(canonical) = adapters.iter().find(|known| known.luid == adapter.luid) {
            if log_detected {
                log::warn!(
                    "gpu-detected-duplicate: ignored_dml_device_id={} canonical_dml_device_id={} luid={:016x} ignored_name='{}' canonical_name='{}' reason=same-luid",
                    adapter.device_id,
                    canonical.device_id,
                    adapter.luid,
                    adapter.name,
                    canonical.name
                );
            } else {
                log::debug!(
                    "gpu-detected-duplicate: ignored_dml_device_id={} canonical_dml_device_id={} luid={:016x} ignored_name='{}' canonical_name='{}' reason=same-luid",
                    adapter.device_id,
                    canonical.device_id,
                    adapter.luid,
                    adapter.name,
                    canonical.name
                );
            }
            continue;
        }
        adapters.push(adapter);
    }
    if log_detected {
        for adapter in &adapters {
            log::info!(
                "gpu-detected: dml_device_id={} luid={:016x} vram_mb={} pci={:04x}:{:04x}:{:08x} name={}",
                adapter.device_id,
                adapter.luid,
                adapter.dedicated_video_memory / (1024 * 1024),
                adapter.vendor_id,
                adapter.pci_device_id,
                adapter.subsys_id,
                adapter.name
            );
        }
    }
    adapters
}

fn enumerate_adapters_impl(log_detected: bool) -> Vec<GpuAdapter> {
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else {
        log::warn!("gpu-enumeration: CreateDXGIFactory1 failed");
        return Vec::new();
    };
    let mut raw = Vec::new();
    for index in 0u32.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
            break;
        };
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }
        raw.push(GpuAdapter {
            device_id: index as i32,
            luid: luid_u64(desc.AdapterLuid),
            name: wide_name(&desc.Description),
            dedicated_video_memory: desc.DedicatedVideoMemory as u64,
            vendor_id: desc.VendorId,
            pci_device_id: desc.DeviceId,
            subsys_id: desc.SubSysId,
            revision: desc.Revision,
        });
    }
    normalize_adapters(raw, log_detected)
}

pub fn enumerate_adapters() -> Vec<GpuAdapter> {
    enumerate_adapters_impl(true)
}

/// Used by the GUI after TensorRT detection has already logged the same DXGI
/// list. Avoid duplicate startup rows while retaining one authoritative list.
pub fn enumerate_adapters_quiet() -> Vec<GpuAdapter> {
    enumerate_adapters_impl(false)
}

/// Return physical DXGI adapter LUIDs in Windows' explicit HIGH_PERFORMANCE
/// preference order. This is a ranking hint only: DirectML device ids still
/// come from EnumAdapters1 and are resolved by LUID. On systems without
/// IDXGIFactory6 support, callers simply fall back to the canonical DXGI order.
pub fn high_performance_luid_order_quiet() -> Vec<u64> {
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory6>() }) else {
        return Vec::new();
    };
    let mut luids = Vec::new();
    for index in 0u32.. {
        let Ok(adapter) = (unsafe {
            factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                index,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            )
        }) else {
            break;
        };
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }
        let luid = luid_u64(desc.AdapterLuid);
        if !luids.contains(&luid) {
            luids.push(luid);
        }
    }
    luids
}

pub fn device_id_for_luid(adapters: &[GpuAdapter], luid: Option<u64>) -> Option<i32> {
    luid.and_then(|wanted| {
        adapters
            .iter()
            .find(|adapter| adapter.luid == wanted)
            .map(|adapter| adapter.device_id)
    })
}

pub fn adapter_for_luid(adapters: &[GpuAdapter], luid: Option<u64>) -> Option<&GpuAdapter> {
    let wanted = luid?;
    adapters.iter().find(|adapter| adapter.luid == wanted)
}

/// User-facing label for a physical adapter. Two real adapters may have the
/// exact same model name, so add a stable GPU ordinal only in that case. Same-
/// LUID duplicates have already been removed by `normalize_adapters`.
pub fn adapter_display_name(adapters: &[GpuAdapter], adapter: &GpuAdapter) -> String {
    let same_name: Vec<&GpuAdapter> = adapters
        .iter()
        .filter(|candidate| candidate.name.eq_ignore_ascii_case(&adapter.name))
        .collect();
    if same_name.len() <= 1 {
        return adapter.name.clone();
    }
    let ordinal = same_name
        .iter()
        .position(|candidate| candidate.luid == adapter.luid)
        .map(|index| index + 1)
        .unwrap_or(1);
    format!("{} (GPU {})", adapter.name, ordinal)
}

/// Windows 11's per-app "Specific GPU" identifier is based on the adapter's
/// PCI vendor/device/subsystem tuple rather than the volatile DXGI list index.
fn specific_adapter_id(adapter: &GpuAdapter) -> String {
    format!(
        "{:04X}&{:04X}&{:08X}",
        adapter.vendor_id, adapter.pci_device_id, adapter.subsys_id
    )
}

fn specific_gpu_preference(adapter: &GpuAdapter) -> String {
    format!(
        "SpecificAdapter={};GpuPreference={};",
        specific_adapter_id(adapter),
        WINDOWS_SPECIFIC_GPU_PREFERENCE
    )
}

fn current_exe_value_name() -> Result<String> {
    let exe = std::env::current_exe().map_err(|error| anyhow!("current_exe failed: {error}"))?;
    Ok(exe.to_string_lossy().into_owned())
}

fn read_current_exe_preference() -> Result<Option<String>> {
    let subkey = wide(USER_GPU_PREFERENCES_KEY);
    let value_name = wide(&current_exe_value_name()?);
    let mut bytes = 0u32;
    let status = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            PCWSTR(subkey.as_ptr()),
            PCWSTR(value_name.as_ptr()),
            RRF_RT_REG_SZ,
            None,
            None,
            Some(&mut bytes),
        )
    };
    if status == ERROR_FILE_NOT_FOUND {
        return Ok(None);
    }
    if status != ERROR_SUCCESS {
        return Err(anyhow!("RegGetValueW(size) failed: {}", status.0));
    }
    if bytes == 0 {
        return Ok(Some(String::new()));
    }
    let mut data = vec![0u8; bytes as usize];
    let status = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            PCWSTR(subkey.as_ptr()),
            PCWSTR(value_name.as_ptr()),
            RRF_RT_REG_SZ,
            None,
            Some(data.as_mut_ptr().cast()),
            Some(&mut bytes),
        )
    };
    if status != ERROR_SUCCESS {
        return Err(anyhow!("RegGetValueW(data) failed: {}", status.0));
    }
    let usable = (bytes as usize).min(data.len());
    let units: Vec<u16> = data[..usable]
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    let end = units
        .iter()
        .position(|value| *value == 0)
        .unwrap_or(units.len());
    Ok(Some(String::from_utf16_lossy(&units[..end])))
}

fn ensure_gpu_preferences_key() -> Result<()> {
    let subkey = wide(USER_GPU_PREFERENCES_KEY);
    let mut created = HKEY::default();
    let status = unsafe { RegCreateKeyW(HKEY_CURRENT_USER, PCWSTR(subkey.as_ptr()), &mut created) };
    if status != ERROR_SUCCESS {
        return Err(anyhow!(
            "RegCreateKeyW(UserGpuPreferences) failed: {}",
            status.0
        ));
    }
    let _ = unsafe { RegCloseKey(created) };
    Ok(())
}

fn write_current_exe_preference(value: Option<&str>) -> Result<()> {
    let subkey = wide(USER_GPU_PREFERENCES_KEY);
    let value_name = wide(&current_exe_value_name()?);
    let status = if let Some(value) = value {
        ensure_gpu_preferences_key()?;
        let value = wide(value);
        unsafe {
            RegSetKeyValueW(
                HKEY_CURRENT_USER,
                PCWSTR(subkey.as_ptr()),
                PCWSTR(value_name.as_ptr()),
                REG_SZ.0,
                Some(value.as_ptr().cast()),
                (value.len() * std::mem::size_of::<u16>()) as u32,
            )
        }
    } else {
        unsafe {
            RegDeleteKeyValueW(
                HKEY_CURRENT_USER,
                PCWSTR(subkey.as_ptr()),
                PCWSTR(value_name.as_ptr()),
            )
        }
    };
    if status == ERROR_SUCCESS || (value.is_none() && status == ERROR_FILE_NOT_FOUND) {
        Ok(())
    } else {
        Err(anyhow!(
            "Windows GPU preference update failed: {}",
            status.0
        ))
    }
}

fn spawn_restarted_exe(ready_path: Option<&Path>) -> Result<std::process::Child> {
    let exe = std::env::current_exe().map_err(|error| anyhow!("current_exe failed: {error}"))?;
    let mut command = Command::new(exe);
    if let Some(path) = ready_path {
        command.env("NEO_GPU_HANDOFF_READY", path);
    }
    command
        .spawn()
        .map_err(|error| anyhow!("GPU-selection relaunch failed: {error}"))
}

fn ready_path_for_parent() -> PathBuf {
    std::env::temp_dir().join(format!(
        "cHiDeScaler-Neo-gpu-handoff-{}-{}.ready",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or_default()
    ))
}

/// Relaunch Neo after the GUI/render worker has already shut down. `None`
/// means Neo Auto and performs no registry mutation. An explicit adapter uses a
/// temporary Windows per-app exact-adapter preference only until the child has
/// created its WGPU/WGL devices, then restores the exact previous user value.
pub fn relaunch_for_selection(selected_luid: Option<u64>) -> Result<()> {
    if selected_luid.is_none() {
        let _ = spawn_restarted_exe(None)?;
        log::info!("gpu-selection-relaunch: requested=Auto registry_override=false");
        return Ok(());
    }

    let adapters = enumerate_adapters_quiet();
    let adapter = adapter_for_luid(&adapters, selected_luid)
        .ok_or_else(|| anyhow!("selected GPU is no longer present"))?;
    let ready_path = ready_path_for_parent();
    let _ = std::fs::remove_file(&ready_path);

    // The exact-adapter Windows setting is only an optimization for moving
    // WGPU/WGL. It must never be required for the user's explicit ONNX choice.
    // If registry access is unavailable, still relaunch once with the handoff
    // marker so the child skips another relaunch and DirectML/TensorRT can use
    // the selected adapter through the normal LUID path.
    let previous = match read_current_exe_preference() {
        Ok(previous) => previous,
        Err(error) => {
            log::warn!(
                "gpu-selection-handoff: exact-render-override=unavailable stage=read error={error:#} fallback=explicit-onnx-only"
            );
            let _ = spawn_restarted_exe(None)?;
            return Ok(());
        }
    };
    let temporary = specific_gpu_preference(adapter);
    if let Err(error) = write_current_exe_preference(Some(&temporary)) {
        log::warn!(
            "gpu-selection-handoff: exact-render-override=unavailable stage=write error={error:#} fallback=explicit-onnx-only"
        );
        let _ = spawn_restarted_exe(None)?;
        return Ok(());
    }

    log::info!(
        "gpu-selection-handoff: requested_luid={:016x} dml_device_id={} name='{}' specific_adapter={} temporary_registry=true previous_present={}",
        adapter.luid,
        adapter.device_id,
        adapter.name,
        specific_adapter_id(adapter),
        previous.is_some()
    );

    let spawn = spawn_restarted_exe(Some(&ready_path));
    if let Err(error) = spawn {
        let _ = write_current_exe_preference(previous.as_deref());
        return Err(error);
    }

    let deadline = Instant::now() + GPU_HANDOFF_TIMEOUT;
    let mut child_ready = false;
    while Instant::now() < deadline {
        if ready_path.is_file() {
            child_ready = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let restore_result = write_current_exe_preference(previous.as_deref());
    let _ = std::fs::remove_file(&ready_path);
    restore_result?;
    if child_ready {
        log::info!(
            "gpu-selection-handoff: child-render-ready=true previous-windows-preference-restored=true"
        );
    } else {
        log::warn!(
            "gpu-selection-handoff: child-render-ready=false timeout_ms={} previous-windows-preference-restored=true",
            GPU_HANDOFF_TIMEOUT.as_millis()
        );
    }
    Ok(())
}

/// Called by the restarted child once both eframe/WGPU initialization and the
/// render worker's WGL context exist. Creating this tiny file is the only
/// parent/child synchronization required for the temporary registry handoff.
pub fn signal_relaunch_render_ready(path: &Path, actual_luid: Option<u64>, renderer: &str) {
    let body = format!(
        "render_ready=1\nactual_luid={}\nrenderer={}\n",
        actual_luid
            .map(|luid| format!("{luid:016x}"))
            .unwrap_or_else(|| "unknown".to_string()),
        renderer.replace('\r', " ").replace('\n', " ")
    );
    if let Err(error) = std::fs::write(path, body) {
        log::warn!(
            "gpu-selection-handoff-ready-signal-failed: path={} error={error}",
            path.display()
        );
    } else {
        log::info!(
            "gpu-selection-handoff-ready: path={} actual_luid={} renderer='{}'",
            path.display(),
            actual_luid
                .map(|luid| format!("{luid:016x}"))
                .unwrap_or_else(|| "unknown".to_string()),
            renderer
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter() -> GpuAdapter {
        GpuAdapter {
            device_id: 1,
            luid: 20,
            name: "GPU B".into(),
            dedicated_video_memory: 2,
            vendor_id: 0x10de,
            pci_device_id: 0x2c05,
            subsys_id: 0x18c21043,
            revision: 0xa1,
        }
    }

    #[test]
    fn resolves_stable_luid_to_current_directml_index() {
        let adapters = vec![
            GpuAdapter {
                device_id: 0,
                luid: 10,
                name: "A".into(),
                dedicated_video_memory: 1,
                vendor_id: 0,
                pci_device_id: 0,
                subsys_id: 0,
                revision: 0,
            },
            adapter(),
        ];
        assert_eq!(device_id_for_luid(&adapters, Some(20)), Some(1));
        assert_eq!(device_id_for_luid(&adapters, Some(30)), None);
        assert_eq!(device_id_for_luid(&adapters, None), None);
    }

    #[test]
    fn specific_gpu_id_uses_stable_pci_identity_not_dxgi_index() {
        assert_eq!(specific_adapter_id(&adapter()), "10DE&2C05&18C21043");
        assert_eq!(
            specific_gpu_preference(&adapter()),
            "SpecificAdapter=10DE&2C05&18C21043;GpuPreference=1073741824;"
        );
    }

    #[test]
    fn duplicate_luids_are_collapsed_but_real_same_name_gpus_remain() {
        let a = GpuAdapter {
            device_id: 0,
            luid: 0x1111,
            name: "NVIDIA GeForce RTX 4070 Ti".into(),
            dedicated_video_memory: 12,
            vendor_id: 0x10de,
            pci_device_id: 1,
            subsys_id: 2,
            revision: 0,
        };
        let duplicate = GpuAdapter {
            device_id: 2,
            ..a.clone()
        };
        let b = GpuAdapter {
            device_id: 3,
            luid: 0x2222,
            ..a.clone()
        };
        let adapters = normalize_adapters(vec![a.clone(), duplicate, b.clone()], false);
        assert_eq!(adapters.len(), 2);
        assert_eq!(adapters[0].luid, a.luid);
        assert_eq!(adapters[0].device_id, 0);
        assert_eq!(adapters[1].luid, b.luid);
        assert_eq!(
            adapter_display_name(&adapters, &adapters[0]),
            "NVIDIA GeForce RTX 4070 Ti (GPU 1)"
        );
        assert_eq!(
            adapter_display_name(&adapters, &adapters[1]),
            "NVIDIA GeForce RTX 4070 Ti (GPU 2)"
        );
    }
}
