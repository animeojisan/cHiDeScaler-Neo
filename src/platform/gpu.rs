//! DXGI adapter discovery used by the portable GPU selector.

use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuAdapter {
    /// DirectML's device id follows IDXGIFactory1::EnumAdapters1 order.
    pub device_id: i32,
    pub luid: u64,
    pub name: String,
    pub dedicated_video_memory: u64,
    pub vendor_id: u32,
}

fn luid_u64(luid: windows::Win32::Foundation::LUID) -> u64 {
    ((luid.HighPart as u32 as u64) << 32) | luid.LowPart as u64
}

fn wide_name(value: &[u16]) -> String {
    let len = value.iter().position(|c| *c == 0).unwrap_or(value.len());
    String::from_utf16_lossy(&value[..len]).trim().to_string()
}

pub fn enumerate_adapters() -> Vec<GpuAdapter> {
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else {
        log::warn!("gpu-enumeration: CreateDXGIFactory1 failed");
        return Vec::new();
    };
    let mut adapters = Vec::new();
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
        adapters.push(GpuAdapter {
            device_id: index as i32,
            luid: luid_u64(desc.AdapterLuid),
            name: wide_name(&desc.Description),
            dedicated_video_memory: desc.DedicatedVideoMemory as u64,
            vendor_id: desc.VendorId,
        });
    }
    for adapter in &adapters {
        log::info!(
            "gpu-detected: dml_device_id={} luid={:016x} vram_mb={} name={}",
            adapter.device_id,
            adapter.luid,
            adapter.dedicated_video_memory / (1024 * 1024),
            adapter.name
        );
    }
    adapters
}

pub fn device_id_for_luid(adapters: &[GpuAdapter], luid: Option<u64>) -> Option<i32> {
    luid.and_then(|wanted| {
        adapters
            .iter()
            .find(|adapter| adapter.luid == wanted)
            .map(|adapter| adapter.device_id)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_stable_luid_to_current_directml_index() {
        let adapters = vec![
            GpuAdapter {
                device_id: 0,
                luid: 10,
                name: "A".into(),
                dedicated_video_memory: 1,
                vendor_id: 0,
            },
            GpuAdapter {
                device_id: 1,
                luid: 20,
                name: "B".into(),
                dedicated_video_memory: 2,
                vendor_id: 0,
            },
        ];
        assert_eq!(device_id_for_luid(&adapters, Some(20)), Some(1));
        assert_eq!(device_id_for_luid(&adapters, Some(30)), None);
        assert_eq!(device_id_for_luid(&adapters, None), None);
    }
}
