//! Optional CUDA/D3D12/OpenGL shared-memory support for TensorRT.
//!
//! CUDA is resolved dynamically from the optional backend pack. The normal
//! DirectML-only package therefore has no CUDA loader dependency.

use anyhow::{Result, anyhow};
use std::ffi::{CString, c_void};
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use windows::Win32::{
    Foundation::{CloseHandle, HANDLE, HMODULE},
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
    System::LibraryLoader::{GetModuleHandleW, GetProcAddress},
};
use windows::core::{PCSTR, w};

const CUDA_SUCCESS: i32 = 0;
const CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE: i32 = 4;
const CUDA_EXTERNAL_MEMORY_DEDICATED: u32 = 1;

type CudaExternalMemory = *mut c_void;
type CudaSetDevice = unsafe extern "system" fn(i32) -> i32;
type CudaGetDeviceCount = unsafe extern "system" fn(*mut i32) -> i32;
type CudaDeviceGetLuid = unsafe extern "system" fn(*mut i8, *mut u32, i32) -> i32;
type CudaImportExternalMemory =
    unsafe extern "system" fn(*mut CudaExternalMemory, *const CudaExternalMemoryHandleDesc) -> i32;
type CudaExternalMemoryGetMappedBuffer = unsafe extern "system" fn(
    *mut *mut c_void,
    CudaExternalMemory,
    *const CudaExternalMemoryBufferDesc,
) -> i32;
type CudaDestroyExternalMemory = unsafe extern "system" fn(CudaExternalMemory) -> i32;
type CudaFree = unsafe extern "system" fn(*mut c_void) -> i32;
type CudaDeviceSynchronize = unsafe extern "system" fn() -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaWin32Handle {
    handle: *mut c_void,
    name: *const c_void,
}

#[repr(C)]
struct CudaExternalMemoryHandleDesc {
    handle_type: i32,
    handle: CudaWin32Handle,
    size: u64,
    flags: u32,
    reserved: [u32; 16],
}

#[repr(C)]
struct CudaExternalMemoryBufferDesc {
    offset: u64,
    size: u64,
    flags: u32,
    reserved: [u32; 16],
}

struct CudaApi {
    set_device: CudaSetDevice,
    get_device_count: CudaGetDeviceCount,
    device_get_luid: Option<CudaDeviceGetLuid>,
    import_external_memory: CudaImportExternalMemory,
    external_memory_get_mapped_buffer: CudaExternalMemoryGetMappedBuffer,
    destroy_external_memory: CudaDestroyExternalMemory,
    free: CudaFree,
    device_synchronize: CudaDeviceSynchronize,
}

unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

static CUDA_API: OnceLock<std::result::Result<CudaApi, String>> = OnceLock::new();
static CUDA_DEVICE_COUNT: OnceLock<std::result::Result<i32, String>> = OnceLock::new();
static CUDA_LUID_FALLBACK_LOGGED: AtomicBool = AtomicBool::new(false);

fn symbol<T: Copy>(module: HMODULE, name: &str) -> Result<T> {
    let name = CString::new(name)?;
    let raw = unsafe { GetProcAddress(module, PCSTR(name.as_ptr().cast())) }
        .ok_or_else(|| anyhow!("CUDA symbol {} is unavailable", name.to_string_lossy()))?;
    Ok(unsafe { std::mem::transmute_copy(&raw) })
}

fn optional_symbol<T: Copy>(module: HMODULE, name: &str) -> Option<T> {
    let name = CString::new(name).ok()?;
    let raw = unsafe { GetProcAddress(module, PCSTR(name.as_ptr().cast())) }?;
    Some(unsafe { std::mem::transmute_copy(&raw) })
}

fn cuda_api() -> Result<&'static CudaApi> {
    CUDA_API
        .get_or_init(|| {
            let module = unsafe { GetModuleHandleW(w!("cudart64_12.dll")) }
                .map_err(|error| format!("cudart64_12.dll is not loaded: {error}"))?;
            Ok(CudaApi {
                set_device: symbol(module, "cudaSetDevice").map_err(|e| e.to_string())?,
                get_device_count: symbol(module, "cudaGetDeviceCount")
                    .map_err(|e| e.to_string())?,
                device_get_luid: optional_symbol(module, "cudaDeviceGetLuid"),
                import_external_memory: symbol(module, "cudaImportExternalMemory")
                    .map_err(|e| e.to_string())?,
                external_memory_get_mapped_buffer: symbol(
                    module,
                    "cudaExternalMemoryGetMappedBuffer",
                )
                .map_err(|e| e.to_string())?,
                destroy_external_memory: symbol(module, "cudaDestroyExternalMemory")
                    .map_err(|e| e.to_string())?,
                free: symbol(module, "cudaFree").map_err(|e| e.to_string())?,
                device_synchronize: symbol(module, "cudaDeviceSynchronize")
                    .map_err(|e| e.to_string())?,
            })
        })
        .as_ref()
        .map_err(|error| anyhow!(error.clone()))
}

fn cuda_ok(status: i32, operation: &str) -> Result<()> {
    if status == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(anyhow!("{operation} failed with CUDA status {status}"))
    }
}

fn cuda_device_count_cached(api: &CudaApi) -> Result<i32> {
    match CUDA_DEVICE_COUNT.get_or_init(|| {
        let mut count = 0;
        let status = unsafe { (api.get_device_count)(&mut count) };
        if status == CUDA_SUCCESS {
            Ok(count)
        } else {
            Err(format!(
                "cudaGetDeviceCount failed with CUDA status {status}"
            ))
        }
    }) {
        Ok(count) => Ok(*count),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

static CUDA_SHARED_ACTIVE_BUFFERS: AtomicU64 = AtomicU64::new(0);
static CUDA_SHARED_ACTIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static CUDA_SHARED_TOTAL_CREATED: AtomicU64 = AtomicU64::new(0);
static CUDA_SHARED_TOTAL_FREED: AtomicU64 = AtomicU64::new(0);
static CUDA_SHARED_RELEASE_FAILURES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaSharedStats {
    pub active_buffers: u64,
    pub active_bytes: u64,
    pub total_created: u64,
    pub total_freed: u64,
    pub release_failures: u64,
}

pub fn cuda_shared_stats() -> CudaSharedStats {
    CudaSharedStats {
        active_buffers: CUDA_SHARED_ACTIVE_BUFFERS.load(Ordering::Relaxed),
        active_bytes: CUDA_SHARED_ACTIVE_BYTES.load(Ordering::Relaxed),
        total_created: CUDA_SHARED_TOTAL_CREATED.load(Ordering::Relaxed),
        total_freed: CUDA_SHARED_TOTAL_FREED.load(Ordering::Relaxed),
        release_failures: CUDA_SHARED_RELEASE_FAILURES.load(Ordering::Relaxed),
    }
}

fn luid_bytes(luid: windows::Win32::Foundation::LUID) -> [u8; 8] {
    let value = ((luid.HighPart as u32 as u64) << 32) | luid.LowPart as u64;
    value.to_le_bytes()
}

fn d3d12_device_for_luid(wanted: [u8; 8]) -> Result<ID3D12Device> {
    let factory = unsafe { CreateDXGIFactory1::<IDXGIFactory1>()? };
    for index in 0u32.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
            break;
        };
        let desc = unsafe { adapter.GetDesc1()? };
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
            || luid_bytes(desc.AdapterLuid) != wanted
        {
            continue;
        }
        let mut device = None;
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device)? };
        return device.ok_or_else(|| anyhow!("D3D12CreateDevice returned no device"));
    }
    Err(anyhow!("no DXGI adapter matched OpenGL LUID {wanted:02x?}"))
}

pub struct CudaSharedBuffer {
    pub key: u64,
    pub device_ptr: *mut c_void,
    pub byte_len: usize,
    pub allocation_byte_len: u64,
    pub luid: [u8; 8],
    device_id: i32,
    resource: ID3D12Resource,
    device: ID3D12Device,
    external_memory: CudaExternalMemory,
}

unsafe impl Send for CudaSharedBuffer {}

impl CudaSharedBuffer {
    pub fn new(key: u64, byte_len: usize, device_id: i32, gl_luid: [u8; 8]) -> Result<Self> {
        let api = cuda_api()?;
        unsafe {
            cuda_ok((api.set_device)(device_id), "cudaSetDevice")?;
        }
        if let Some(device_get_luid) = api.device_get_luid {
            let mut cuda_luid = [0i8; 8];
            let mut node_mask = 0u32;
            unsafe {
                cuda_ok(
                    device_get_luid(cuda_luid.as_mut_ptr(), &mut node_mask, device_id),
                    "cudaDeviceGetLuid",
                )?;
            }
            let cuda_luid = cuda_luid.map(|value| value as u8);
            if cuda_luid != gl_luid {
                return Err(anyhow!(
                    "gpu-luid-mismatch gl_luid={gl_luid:02x?} cuda_luid={cuda_luid:02x?}"
                ));
            }
        } else {
            let count = cuda_device_count_cached(api)?;
            anyhow::ensure!(
                count == 1 && device_id == 0,
                "cudaDeviceGetLuid unavailable with {count} CUDA devices"
            );
            if !CUDA_LUID_FALLBACK_LOGGED.swap(true, Ordering::AcqRel) {
                log::info!(
                    "trt-gpu-bridge: cudaDeviceGetLuid unavailable; using the single CUDA device selected by backend preflight (verified once)"
                );
            }
        }
        let device = d3d12_device_for_luid(gl_luid)?;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: byte_len as u64,
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
        let allocation = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
        anyhow::ensure!(
            allocation.SizeInBytes != 0 && allocation.SizeInBytes != u64::MAX,
            "invalid D3D12 shared allocation size"
        );
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut resource = None;
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )?;
        }
        let resource = resource.ok_or_else(|| anyhow!("shared D3D12 resource is null"))?;
        let cuda_handle = unsafe {
            device.CreateSharedHandle(
                &resource,
                None,
                0x1000_0000,
                windows::core::PCWSTR::null(),
            )?
        };
        let mut external_memory = std::ptr::null_mut();
        let import_desc = CudaExternalMemoryHandleDesc {
            handle_type: CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
            handle: CudaWin32Handle {
                handle: cuda_handle.0,
                name: std::ptr::null(),
            },
            size: allocation.SizeInBytes,
            flags: CUDA_EXTERNAL_MEMORY_DEDICATED,
            reserved: [0; 16],
        };
        let import_result =
            unsafe { (api.import_external_memory)(&mut external_memory, &import_desc) };
        unsafe {
            let _ = CloseHandle(cuda_handle);
        }
        cuda_ok(import_result, "cudaImportExternalMemory")?;
        let mut device_ptr = std::ptr::null_mut();
        let map_desc = CudaExternalMemoryBufferDesc {
            offset: 0,
            size: byte_len as u64,
            flags: 0,
            reserved: [0; 16],
        };
        if let Err(error) = unsafe {
            cuda_ok(
                (api.external_memory_get_mapped_buffer)(
                    &mut device_ptr,
                    external_memory,
                    &map_desc,
                ),
                "cudaExternalMemoryGetMappedBuffer",
            )
        } {
            unsafe {
                let _ = (api.destroy_external_memory)(external_memory);
            }
            return Err(error);
        }
        if device_ptr.is_null() {
            unsafe {
                let _ = (api.destroy_external_memory)(external_memory);
            }
            return Err(anyhow!("CUDA shared buffer pointer is null"));
        }
        CUDA_SHARED_ACTIVE_BUFFERS.fetch_add(1, Ordering::Relaxed);
        CUDA_SHARED_ACTIVE_BYTES.fetch_add(allocation.SizeInBytes, Ordering::Relaxed);
        CUDA_SHARED_TOTAL_CREATED.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            key,
            device_ptr,
            byte_len,
            allocation_byte_len: allocation.SizeInBytes,
            luid: gl_luid,
            device_id,
            resource,
            device,
            external_memory,
        })
    }

    pub fn shared_handle(&self) -> Result<HANDLE> {
        Ok(unsafe {
            self.device.CreateSharedHandle(
                &self.resource,
                None,
                0x1000_0000,
                windows::core::PCWSTR::null(),
            )?
        })
    }

    pub fn synchronize(device_id: i32) -> Result<()> {
        let api = cuda_api()?;
        unsafe {
            cuda_ok((api.set_device)(device_id), "cudaSetDevice")?;
            cuda_ok((api.device_synchronize)(), "cudaDeviceSynchronize")
        }
    }
}

impl Drop for CudaSharedBuffer {
    fn drop(&mut self) {
        if self.external_memory.is_null() {
            return;
        }
        let mut released = true;
        if let Ok(api) = cuda_api() {
            unsafe {
                let set_status = (api.set_device)(self.device_id);
                if set_status != CUDA_SUCCESS {
                    released = false;
                    log::error!(
                        "cuda-shared-release: key={} cudaSetDevice({}) failed status={}",
                        self.key,
                        self.device_id,
                        set_status
                    );
                }
                // cudaExternalMemoryGetMappedBuffer returns a CUDA allocation
                // that must be released with cudaFree before destroying the
                // external-memory object. Omitting this leaked one mapping per
                // bridge slot and eventually exhausted the CUDA address/VRAM
                // budget during repeated x2/x4/x5 or geometry rebuilds.
                if !self.device_ptr.is_null() {
                    let free_status = (api.free)(self.device_ptr);
                    if free_status != CUDA_SUCCESS {
                        released = false;
                        log::error!(
                            "cuda-shared-release: key={} cudaFree failed status={}",
                            self.key,
                            free_status
                        );
                    }
                }
                let destroy_status = (api.destroy_external_memory)(self.external_memory);
                if destroy_status != CUDA_SUCCESS {
                    released = false;
                    log::error!(
                        "cuda-shared-release: key={} cudaDestroyExternalMemory failed status={}",
                        self.key,
                        destroy_status
                    );
                }
            }
        } else {
            released = false;
            log::error!(
                "cuda-shared-release: key={} CUDA runtime unavailable during drop",
                self.key
            );
        }
        self.external_memory = std::ptr::null_mut();
        self.device_ptr = std::ptr::null_mut();
        if released {
            CUDA_SHARED_ACTIVE_BUFFERS.fetch_sub(1, Ordering::Relaxed);
            CUDA_SHARED_ACTIVE_BYTES.fetch_sub(self.allocation_byte_len, Ordering::Relaxed);
            CUDA_SHARED_TOTAL_FREED.fetch_add(1, Ordering::Relaxed);
        } else {
            // Keep the allocation in the diagnostic active total. The Rust
            // owner is gone and it cannot be retried, so hiding a failed native
            // release would make a genuine driver/runtime leak look clean.
            CUDA_SHARED_RELEASE_FAILURES.fetch_add(1, Ordering::Relaxed);
        }
    }
}
