//! Shared GL context wrapper: texture pool + CPU<->GPU transfer + program cache.
//!
//! Rendering invariants used by this implementation:
//! - Every per-frame texture is tracked and returned to a (w,h,comps,dtype)
//!   pool at end of frame (`release_frame`). Leaking textures exhausts VRAM
//!   within minutes (INCOMPLETE_ATTACHMENT, DWM corruption); create/destroy
//!   churn causes intermittent driver stalls (stutter). The pool fixes both.
//! - Intermediates are RGBA16F-family with NEAREST + clamp-to-edge (mpv hook
//!   shaders do integer-offset taps / depth-to-space; LINEAR breaks them).
//! - Capture frames upload as raw uint8 RGBA8 (normalized on sample); no CPU
//!   float conversion on the normal GPU-resident path.

use glow::HasContext;
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::rc::Rc;
use windows::Win32::Foundation::{CloseHandle, HANDLE};

pub const SWZ: [&str; 5] = ["", ".r", ".rg", ".rgb", ""];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dtype {
    U8,
    F16,
    U32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TexKey {
    pub w: i32,
    pub h: i32,
    pub d: i32,
    pub comps: u8,
    pub dtype: Dtype,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuTex {
    pub tex: glow::Texture,
    pub key: TexKey,
    /// mpv user-shader pixel phase waiting to be corrected by the next scale.
    /// This is metadata only; pooled storage identity remains TexKey-based.
    pub offset_x: f32,
    pub offset_y: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum InterpAuxPlane {
    Constant(f32),
    HorizontalGrid,
    VerticalGrid,
}

impl GpuTex {
    pub fn w(&self) -> i32 {
        self.key.w
    }
    pub fn h(&self) -> i32 {
        self.key.h
    }
    pub fn d(&self) -> i32 {
        self.key.d
    }
    pub fn target(&self) -> u32 {
        if self.key.d > 1 {
            glow::TEXTURE_3D
        } else {
            glow::TEXTURE_2D
        }
    }
    pub fn comps(&self) -> u8 {
        self.key.comps
    }
    pub fn offset(&self) -> (f32, f32) {
        (self.offset_x, self.offset_y)
    }
    pub fn has_offset(&self) -> bool {
        self.offset_x.abs() > 1.0e-6 || self.offset_y.abs() > 1.0e-6
    }
    pub fn with_offset(mut self, x: f32, y: f32) -> Self {
        self.offset_x = x;
        self.offset_y = y;
        self
    }
    pub fn clear_offset(mut self) -> Self {
        self.offset_x = 0.0;
        self.offset_y = 0.0;
        self
    }
}

#[derive(Default)]
pub struct ExternalNeoFlowState {
    pub shader_fingerprint: u64,
    pub size: Option<(i32, i32)>,
    pub mode_state: Option<GpuTex>,
    pub previous_stable_forward: Option<GpuTex>,
    pub previous_backward: Option<GpuTex>,
    pub previous_stable_risk: Option<GpuTex>,
    pub mode_history_valid: bool,
    pub flow_history_valid: bool,
}

impl ExternalNeoFlowState {
    fn textures(&self) -> impl Iterator<Item = GpuTex> + '_ {
        [
            self.mode_state,
            self.previous_stable_forward,
            self.previous_backward,
            self.previous_stable_risk,
        ]
        .into_iter()
        .flatten()
    }
}

fn formats(comps: u8, dtype: Dtype) -> (i32, u32, u32) {
    // (internal_format, format, type)
    let (internal, format) = match (comps, dtype) {
        (1, Dtype::U8) => (glow::R8, glow::RED),
        (2, Dtype::U8) => (glow::RG8, glow::RG),
        (3, Dtype::U8) => (glow::RGB8, glow::RGB),
        (4, Dtype::U8) => (glow::RGBA8, glow::RGBA),
        (1, Dtype::F16) => (glow::R16F, glow::RED),
        (2, Dtype::F16) => (glow::RG16F, glow::RG),
        (3, Dtype::F16) => (glow::RGB16F, glow::RGB),
        (1, Dtype::U32) => (glow::R32UI, glow::RED_INTEGER),
        _ => (glow::RGBA16F, glow::RGBA),
    };
    let ty = match dtype {
        Dtype::U8 => glow::UNSIGNED_BYTE,
        Dtype::F16 => glow::HALF_FLOAT,
        Dtype::U32 => glow::UNSIGNED_INT,
    };
    (internal as i32, format, ty)
}

pub struct GlContext {
    pub gl: Rc<glow::Context>,
    tracked: Vec<GpuTex>,
    pool: HashMap<TexKey, Vec<GpuTex>>,
    prog_cache: HashMap<String, glow::Program>,
    /// //!TEXTURE LUTs etc: uploaded once, keyed by shader path + name,
    /// OUTSIDE the tracked/pool lifecycle (their declared filter/wrap must
    /// survive; re-uploading megabytes per frame would be waste)
    persist: HashMap<String, GpuTex>,
    /// Dedicated RGBA8 capture-upload rotation for fullscreen monitor-region
    /// fallback. These textures intentionally live outside tracked/pool so the
    /// generic frame recycler cannot hand a just-presented surface back to the
    /// uploader on the next refresh. They are not shader //!TEXTUREs and thus
    /// still participate in the normal LINEAR/NEAREST source-filter dance.
    capture_upload_ring: Vec<GpuTex>,
    capture_upload_ring_key: Option<TexKey>,
    external_neoflow_state: ExternalNeoFlowState,
    fbo: glow::Framebuffer,
    pub quad_vao: glow::VertexArray,
    external_api: Option<ExternalMemoryApi>,
    external_imports: HashMap<u64, ExternalBufferImport>,
    // Imports whose GL read fence did not retire within the normal bounded
    // wait. Keep ownership and retry later instead of mem::forget leaking the
    // Win32 handle, GL memory object and D3D12 allocation permanently.
    external_retired: Vec<(u64, ExternalBufferImport)>,
    external_import_count: u64,
    // Some OpenGL drivers (notably AMD) expose the external-memory DSA entry
    // point but reject it, or require more conservative cross-API ownership
    // handoff. Keep this as a one-way safety latch for the current GL context.
    external_conservative_sync: bool,
    external_storage_fallbacks: u64,
    gl_vendor: String,
    // GL command fences used by the asynchronous interpolation handoff.
    // Tokens keep glow::Fence values on the render thread; worker threads
    // receive only shared D3D12/CUDA resources and never touch OpenGL.
    command_fences: HashMap<u64, glow::Fence>,
    next_command_fence: u64,
    gpu_timer_queries: VecDeque<(glow::Query, String)>,
    gpu_timer_last_sample: HashMap<String, std::time::Instant>,
}

#[derive(Clone, Copy)]
struct ExternalMemoryApi {
    create_memory_objects: unsafe extern "system" fn(i32, *mut u32),
    delete_memory_objects: unsafe extern "system" fn(i32, *const u32),
    memory_object_parameter_iv: unsafe extern "system" fn(u32, u32, *const i32),
    import_memory_handle: unsafe extern "system" fn(u32, u64, u32, *mut c_void),
    named_buffer_storage_mem: unsafe extern "system" fn(u32, isize, u32, u64),
    buffer_storage_mem: unsafe extern "system" fn(u32, isize, u32, u64),
    device_luid: [u8; 8],
}

struct ExternalBufferImport {
    buffer: glow::Buffer,
    memory: u32,
    handle: HANDLE,
    read_fence: Option<glow::Fence>,
    byte_len: usize,
    // True only when glNamedBufferStorageMemEXT was rejected and the driver
    // accepted the equivalent bound glBufferStorageMemEXT route.  Keep this
    // per import: v807 uses the actual problematic resource state rather than
    // GPU vendor or frame resolution to select the DRBA snapshot path.
    bound_storage_fallback: bool,
    // Lazily allocated ordinary GL buffer used only by v807's DRBA safe lane.
    // It is kept with the import to avoid per-frame buffer allocation churn.
    local_snapshot: Option<glow::Buffer>,
}

const GL_HANDLE_TYPE_D3D12_RESOURCE_EXT: u32 = 0x958A;
const GL_DEDICATED_MEMORY_OBJECT_EXT: u32 = 0x9581;
const GL_DEVICE_LUID_EXT: u32 = 0x9599;

impl GlContext {
    pub fn new(gl: Rc<glow::Context>) -> Self {
        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            let fbo = gl.create_framebuffer().expect("fbo");
            // fullscreen-quad VAO shared by every pass (locations fixed at 0/1)
            let vao = gl.create_vertex_array().expect("vao");
            let vbo = gl.create_buffer().expect("vbo");
            #[rustfmt::skip]
            let quad: [f32; 16] = [
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 1.0,
                 1.0,  1.0, 1.0, 1.0,
            ];
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck_cast_slice(&quad),
                glow::STATIC_DRAW,
            );
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 16, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 16, 8);
            gl.bind_vertex_array(None);
            let gl_vendor = gl.get_parameter_string(glow::VENDOR);
            let vendor_lower = gl_vendor.to_ascii_lowercase();
            let external_conservative_sync = vendor_lower.contains("amd")
                || vendor_lower.contains("ati technologies")
                || vendor_lower.contains("advanced micro devices");
            Self {
                gl,
                tracked: Vec::new(),
                pool: HashMap::new(),
                prog_cache: HashMap::new(),
                persist: HashMap::new(),
                capture_upload_ring: Vec::new(),
                capture_upload_ring_key: None,
                external_neoflow_state: ExternalNeoFlowState::default(),
                fbo,
                quad_vao: vao,
                external_api: None,
                external_imports: HashMap::new(),
                external_retired: Vec::new(),
                external_import_count: 0,
                external_conservative_sync,
                external_storage_fallbacks: 0,
                gl_vendor,
                command_fences: HashMap::new(),
                next_command_fence: 1,
                gpu_timer_queries: VecDeque::new(),
                gpu_timer_last_sample: HashMap::new(),
            }
        }
    }

    pub fn init_external_memory(
        &mut self,
        loader: impl Fn(&str) -> *const c_void,
    ) -> Result<(), String> {
        for required in ["GL_EXT_memory_object", "GL_EXT_memory_object_win32"] {
            if !self.gl.supported_extensions().contains(required) {
                return Err(format!("missing {required}"));
            }
        }
        let required = |name: &str| {
            let ptr = loader(name);
            if ptr.is_null() {
                Err(format!("missing OpenGL function {name}"))
            } else {
                Ok(ptr)
            }
        };
        unsafe {
            let get_unsigned_byte_v: unsafe extern "system" fn(u32, *mut u8) =
                std::mem::transmute(required("glGetUnsignedBytevEXT")?);
            let mut device_luid = [0u8; 8];
            get_unsigned_byte_v(GL_DEVICE_LUID_EXT, device_luid.as_mut_ptr());
            self.external_api = Some(ExternalMemoryApi {
                create_memory_objects: std::mem::transmute(required("glCreateMemoryObjectsEXT")?),
                delete_memory_objects: std::mem::transmute(required("glDeleteMemoryObjectsEXT")?),
                memory_object_parameter_iv: std::mem::transmute(required(
                    "glMemoryObjectParameterivEXT",
                )?),
                import_memory_handle: std::mem::transmute(required(
                    "glImportMemoryWin32HandleEXT",
                )?),
                named_buffer_storage_mem: std::mem::transmute(required(
                    "glNamedBufferStorageMemEXT",
                )?),
                buffer_storage_mem: std::mem::transmute(required("glBufferStorageMemEXT")?),
                device_luid,
            });
        }
        Ok(())
    }

    pub fn external_device_luid(&self) -> Option<[u8; 8]> {
        self.external_api.map(|api| api.device_luid)
    }

    pub fn external_import_key(&self) -> Option<u64> {
        self.external_imports.keys().next().copied()
    }

    pub fn has_external_import(&self, key: u64) -> bool {
        self.external_imports.contains_key(&key)
    }

    pub fn external_import_count(&self) -> u64 {
        self.external_import_count
    }

    /// True when the current OpenGL driver should use a conservative
    /// OpenGL<->D3D12 handoff. This is deliberately sticky: once the driver
    /// rejects the DSA external-buffer binding, do not retry the aggressive
    /// path later in the same process.
    pub fn external_interop_conservative_recommended(&self) -> bool {
        self.external_conservative_sync
    }

    pub fn external_storage_fallback_count(&self) -> u64 {
        self.external_storage_fallbacks
    }

    /// Whether this exact imported D3D12 buffer had to use the bound-storage
    /// fallback.  Unlike the process-wide conservative recommendation this is
    /// resource-specific, so unrelated NVIDIA/AMD paths are not changed.
    pub fn external_import_uses_bound_storage_fallback(&self, key: u64) -> bool {
        self.external_imports
            .get(&key)
            .is_some_and(|item| item.bound_storage_fallback)
    }

    pub fn gl_vendor(&self) -> &str {
        &self.gl_vendor
    }

    pub fn external_import_active_count(&self) -> usize {
        self.external_imports.len() + self.external_retired.len()
    }

    pub fn external_import_active_bytes(&self) -> usize {
        self.external_imports
            .values()
            .map(|item| item.byte_len)
            .chain(self.external_retired.iter().map(|(_, item)| item.byte_len))
            .sum()
    }

    pub fn external_import_retired_count(&self) -> usize {
        self.external_retired.len()
    }

    /// Import an app-owned shared D3D12 heap containing a buffer at offset 0.
    /// Takes ownership of `handle`.
    pub fn import_external_d3d12_buffer(
        &mut self,
        key: u64,
        handle: HANDLE,
        byte_len: usize,
        allocation_byte_len: u64,
        device_luid: [u8; 8],
    ) -> Result<(), String> {
        self.reap_external_buffer_retirement(false);
        let Some(api) = self.external_api else {
            unsafe {
                let _ = CloseHandle(handle);
            }
            return Err("OpenGL external memory is unavailable".into());
        };
        if api.device_luid != device_luid {
            unsafe {
                let _ = CloseHandle(handle);
            }
            return Err(format!(
                "DirectML/OpenGL GPU mismatch: DML={device_luid:02x?}, GL={:02x?}",
                api.device_luid
            ));
        }
        if self.external_imports.contains_key(&key) {
            unsafe {
                let _ = CloseHandle(handle);
            }
            return Ok(());
        }
        unsafe {
            let mut failures = Vec::new();
            for (handle_type, import_byte_len, dedicated, route) in [
                // EXT_external_objects_win32 requires a D3D12 resource handle
                // to use this type. Its import size is ignored, and revision 10
                // recommends zero because some Windows drivers require it.
                (GL_HANDLE_TYPE_D3D12_RESOURCE_EXT, 0, true, "d3d12-resource"),
            ] {
                // Do not attribute an older GL error to this optional bridge.
                while self.gl.get_error() != glow::NO_ERROR {}

                let mut memory = 0u32;
                (api.create_memory_objects)(1, &mut memory);
                let error = self.gl.get_error();
                if error != glow::NO_ERROR || memory == 0 {
                    failures.push(format!("{route}:create=0x{error:04x}"));
                    continue;
                }

                if dedicated {
                    let value = 1i32;
                    (api.memory_object_parameter_iv)(
                        memory,
                        GL_DEDICATED_MEMORY_OBJECT_EXT,
                        &value,
                    );
                    let error = self.gl.get_error();
                    if error != glow::NO_ERROR {
                        (api.delete_memory_objects)(1, &memory);
                        failures.push(format!("{route}:dedicated=0x{error:04x}"));
                        continue;
                    }
                }

                (api.import_memory_handle)(memory, import_byte_len, handle_type, handle.0);
                let error = self.gl.get_error();
                if error != glow::NO_ERROR {
                    (api.delete_memory_objects)(1, &memory);
                    failures.push(format!("{route}:import=0x{error:04x}"));
                    continue;
                }

                let buffer = match self.gl.create_buffer() {
                    Ok(buffer) => buffer,
                    Err(error) => {
                        (api.delete_memory_objects)(1, &memory);
                        failures.push(format!("{route}:buffer={error}"));
                        continue;
                    }
                };
                let mut used_bound_storage_fallback = false;
                (api.named_buffer_storage_mem)(buffer.0.get(), byte_len as isize, memory, 0);
                let named_error = self.gl.get_error();
                if named_error != glow::NO_ERROR {
                    // Some AMD OpenGL drivers expose the DSA entry point but
                    // reject external buffer storage through it. The bound
                    // form is equivalent and avoids that driver path.
                    while self.gl.get_error() != glow::NO_ERROR {}
                    self.gl
                        .bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(buffer));
                    (api.buffer_storage_mem)(
                        glow::SHADER_STORAGE_BUFFER,
                        byte_len as isize,
                        memory,
                        0,
                    );
                    self.gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);
                    let bound_error = self.gl.get_error();
                    if bound_error != glow::NO_ERROR {
                        self.gl.delete_buffer(buffer);
                        (api.delete_memory_objects)(1, &memory);
                        failures.push(format!(
                            "{route}:storage=named:0x{named_error:04x},bound:0x{bound_error:04x}"
                        ));
                        continue;
                    }
                    used_bound_storage_fallback = true;
                    self.external_conservative_sync = true;
                    self.external_storage_fallbacks =
                        self.external_storage_fallbacks.saturating_add(1);
                    log::info!(
                        "OpenGL external buffer used bound storage fallback: route={route} named_error=0x{named_error:04x}; conservative_sync=enabled"
                    );
                }
                self.external_imports.insert(
                    key,
                    ExternalBufferImport {
                        buffer,
                        memory,
                        handle,
                        read_fence: None,
                        byte_len,
                        bound_storage_fallback: used_bound_storage_fallback,
                        local_snapshot: None,
                    },
                );
                self.external_import_count = self.external_import_count.saturating_add(1);
                log::info!(
                    "OpenGL D3D12 shared buffer imported: route={route} key={key} data_bytes={byte_len} allocation_bytes={allocation_byte_len} LUID={device_luid:02x?}"
                );
                return Ok(());
            }
            let _ = CloseHandle(handle);
            return Err(format!(
                "OpenGL D3D12 shared import failed: {}",
                failures.join(" | ")
            ));
        }
    }

    /// Wait only for the previous shared-buffer conversion, not the complete
    /// render frame, before DirectML overwrites the buffer.
    pub fn wait_external_buffer_idle(&mut self, key: u64) -> Result<(), String> {
        let Some(import) = self.external_imports.get_mut(&key) else {
            return Ok(());
        };
        let Some(fence) = import.read_fence.take() else {
            return Ok(());
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
        unsafe {
            loop {
                let status =
                    self.gl
                        .client_wait_sync(fence, glow::SYNC_FLUSH_COMMANDS_BIT, 5_000_000);
                if status == glow::ALREADY_SIGNALED || status == glow::CONDITION_SATISFIED {
                    break;
                }
                if status == glow::WAIT_FAILED {
                    self.gl.delete_sync(fence);
                    return Err("OpenGL shared-buffer fence failed".into());
                }
                if std::time::Instant::now() >= deadline {
                    self.gl.delete_sync(fence);
                    return Err("OpenGL shared-buffer fence timed out after 100ms".into());
                }
            }
            self.gl.delete_sync(fence);
        }
        Ok(())
    }

    /// Convert an imported NCHW FP16 buffer to a regular pooled RGBA8 texture.
    pub fn external_nchw_f16_to_rgba8(
        &mut self,
        key: u64,
        width: i32,
        height: i32,
    ) -> Result<GpuTex, String> {
        self.external_nchw_f16_to_rgba8_crop(key, width, height, width, height)
    }

    pub fn external_nchw_f16_to_rgba8_crop(
        &mut self,
        key: u64,
        width: i32,
        height: i32,
        plane_width: i32,
        plane_height: i32,
    ) -> Result<GpuTex, String> {
        let buffer = self
            .external_imports
            .get(&key)
            .map(|item| item.buffer)
            .ok_or_else(|| "shared D3D12 buffer is not imported".to_string())?;
        self.nchw_f16_buffer_to_rgba8_crop(
            buffer,
            Some(key),
            width,
            height,
            plane_width,
            plane_height,
        )
    }

    /// v807 DirectML-DRBA safety lane.  Copy the imported external NCHW tensor
    /// into an ordinary OpenGL-owned buffer before the conversion shader reads
    /// it. This deliberately avoids CPU readback/upload and is selected only
    /// for DRBA output resources whose import actually used the AMD-style
    /// bound-storage fallback. A single glFinish at the end completes both the
    /// snapshot read and RGBA imageStore before the provider may recycle the
    /// shared allocation.
    pub fn external_nchw_f16_to_rgba8_crop_local_snapshot(
        &mut self,
        key: u64,
        width: i32,
        height: i32,
        plane_width: i32,
        plane_height: i32,
    ) -> Result<GpuTex, String> {
        let (source, imported_bytes, existing_snapshot) = self
            .external_imports
            .get(&key)
            .map(|item| (item.buffer, item.byte_len, item.local_snapshot))
            .ok_or_else(|| "shared D3D12 buffer is not imported".to_string())?;
        let plane = usize::try_from(plane_width)
            .ok()
            .and_then(|w| usize::try_from(plane_height).ok().and_then(|h| w.checked_mul(h)))
            .ok_or_else(|| "invalid shared output plane dimensions".to_string())?;
        let snapshot_bytes = plane
            .checked_mul(3)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<half::f16>()))
            .ok_or_else(|| "shared output snapshot size overflow".to_string())?;
        if snapshot_bytes > imported_bytes {
            return Err(format!(
                "shared output snapshot exceeds imported buffer: required={} imported={}",
                snapshot_bytes, imported_bytes
            ));
        }
        let snapshot_size = i32::try_from(snapshot_bytes)
            .map_err(|_| "shared output snapshot exceeds OpenGL buffer size range".to_string())?;

        let snapshot = if let Some(snapshot) = existing_snapshot {
            snapshot
        } else {
            let snapshot = unsafe { self.gl.create_buffer().map_err(|error| error.to_string())? };
            unsafe {
                self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(snapshot));
                self.gl.buffer_data_size(
                    glow::COPY_WRITE_BUFFER,
                    snapshot_size,
                    glow::STREAM_COPY,
                );
                self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            }
            if let Some(import) = self.external_imports.get_mut(&key) {
                import.local_snapshot = Some(snapshot);
            } else {
                unsafe {
                    self.gl.delete_buffer(snapshot);
                }
                return Err("shared D3D12 buffer disappeared during snapshot allocation".into());
            }
            snapshot
        };

        unsafe {
            // D3D12 completion is already established by the provider-copy
            // fence. Publish the external consumer boundary, then force an
            // actual GL buffer-copy read into driver-local storage before any
            // shader samples the tensor.
            self.gl.memory_barrier(glow::ALL_BARRIER_BITS);
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(source));
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(snapshot));
            self.gl.copy_buffer_sub_data(
                glow::COPY_READ_BUFFER,
                glow::COPY_WRITE_BUFFER,
                0,
                0,
                snapshot_size,
            );
            self.gl.memory_barrier(
                glow::BUFFER_UPDATE_BARRIER_BIT | glow::SHADER_STORAGE_BARRIER_BIT,
            );
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, None);
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
        }

        let conversion = self.nchw_f16_buffer_to_rgba8_crop(
            snapshot,
            None,
            width,
            height,
            plane_width,
            plane_height,
        );
        unsafe {
            // One completion point for this narrow safe lane. This makes the
            // local snapshot self-contained; no external read fence is needed
            // and the provider may reuse its shared output after return.
            self.gl.finish();
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, None);
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
        }
        conversion
    }

    fn nchw_f16_buffer_to_rgba8_crop(
        &mut self,
        buffer: glow::Buffer,
        external_read_key: Option<u64>,
        width: i32,
        height: i32,
        plane_width: i32,
        plane_height: i32,
    ) -> Result<GpuTex, String> {
        let dest = self.make_tex(width, height, 4, Dtype::U8);
        let visible_pixels = usize::try_from(width)
            .ok()
            .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
            .ok_or_else(|| "invalid shared output dimensions".to_string())?;
        let fast_pair_unpack = width == plane_width
            && height == plane_height
            && visible_pixels % 2 == 0;
        let program = if fast_pair_unpack {
            self.compute_program(EXTERNAL_NCHW_F16_TO_RGBA8_PAIR_PIXELS)?
        } else {
            self.compute_program(EXTERNAL_NCHW_F16_TO_RGBA8_LEGACY)?
        };
        unsafe {
            // For imported buffers this is the D3D12 -> GL consumer boundary;
            // for v807 local snapshots it publishes glCopyBufferSubData before
            // the SSBO shader read. Both remain GPU-only.
            self.gl.memory_barrier(glow::ALL_BARRIER_BITS);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            self.gl.bind_image_texture(
                0,
                Some(dest.tex),
                0,
                false,
                0,
                glow::WRITE_ONLY,
                glow::RGBA8,
            );
            if let Some(loc) = self.gl.get_uniform_location(program, "width") {
                self.gl.uniform_1_i32(Some(&loc), width);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "height") {
                self.gl.uniform_1_i32(Some(&loc), height);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "plane_width") {
                self.gl.uniform_1_i32(Some(&loc), plane_width);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "plane_height") {
                self.gl.uniform_1_i32(Some(&loc), plane_height);
            }
            if fast_pair_unpack {
                if let Some(loc) = self.gl.get_uniform_location(program, "pixel_count") {
                    self.gl.uniform_1_u32(Some(&loc), visible_pixels as u32);
                }
                self.gl
                    .dispatch_compute(((visible_pixels / 2) as u32).div_ceil(256), 1, 1);
            } else {
                self.gl.dispatch_compute(
                    (width as u32).div_ceil(16),
                    (height as u32).div_ceil(16),
                    1,
                );
            }
            self.gl.memory_barrier(
                glow::SHADER_IMAGE_ACCESS_BARRIER_BIT | glow::TEXTURE_FETCH_BARRIER_BIT,
            );
            if let Some(key) = external_read_key {
                let fence = self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)?;
                if let Some(import) = self.external_imports.get_mut(&key) {
                    if let Some(old) = import.read_fence.replace(fence) {
                        self.gl.delete_sync(old);
                    }
                }
            }
        }
        Ok(dest)
    }

    /// Pack a regular RGBA texture into an imported shared D3D12 RGBA8 buffer.
    /// v743 uses this before the isolated DLSSNR worker consumes the same buffer.
    /// `glFinish` is intentionally retained as the conservative cross-API ownership
    /// boundary; the full-frame CPU readback is still eliminated.
    pub fn rgba_texture_to_external_rgba8(
        &mut self,
        key: u64,
        source: GpuTex,
        width: i32,
        height: i32,
    ) -> Result<(), String> {
        let expected = usize::try_from(width)
            .ok()
            .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "external RGBA8 size overflow".to_string())?;
        let (buffer, byte_len) = self
            .external_imports
            .get(&key)
            .map(|item| (item.buffer, item.byte_len))
            .ok_or_else(|| "shared D3D12 RGBA8 buffer is not imported".to_string())?;
        if byte_len < expected {
            return Err(format!("shared D3D12 RGBA8 buffer is too small: {byte_len} < {expected}"));
        }
        let program = self.compute_program(RGBA_TEXTURE_TO_EXTERNAL_RGBA8)?;
        unsafe {
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(source.target(), Some(source.tex));
            if let Some(loc) = self.gl.get_uniform_location(program, "source_tex") {
                self.gl.uniform_1_i32(Some(&loc), 0);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "width") {
                self.gl.uniform_1_i32(Some(&loc), width);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "height") {
                self.gl.uniform_1_i32(Some(&loc), height);
            }
            self.gl.dispatch_compute(
                (width as u32).div_ceil(16),
                (height as u32).div_ceil(8),
                1,
            );
            self.gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
            self.gl.finish();
            self.gl.bind_texture(source.target(), None);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
        }
        Ok(())
    }

    /// Convert shared RGBA8 output back to a pooled texture while preserving the
    /// source alpha on GPU. Feature 18 is an RGB enhancement and the legacy CPU
    /// path restored alpha after readback; this shader keeps the same contract.
    pub fn external_rgba8_buffer_to_texture_with_source_alpha(
        &mut self,
        key: u64,
        source_alpha: GpuTex,
        width: i32,
        height: i32,
    ) -> Result<GpuTex, String> {
        let expected = usize::try_from(width)
            .ok()
            .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "external RGBA8 size overflow".to_string())?;
        let (buffer, byte_len) = self
            .external_imports
            .get(&key)
            .map(|item| (item.buffer, item.byte_len))
            .ok_or_else(|| "shared D3D12 RGBA8 buffer is not imported".to_string())?;
        if byte_len < expected {
            return Err(format!("shared D3D12 RGBA8 buffer is too small: {byte_len} < {expected}"));
        }
        let dest = self.make_tex(width, height, 4, Dtype::U8);
        let program = self.compute_program(EXTERNAL_RGBA8_BUFFER_TO_TEXTURE_ALPHA)?;
        unsafe {
            self.gl.memory_barrier(glow::ALL_BARRIER_BITS);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(source_alpha.target(), Some(source_alpha.tex));
            if let Some(loc) = self.gl.get_uniform_location(program, "alpha_tex") {
                self.gl.uniform_1_i32(Some(&loc), 0);
            }
            self.gl.bind_image_texture(
                0,
                Some(dest.tex),
                0,
                false,
                0,
                glow::WRITE_ONLY,
                glow::RGBA8,
            );
            if let Some(loc) = self.gl.get_uniform_location(program, "width") {
                self.gl.uniform_1_i32(Some(&loc), width);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "height") {
                self.gl.uniform_1_i32(Some(&loc), height);
            }
            self.gl.dispatch_compute(
                (width as u32).div_ceil(16),
                (height as u32).div_ceil(8),
                1,
            );
            self.gl.memory_barrier(
                glow::SHADER_IMAGE_ACCESS_BARRIER_BIT | glow::TEXTURE_FETCH_BARRIER_BIT,
            );
            self.gl.bind_texture(source_alpha.target(), None);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
            let fence = self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)?;
            if let Some(import) = self.external_imports.get_mut(&key) {
                if let Some(old) = import.read_fence.replace(fence) {
                    self.gl.delete_sync(old);
                }
            }
        }
        Ok(dest)
    }

    /// Convert a packed RGBA8 buffer imported through GL_EXT_memory_object into
    /// a regular pooled RGBA8 texture without staging the frame through CPU memory.
    /// The producer (Vulkan) is synchronized by its queue fence before this call;
    /// the GL fence recorded here protects the shared buffer from being overwritten
    /// until this compute read has retired.
    pub fn external_rgba8_buffer_to_texture(
        &mut self,
        key: u64,
        width: i32,
        height: i32,
    ) -> Result<GpuTex, String> {
        let expected = usize::try_from(width)
            .ok()
            .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "external RGBA8 size overflow".to_string())?;
        let (buffer, byte_len) = self
            .external_imports
            .get(&key)
            .map(|item| (item.buffer, item.byte_len))
            .ok_or_else(|| "shared Vulkan RGBA8 buffer is not imported".to_string())?;
        if byte_len < expected {
            return Err(format!(
                "shared Vulkan RGBA8 buffer is too small: {} < {}",
                byte_len, expected
            ));
        }
        let dest = self.make_tex(width, height, 4, Dtype::U8);
        let program = self.compute_program(EXTERNAL_RGBA8_BUFFER_TO_TEXTURE)?;
        unsafe {
            self.gl.memory_barrier(glow::ALL_BARRIER_BITS);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            self.gl.bind_image_texture(
                0,
                Some(dest.tex),
                0,
                false,
                0,
                glow::WRITE_ONLY,
                glow::RGBA8,
            );
            if let Some(loc) = self.gl.get_uniform_location(program, "width") {
                self.gl.uniform_1_i32(Some(&loc), width);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "height") {
                self.gl.uniform_1_i32(Some(&loc), height);
            }
            self.gl
                .dispatch_compute((width as u32).div_ceil(16), (height as u32).div_ceil(8), 1);
            self.gl.memory_barrier(
                glow::SHADER_IMAGE_ACCESS_BARRIER_BIT | glow::TEXTURE_FETCH_BARRIER_BIT,
            );
            let fence = self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)?;
            if let Some(import) = self.external_imports.get_mut(&key) {
                if let Some(old) = import.read_fence.replace(fence) {
                    self.gl.delete_sync(old);
                }
            }
        }
        Ok(dest)
    }

    pub fn external_nchw_f32_to_rgba8_crop(
        &mut self,
        key: u64,
        width: i32,
        height: i32,
        plane_width: i32,
        plane_height: i32,
    ) -> Result<GpuTex, String> {
        let buffer = self
            .external_imports
            .get(&key)
            .map(|v| v.buffer)
            .ok_or_else(|| "shared D3D12 buffer is not imported".to_string())?;
        let dest = self.make_tex(width, height, 4, Dtype::U8);
        let program = self.compute_program(EXTERNAL_NCHW_F32_TO_RGBA8)?;
        unsafe {
            self.gl.memory_barrier(glow::ALL_BARRIER_BITS);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            self.gl.bind_image_texture(
                0,
                Some(dest.tex),
                0,
                false,
                0,
                glow::WRITE_ONLY,
                glow::RGBA8,
            );
            for (n, v) in [
                ("width", width),
                ("height", height),
                ("plane_width", plane_width),
                ("plane_height", plane_height),
            ] {
                if let Some(l) = self.gl.get_uniform_location(program, n) {
                    self.gl.uniform_1_i32(Some(&l), v);
                }
            }
            self.gl
                .dispatch_compute((width as u32).div_ceil(16), (height as u32).div_ceil(16), 1);
            // The conversion compute shader writes `dest` with imageStore,
            // while the very next operation is commonly an mpv/Anime4K GLSL
            // pass that reads `dest` through a sampler. IMAGE_ACCESS alone does
            // not guarantee visibility to texture fetches; include the texture
            // fetch barrier so interpolation -> GLSL never samples a stale
            // pre-conversion image on aggressive/asynchronous drivers.
            self.gl.memory_barrier(
                glow::SHADER_IMAGE_ACCESS_BARRIER_BIT | glow::TEXTURE_FETCH_BARRIER_BIT,
            );
            let fence = self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)?;
            if let Some(import) = self.external_imports.get_mut(&key) {
                if let Some(old) = import.read_fence.replace(fence) {
                    self.gl.delete_sync(old);
                }
            }
        }
        Ok(dest)
    }

    /// Convert imported interpolation output to a regular RGBA16F texture.
    /// Used for DirectML -> post-GLSL handoff so an FP16 RIFE/DRBA result is
    /// not quantized to 8-bit before a CNN upscaler/restorer consumes it.
    pub fn external_nchw_f16_to_rgba16f_crop(
        &mut self,
        key: u64,
        width: i32,
        height: i32,
        plane_width: i32,
        plane_height: i32,
    ) -> Result<GpuTex, String> {
        let buffer = self
            .external_imports
            .get(&key)
            .map(|item| item.buffer)
            .ok_or_else(|| "shared D3D12 buffer is not imported".to_string())?;
        let dest = self.make_tex(width, height, 4, Dtype::F16);
        let program = self.compute_program(EXTERNAL_NCHW_F16_TO_RGBA16F)?;
        unsafe {
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            self.gl.bind_image_texture(
                0,
                Some(dest.tex),
                0,
                false,
                0,
                glow::WRITE_ONLY,
                glow::RGBA16F,
            );
            for (name, value) in [
                ("width", width),
                ("height", height),
                ("plane_width", plane_width),
                ("plane_height", plane_height),
            ] {
                if let Some(loc) = self.gl.get_uniform_location(program, name) {
                    self.gl.uniform_1_i32(Some(&loc), value);
                }
            }
            self.gl
                .dispatch_compute((width as u32).div_ceil(16), (height as u32).div_ceil(16), 1);
            self.gl.memory_barrier(
                glow::SHADER_IMAGE_ACCESS_BARRIER_BIT | glow::TEXTURE_FETCH_BARRIER_BIT,
            );
            let fence = self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)?;
            if let Some(import) = self.external_imports.get_mut(&key) {
                if let Some(old) = import.read_fence.replace(fence) {
                    self.gl.delete_sync(old);
                }
            }
        }
        Ok(dest)
    }

    pub fn external_nchw_f32_to_rgba16f_crop(
        &mut self,
        key: u64,
        width: i32,
        height: i32,
        plane_width: i32,
        plane_height: i32,
    ) -> Result<GpuTex, String> {
        let buffer = self
            .external_imports
            .get(&key)
            .map(|item| item.buffer)
            .ok_or_else(|| "shared D3D12 buffer is not imported".to_string())?;
        let dest = self.make_tex(width, height, 4, Dtype::F16);
        let program = self.compute_program(EXTERNAL_NCHW_F32_TO_RGBA16F)?;
        unsafe {
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            self.gl.bind_image_texture(
                0,
                Some(dest.tex),
                0,
                false,
                0,
                glow::WRITE_ONLY,
                glow::RGBA16F,
            );
            for (name, value) in [
                ("width", width),
                ("height", height),
                ("plane_width", plane_width),
                ("plane_height", plane_height),
            ] {
                if let Some(loc) = self.gl.get_uniform_location(program, name) {
                    self.gl.uniform_1_i32(Some(&loc), value);
                }
            }
            self.gl
                .dispatch_compute((width as u32).div_ceil(16), (height as u32).div_ceil(16), 1);
            self.gl.memory_barrier(
                glow::SHADER_IMAGE_ACCESS_BARRIER_BIT | glow::TEXTURE_FETCH_BARRIER_BIT,
            );
            let fence = self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)?;
            if let Some(import) = self.external_imports.get_mut(&key) {
                if let Some(old) = import.read_fence.replace(fence) {
                    self.gl.delete_sync(old);
                }
            }
        }
        Ok(dest)
    }

    /// Pack a regular RGBA texture directly into an imported planar NCHW FP16
    /// buffer. Stage-A synchronization intentionally uses glFinish so DML
    /// never observes a partially written buffer; this can later be replaced
    /// by an external semaphore without changing the chain contract.
    pub fn rgba_texture_to_external_nchw_f16(
        &mut self,
        key: u64,
        source: GpuTex,
        width: i32,
        height: i32,
    ) -> Result<(), String> {
        let buffer = self
            .external_imports
            .get(&key)
            .map(|item| item.buffer)
            .ok_or_else(|| "shared D3D12 input buffer is not imported".to_string())?;
        let pixels = usize::try_from(width)
            .ok()
            .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
            .ok_or_else(|| "invalid shared input dimensions".to_string())?;
        let fast_pair_pack = pixels % 2 == 0;
        let program = if fast_pair_pack {
            self.compute_program(RGBA_TEXTURE_TO_EXTERNAL_NCHW_FP16_PAIR_PIXELS)?
        } else {
            self.compute_program(RGBA_TEXTURE_TO_EXTERNAL_NCHW_FP16_LEGACY)?
        };
        let elements = pixels
            .checked_mul(3)
            .ok_or_else(|| "shared input element count overflow".to_string())?;
        let work_items = if fast_pair_pack { pixels / 2 } else { elements.div_ceil(2) };
        unsafe {
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(source.target(), Some(source.tex));
            if let Some(loc) = self.gl.get_uniform_location(program, "source_tex") {
                self.gl.uniform_1_i32(Some(&loc), 0);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "width") {
                self.gl.uniform_1_i32(Some(&loc), width);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "height") {
                self.gl.uniform_1_i32(Some(&loc), height);
            }
            if fast_pair_pack {
                if let Some(loc) = self.gl.get_uniform_location(program, "pixel_count") {
                    self.gl.uniform_1_u32(Some(&loc), pixels as u32);
                }
            } else if let Some(loc) = self.gl.get_uniform_location(program, "element_count") {
                self.gl.uniform_1_u32(Some(&loc), elements as u32);
            }
            self.gl
                .dispatch_compute((work_items as u32).div_ceil(256), 1, 1);
            self.gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
            self.gl.finish();
            self.gl.bind_texture(source.target(), None);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
        }
        Ok(())
    }

    /// Pack one RGB texture into three channels of a padded FP16 NCHW shared
    /// buffer. Padding must exactly match the long-standing CPU-visible
    /// interpolation path: replicate the last source row/column. Zero padding
    /// leaks artificial black borders into RIFE/DRBA internal Conv/GridSample
    /// stages whenever the logical size is not already tile-aligned (notably
    /// 1920x1080 -> 1920x1152 for RIFE and -> 1920x1088 for DRBA).
    pub fn pack_interp_rgb_f16(
        &mut self,
        key: u64,
        source: GpuTex,
        source_size: (i32, i32),
        padded_size: (i32, i32),
        destination_channel: usize,
    ) -> Result<(), String> {
        let buffer = self
            .external_imports
            .get(&key)
            .map(|item| item.buffer)
            .ok_or_else(|| "shared interpolation buffer is not imported".to_string())?;
        let plane = usize::try_from(padded_size.0)
            .ok()
            .and_then(|w| usize::try_from(padded_size.1).ok().map(|h| w * h))
            .ok_or_else(|| "invalid padded interpolation dimensions".to_string())?;
        let destination_element = destination_channel * plane;
        if destination_element % 2 != 0 || (plane * 3) % 2 != 0 {
            return Err("interpolation RGB pack requires half2 alignment".into());
        }
        let program = self.compute_program(PACK_INTERP_RGB_F16)?;
        let words = plane * 3 / 2;
        unsafe {
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(source.target(), Some(source.tex));
            if let Some(loc) = self.gl.get_uniform_location(program, "source_tex") {
                self.gl.uniform_1_i32(Some(&loc), 0);
            }
            for (name, value) in [
                ("source_width", source_size.0),
                ("source_height", source_size.1),
                ("padded_width", padded_size.0),
                ("padded_height", padded_size.1),
            ] {
                if let Some(loc) = self.gl.get_uniform_location(program, name) {
                    self.gl.uniform_1_i32(Some(&loc), value);
                }
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "destination_word") {
                self.gl
                    .uniform_1_u32(Some(&loc), (destination_element / 2) as u32);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "word_count") {
                self.gl.uniform_1_u32(Some(&loc), words as u32);
            }
            self.gl.dispatch_compute((words as u32).div_ceil(256), 1, 1);
            self.gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
            self.gl.bind_texture(source.target(), None);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
        }
        Ok(())
    }

    pub fn pack_interp_rgb_f32(
        &mut self,
        key: u64,
        source: GpuTex,
        source_size: (i32, i32),
        padded_size: (i32, i32),
        destination_channel: usize,
    ) -> Result<(), String> {
        let buffer = self
            .external_imports
            .get(&key)
            .map(|v| v.buffer)
            .ok_or_else(|| "shared interpolation buffer is not imported".to_string())?;
        let plane = padded_size.0 as usize * padded_size.1 as usize;
        let program = self.compute_program(PACK_INTERP_RGB_F32)?;
        unsafe {
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(source.target(), Some(source.tex));
            if let Some(l) = self.gl.get_uniform_location(program, "source_tex") {
                self.gl.uniform_1_i32(Some(&l), 0);
            }
            for (n, v) in [
                ("source_width", source_size.0),
                ("source_height", source_size.1),
                ("padded_width", padded_size.0),
                ("padded_height", padded_size.1),
            ] {
                if let Some(l) = self.gl.get_uniform_location(program, n) {
                    self.gl.uniform_1_i32(Some(&l), v);
                }
            }
            if let Some(l) = self.gl.get_uniform_location(program, "destination_element") {
                self.gl
                    .uniform_1_u32(Some(&l), (destination_channel * plane) as u32);
            }
            self.gl
                .dispatch_compute(((plane * 3) as u32).div_ceil(256), 1, 1);
            // RIFE's v1 single-input models in this package expose FP32 I/O
            // even when their internal weights are FP16. The next operation
            // copies these freshly packed RGB planes with glCopyBufferSubData.
            // SHADER_STORAGE_BARRIER_BIT alone only publishes SSBO visibility
            // to later shader accesses; BUFFER_UPDATE_BARRIER_BIT is required
            // before a GL buffer-copy/update command consumes those writes.
            // At large pre-upscaled resolutions the missing boundary could
            // copy the still-pending tail of the buffer, so only the lower
            // part of generated RIFE frames flickered while real/DRBA frames
            // remained correct. Keep this fence-free and GPU-resident.
            self.gl
                .memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT | glow::BUFFER_UPDATE_BARRIER_BIT);
            self.gl.bind_texture(source.target(), None);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
        }
        Ok(())
    }

    /// Fill one padded FP16 NCHW auxiliary plane entirely on the GPU.
    pub fn fill_interp_aux_f16(
        &mut self,
        key: u64,
        padded_size: (i32, i32),
        destination_channel: usize,
        plane_kind: InterpAuxPlane,
    ) -> Result<(), String> {
        let buffer = self
            .external_imports
            .get(&key)
            .map(|item| item.buffer)
            .ok_or_else(|| "shared interpolation buffer is not imported".to_string())?;
        let plane = usize::try_from(padded_size.0)
            .ok()
            .and_then(|w| usize::try_from(padded_size.1).ok().map(|h| w * h))
            .ok_or_else(|| "invalid padded interpolation dimensions".to_string())?;
        let destination_element = destination_channel * plane;
        if destination_element % 2 != 0 || plane % 2 != 0 {
            return Err("interpolation auxiliary pack requires half2 alignment".into());
        }
        let (mode, value) = match plane_kind {
            InterpAuxPlane::Constant(value) => (0, value),
            InterpAuxPlane::HorizontalGrid => (1, 0.0),
            InterpAuxPlane::VerticalGrid => (2, 0.0),
        };
        let words = plane / 2;
        let program = self.compute_program(FILL_INTERP_AUX_F16)?;
        unsafe {
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            if let Some(loc) = self.gl.get_uniform_location(program, "destination_word") {
                self.gl
                    .uniform_1_u32(Some(&loc), (destination_element / 2) as u32);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "word_count") {
                self.gl.uniform_1_u32(Some(&loc), words as u32);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "width") {
                self.gl.uniform_1_i32(Some(&loc), padded_size.0);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "height") {
                self.gl.uniform_1_i32(Some(&loc), padded_size.1);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "mode") {
                self.gl.uniform_1_i32(Some(&loc), mode);
            }
            if let Some(loc) = self.gl.get_uniform_location(program, "constant_value") {
                self.gl.uniform_1_f32(Some(&loc), value);
            }
            self.gl.dispatch_compute((words as u32).div_ceil(256), 1, 1);
            self.gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
        }
        Ok(())
    }

    pub fn fill_interp_aux_f32(
        &mut self,
        key: u64,
        padded_size: (i32, i32),
        destination_channel: usize,
        plane_kind: InterpAuxPlane,
    ) -> Result<(), String> {
        let buffer = self
            .external_imports
            .get(&key)
            .map(|v| v.buffer)
            .ok_or_else(|| "shared interpolation buffer is not imported".to_string())?;
        let plane = padded_size.0 as usize * padded_size.1 as usize;
        let (mode, value) = match plane_kind {
            InterpAuxPlane::Constant(v) => (0, v),
            InterpAuxPlane::HorizontalGrid => (1, 0.0),
            InterpAuxPlane::VerticalGrid => (2, 0.0),
        };
        let program = self.compute_program(FILL_INTERP_AUX_F32)?;
        unsafe {
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(buffer));
            self.gl.use_program(Some(program));
            for (n, v) in [
                ("width", padded_size.0),
                ("height", padded_size.1),
                ("mode", mode),
            ] {
                if let Some(l) = self.gl.get_uniform_location(program, n) {
                    self.gl.uniform_1_i32(Some(&l), v);
                }
            }
            if let Some(l) = self.gl.get_uniform_location(program, "destination_element") {
                self.gl
                    .uniform_1_u32(Some(&l), (destination_channel * plane) as u32);
            }
            if let Some(l) = self.gl.get_uniform_location(program, "constant_value") {
                self.gl.uniform_1_f32(Some(&l), value);
            }
            self.gl.dispatch_compute((plane as u32).div_ceil(256), 1, 1);
            self.gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
        }
        Ok(())
    }

    pub fn copy_external_nchw_f32(
        &mut self,
        source_key: u64,
        destination_key: u64,
        source_element: usize,
        destination_element: usize,
        element_count: usize,
    ) -> Result<(), String> {
        let source = self
            .external_imports
            .get(&source_key)
            .map(|v| v.buffer)
            .ok_or_else(|| "shared source buffer is not imported".to_string())?;
        let destination = self
            .external_imports
            .get(&destination_key)
            .map(|v| v.buffer)
            .ok_or_else(|| "shared destination buffer is not imported".to_string())?;
        unsafe {
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(source));
            self.gl
                .bind_buffer(glow::COPY_WRITE_BUFFER, Some(destination));
            self.gl.copy_buffer_sub_data(
                glow::COPY_READ_BUFFER,
                glow::COPY_WRITE_BUFFER,
                (source_element * 4) as i32,
                (destination_element * 4) as i32,
                (element_count * 4) as i32,
            );
            self.gl
                .memory_barrier(glow::BUFFER_UPDATE_BARRIER_BIT | glow::SHADER_STORAGE_BARRIER_BIT);
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, None);
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
        }
        Ok(())
    }

    /// Copy planar FP16 data between two imported GPU buffers. Offsets and
    /// length are in FP16 elements and must be half2 aligned.
    pub fn copy_external_nchw_f16(
        &mut self,
        source_key: u64,
        destination_key: u64,
        source_element: usize,
        destination_element: usize,
        element_count: usize,
    ) -> Result<(), String> {
        if source_element % 2 != 0 || destination_element % 2 != 0 || element_count % 2 != 0 {
            return Err("external FP16 copy requires half2-aligned ranges".into());
        }
        let source = self
            .external_imports
            .get(&source_key)
            .map(|v| v.buffer)
            .ok_or_else(|| "shared source buffer is not imported".to_string())?;
        let destination = self
            .external_imports
            .get(&destination_key)
            .map(|v| v.buffer)
            .ok_or_else(|| "shared destination buffer is not imported".to_string())?;
        let program = self.compute_program(COPY_EXTERNAL_NCHW_F16)?;
        let words = element_count / 2;
        unsafe {
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(source));
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(destination));
            self.gl.use_program(Some(program));
            for (name, value) in [
                ("source_word", (source_element / 2) as u32),
                ("destination_word", (destination_element / 2) as u32),
                ("word_count", words as u32),
            ] {
                if let Some(loc) = self.gl.get_uniform_location(program, name) {
                    self.gl.uniform_1_u32(Some(&loc), value);
                }
            }
            self.gl.dispatch_compute((words as u32).div_ceil(256), 1, 1);
            self.gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, None);
        }
        Ok(())
    }

    fn delete_external_buffer_import(&mut self, mut import: ExternalBufferImport) {
        let Some(api) = self.external_api else {
            unsafe {
                if let Some(fence) = import.read_fence.take() {
                    self.gl.delete_sync(fence);
                }
                if let Some(snapshot) = import.local_snapshot.take() {
                    self.gl.delete_buffer(snapshot);
                }
                self.gl.delete_buffer(import.buffer);
                let _ = CloseHandle(import.handle);
            }
            return;
        };
        unsafe {
            if let Some(fence) = import.read_fence.take() {
                self.gl.delete_sync(fence);
            }
            if let Some(snapshot) = import.local_snapshot.take() {
                self.gl.delete_buffer(snapshot);
            }
            self.gl.delete_buffer(import.buffer);
            (api.delete_memory_objects)(1, &import.memory);
            let _ = CloseHandle(import.handle);
        }
    }

    fn reap_external_buffer_retirement(&mut self, force: bool) {
        if self.external_retired.is_empty() {
            return;
        }
        if force {
            unsafe {
                self.gl.finish();
            }
        }
        let mut pending = Vec::new();
        for (key, mut import) in std::mem::take(&mut self.external_retired) {
            let ready = if force {
                true
            } else if let Some(fence) = import.read_fence.take() {
                let status = unsafe { self.gl.client_wait_sync(fence, 0, 0) };
                let ready = status == glow::ALREADY_SIGNALED
                    || status == glow::CONDITION_SATISFIED
                    || status == glow::WAIT_FAILED;
                if ready {
                    unsafe {
                        self.gl.delete_sync(fence);
                    }
                } else {
                    import.read_fence = Some(fence);
                }
                ready
            } else {
                true
            };
            if ready {
                self.delete_external_buffer_import(import);
                log::info!("OpenGL shared-buffer quarantine retired: key={key}");
            } else {
                pending.push((key, import));
            }
        }
        self.external_retired = pending;
    }

    pub fn clear_external_buffer_key(&mut self, key: u64) {
        self.reap_external_buffer_retirement(false);
        let Some(mut import) = self.external_imports.remove(&key) else {
            return;
        };
        if let Some(fence) = import.read_fence.take() {
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
            let mut ready = false;
            unsafe {
                loop {
                    let status =
                        self.gl
                            .client_wait_sync(fence, glow::SYNC_FLUSH_COMMANDS_BIT, 5_000_000);
                    if status == glow::ALREADY_SIGNALED
                        || status == glow::CONDITION_SATISFIED
                        || status == glow::WAIT_FAILED
                    {
                        ready = true;
                        break;
                    }
                    if std::time::Instant::now() >= deadline {
                        break;
                    }
                }
            }
            if !ready {
                import.read_fence = Some(fence);
                log::warn!(
                    "OpenGL shared-buffer retirement delayed: key={} bytes={} action=retry-later",
                    key,
                    import.byte_len
                );
                self.external_retired.push((key, import));
                return;
            }
            unsafe {
                self.gl.delete_sync(fence);
            }
        }
        self.delete_external_buffer_import(import);
    }

    pub fn clear_external_buffer(&mut self) {
        let keys: Vec<u64> = self.external_imports.keys().copied().collect();
        for key in keys {
            self.clear_external_buffer_key(key);
        }
        // clear_pool is called only after the provider worker has stopped. A
        // single glFinish here is preferable to leaking quarantined imports
        // across repeated capture sessions.
        self.reap_external_buffer_retirement(true);
    }

    pub fn make_tex(&mut self, w: i32, h: i32, comps: u8, dtype: Dtype) -> GpuTex {
        let key = TexKey {
            w,
            h,
            d: 1,
            comps,
            dtype,
        };
        if let Some(free) = self.pool.get_mut(&key) {
            if let Some(mut t) = free.pop() {
                t.offset_x = 0.0;
                t.offset_y = 0.0;
                // a texture may come back with LINEAR filtering if an error
                // path skipped the restore (this corrupted colours after live
                // chain edits) — force the pool invariant on every reuse
                let gl = &self.gl;
                unsafe {
                    gl.bind_texture(glow::TEXTURE_2D, Some(t.tex));
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MIN_FILTER,
                        glow::NEAREST as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MAG_FILTER,
                        glow::NEAREST as i32,
                    );
                }
                self.tracked.push(t);
                return t;
            }
        }
        let gl = &self.gl;
        unsafe {
            let tex = gl.create_texture().expect("texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            let (internal, format, ty) = formats(comps, dtype);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                internal,
                w,
                h,
                0,
                format,
                ty,
                glow::PixelUnpackData::Slice(None),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            let t = GpuTex {
                tex,
                key,
                offset_x: 0.0,
                offset_y: 0.0,
            };
            self.tracked.push(t);
            t
        }
    }

    /// Upload tightly packed RGBA8 into a fixed-size rotation that is owned by
    /// the GL context rather than the transient pool. The first geometry seen
    /// in a capture session owns the ring; if geometry changes unexpectedly,
    /// return an error so the caller can fall back to the established pool
    /// instead of deleting a texture that may still be the last presented
    /// frame. clear_pool() resets the ring between capture sessions.
    pub fn upload_rgba8_capture_ring(
        &mut self,
        slot: usize,
        slots: usize,
        w: i32,
        h: i32,
        data: &[u8],
    ) -> Result<GpuTex, String> {
        let expected = usize::try_from(w)
            .ok()
            .and_then(|w| usize::try_from(h).ok().and_then(|h| w.checked_mul(h)))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "capture upload ring size overflow".to_string())?;
        if w <= 0 || h <= 0 || data.len() != expected || slots < 2 {
            return Err(format!(
                "invalid capture upload ring frame: {}x{} bytes={} expected={} slots={}",
                w,
                h,
                data.len(),
                expected,
                slots
            ));
        }

        let key = TexKey {
            w,
            h,
            d: 1,
            comps: 4,
            dtype: Dtype::U8,
        };
        if let Some(existing) = self.capture_upload_ring_key {
            if existing != key || self.capture_upload_ring.len() != slots {
                return Err(format!(
                    "geometry changed from {}x{} to {}x{}",
                    existing.w, existing.h, w, h
                ));
            }
        } else {
            let gl = &self.gl;
            let mut ring: Vec<GpuTex> = Vec::with_capacity(slots);
            let (internal, format, ty) = formats(4, Dtype::U8);
            unsafe {
                for _ in 0..slots {
                    let tex = match gl.create_texture() {
                        Ok(texture) => texture,
                        Err(error) => {
                            for allocated in ring.drain(..) {
                                gl.delete_texture(allocated.tex);
                            }
                            return Err(error.to_string());
                        }
                    };
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                    gl.tex_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        internal,
                        w,
                        h,
                        0,
                        format,
                        ty,
                        glow::PixelUnpackData::Slice(None),
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MIN_FILTER,
                        glow::NEAREST as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MAG_FILTER,
                        glow::NEAREST as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_WRAP_S,
                        glow::CLAMP_TO_EDGE as i32,
                    );
                    gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_WRAP_T,
                        glow::CLAMP_TO_EDGE as i32,
                    );
                    ring.push(GpuTex {
                        tex,
                        key,
                        offset_x: 0.0,
                        offset_y: 0.0,
                    });
                }
            }
            self.capture_upload_ring = ring;
            self.capture_upload_ring_key = Some(key);
        }

        let texture = self.capture_upload_ring[slot % self.capture_upload_ring.len()];
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture.tex));
            let (_, format, ty) = formats(4, Dtype::U8);
            self.gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                format,
                ty,
                glow::PixelUnpackData::Slice(Some(data)),
            );
        }
        Ok(texture)
    }

    /// Upload raw uint8 pixels (RGBA, tightly packed) into a pooled RGBA8 tex.
    pub fn upload_rgba8(&mut self, w: i32, h: i32, data: &[u8]) -> GpuTex {
        let t = self.make_tex(w, h, 4, Dtype::U8);
        let gl = &self.gl;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(t.tex));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(data)),
            );
        }
        t
    }

    /// Create or replace a persistent RGBA8 image owned by the GL context.
    /// Unlike pooled frame textures, this survives `release_frame` and is used
    /// by NeoAccel as the exact previous ONNX result.
    pub fn update_persistent_rgba8(
        &mut self,
        key: &str,
        w: i32,
        h: i32,
        data: &[u8],
    ) -> Result<GpuTex, String> {
        let expected = usize::try_from(w)
            .ok()
            .and_then(|w| usize::try_from(h).ok().and_then(|h| w.checked_mul(h)))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "persistent RGBA8 size overflow".to_string())?;
        if data.len() != expected {
            return Err(format!(
                "persistent RGBA8 byte count mismatch: got {}, expected {}",
                data.len(),
                expected
            ));
        }

        let recreate = self.persist.get(key).is_some_and(|existing| {
            existing.w() != w
                || existing.h() != h
                || existing.comps() != 4
                || existing.key.dtype != Dtype::U8
        });
        if recreate {
            self.remove_persistent_texture(key);
        }
        if let Some(existing) = self.persist.get(key).copied() {
            unsafe {
                self.gl.bind_texture(glow::TEXTURE_2D, Some(existing.tex));
                self.gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    w,
                    h,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(data)),
                );
            }
            return Ok(existing);
        }

        Ok(self.persistent_texture(key, w, h, 1, 4, Dtype::U8, false, glow::CLAMP_TO_EDGE, data))
    }

    /// Update a rectangular region of a persistent RGBA8 texture.
    pub fn update_persistent_rgba8_region(
        &mut self,
        key: &str,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        data: &[u8],
    ) -> Result<GpuTex, String> {
        let texture = self
            .persist
            .get(key)
            .copied()
            .ok_or_else(|| format!("persistent texture not found: {key}"))?;
        if texture.comps() != 4 || texture.key.dtype != Dtype::U8 {
            return Err("persistent texture is not RGBA8".into());
        }
        if x < 0
            || y < 0
            || w <= 0
            || h <= 0
            || x.saturating_add(w) > texture.w()
            || y.saturating_add(h) > texture.h()
        {
            return Err("persistent RGBA8 update is out of bounds".into());
        }
        let expected = usize::try_from(w)
            .ok()
            .and_then(|w| usize::try_from(h).ok().and_then(|h| w.checked_mul(h)))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "persistent RGBA8 region size overflow".to_string())?;
        if data.len() != expected {
            return Err(format!(
                "persistent RGBA8 region byte count mismatch: got {}, expected {}",
                data.len(),
                expected
            ));
        }
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture.tex));
            self.gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                x,
                y,
                w,
                h,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(data)),
            );
        }
        Ok(texture)
    }

    /// Copy a 2D RGBA U8/F16 texture into persistent storage of the same
    /// precision without a CPU readback. Slang pass feedback/history uses this
    /// so the external preset's temporal resources survive `release_frame`.
    pub fn copy_to_persistent_rgba(&mut self, key: &str, src: GpuTex) -> Result<GpuTex, String> {
        if src.comps() != 4 || !matches!(src.key.dtype, Dtype::U8 | Dtype::F16) {
            return Err(format!(
                "persistent RGBA copy requires 4-channel U8/F16 input, got comps={} dtype={:?}",
                src.comps(),
                src.key.dtype
            ));
        }
        let recreate = self.persist.get(key).is_some_and(|existing| {
            existing.w() != src.w()
                || existing.h() != src.h()
                || existing.comps() != 4
                || existing.key.dtype != src.key.dtype
        });
        if recreate {
            self.remove_persistent_texture(key);
        }
        let dst = if let Some(existing) = self.persist.get(key).copied() {
            existing
        } else {
            let texture = self.make_persistent_empty_rgba(key, src.w(), src.h(), src.key.dtype)?;
            texture
        };
        const COPY_FRAG: &str = "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D tex;\nvoid main(){ frag = texture(tex, v_uv); }\n";
        let prog = self.program_named("persistent-copy-rgba-generic", COPY_FRAG)?;
        self.bind_target(dst);
        let gl = self.gl.clone();
        unsafe {
            gl.use_program(Some(prog));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(src.tex));
            if let Some(loc) = gl.get_uniform_location(prog, "tex") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            gl.bind_vertex_array(Some(self.quad_vao));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_vertex_array(None);
            gl.memory_barrier(glow::ALL_BARRIER_BITS);
        }
        self.unbind_target();
        Ok(dst)
    }

    fn make_persistent_empty_rgba(
        &mut self,
        key: &str,
        w: i32,
        h: i32,
        dtype: Dtype,
    ) -> Result<GpuTex, String> {
        let gl = &self.gl;
        let texture = unsafe {
            let tex = gl.create_texture().map_err(|e| e.to_string())?;
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            let (internal, format, ty) = formats(4, dtype);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                internal,
                w,
                h,
                0,
                format,
                ty,
                glow::PixelUnpackData::Slice(None),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            GpuTex {
                tex,
                key: TexKey {
                    w,
                    h,
                    d: 1,
                    comps: 4,
                    dtype,
                },
                offset_x: 0.0,
                offset_y: 0.0,
            }
        };
        self.persist.insert(key.to_string(), texture);
        Ok(texture)
    }

    /// Copy any 2D texture into a persistent RGBA8 texture without a CPU
    /// readback. v684 NeoAccel uses this to retain an exact ONNX frame and its
    /// matching non-AI baseline across the following source frame.
    pub fn copy_to_persistent_rgba8(&mut self, key: &str, src: GpuTex) -> Result<GpuTex, String> {
        let recreate = self.persist.get(key).is_some_and(|existing| {
            existing.w() != src.w()
                || existing.h() != src.h()
                || existing.comps() != 4
                || existing.key.dtype != Dtype::U8
        });
        if recreate {
            self.remove_persistent_texture(key);
        }
        let dst = if let Some(existing) = self.persist.get(key).copied() {
            existing
        } else {
            let gl = &self.gl;
            let texture = unsafe {
                let tex = gl.create_texture().map_err(|e| e.to_string())?;
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA8 as i32,
                    src.w(),
                    src.h(),
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(None),
                );
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MIN_FILTER,
                    glow::NEAREST as i32,
                );
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MAG_FILTER,
                    glow::NEAREST as i32,
                );
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_WRAP_S,
                    glow::CLAMP_TO_EDGE as i32,
                );
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_WRAP_T,
                    glow::CLAMP_TO_EDGE as i32,
                );
                GpuTex {
                    tex,
                    key: TexKey {
                        w: src.w(),
                        h: src.h(),
                        d: 1,
                        comps: 4,
                        dtype: Dtype::U8,
                    },
                    offset_x: 0.0,
                    offset_y: 0.0,
                }
            };
            self.persist.insert(key.to_string(), texture);
            texture
        };

        const COPY_FRAG: &str = "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D tex;\nvoid main(){ frag = texture(tex, v_uv); }\n";
        let prog = self.program_named("neoaccel-temporal-copy-rgba8", COPY_FRAG)?;
        self.bind_target(dst);
        let gl = self.gl.clone();
        unsafe {
            gl.use_program(Some(prog));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(src.tex));
            if let Some(loc) = gl.get_uniform_location(prog, "tex") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            gl.bind_vertex_array(Some(self.quad_vao));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_vertex_array(None);
            gl.memory_barrier(glow::ALL_BARRIER_BITS);
        }
        self.unbind_target();
        Ok(dst)
    }

    /// Compose the current non-AI upscale with motion-warped detail from the
    /// previous exact ONNX output. `flow` is a coarse current->previous field
    /// encoded by NeoAccel. Low-confidence blocks receive no historical detail.
    pub fn neoaccel_temporal_detail_compose(
        &mut self,
        current_base: GpuTex,
        previous_ai: GpuTex,
        previous_base: GpuTex,
        flow: GpuTex,
        input_w: i32,
        input_h: i32,
    ) -> Result<GpuTex, String> {
        if current_base.w() != previous_ai.w()
            || current_base.h() != previous_ai.h()
            || current_base.w() != previous_base.w()
            || current_base.h() != previous_base.h()
            || input_w <= 0
            || input_h <= 0
        {
            return Err("NeoAccel temporal texture geometry mismatch".into());
        }
        const FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D current_base;
uniform sampler2D previous_ai;
uniform sampler2D previous_base;
uniform sampler2D flow_tex;
uniform vec2 inv_input_size;
void main(){
    vec4 flow_sample = texture(flow_tex, v_uv);
    vec2 motion_px = floor(flow_sample.rg * 255.0 + 0.5) - vec2(128.0);
    float confidence = smoothstep(0.56, 0.86, flow_sample.b);
    vec2 prev_uv = clamp(v_uv + motion_px * inv_input_size, vec2(0.0), vec2(1.0));
    vec3 base = texture(current_base, v_uv).rgb;
    vec3 detail = texture(previous_ai, prev_uv).rgb - texture(previous_base, prev_uv).rgb;
    // Hard cap only pathological residuals. Normal AI edge/detail energy passes
    // unchanged; the confidence gate handles occlusion/content changes.
    detail = clamp(detail, vec3(-0.25), vec3(0.25));
    frag = vec4(clamp(base + detail * confidence, 0.0, 1.0), 1.0);
}
"#;
        let prog = self.program_named("neoaccel-temporal-detail-v686", FRAG)?;
        let out = self.make_tex(current_base.w(), current_base.h(), 4, Dtype::F16);
        self.bind_target(out);
        let gl = self.gl.clone();
        unsafe {
            gl.use_program(Some(prog));
            for (unit, name, tex) in [
                (0u32, "current_base", current_base),
                (1u32, "previous_ai", previous_ai),
                (2u32, "previous_base", previous_base),
                (3u32, "flow_tex", flow),
            ] {
                gl.active_texture(glow::TEXTURE0 + unit);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex.tex));
                if let Some(loc) = gl.get_uniform_location(prog, name) {
                    gl.uniform_1_i32(Some(&loc), unit as i32);
                }
            }
            if let Some(loc) = gl.get_uniform_location(prog, "inv_input_size") {
                gl.uniform_2_f32(Some(&loc), 1.0 / input_w as f32, 1.0 / input_h as f32);
            }
            gl.bind_vertex_array(Some(self.quad_vao));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_vertex_array(None);
            gl.memory_barrier(glow::ALL_BARRIER_BITS);
        }
        self.unbind_target();
        Ok(out)
    }

    pub fn persistent_texture_by_key(&self, key: &str) -> Option<GpuTex> {
        self.persist.get(key).copied()
    }

    pub fn remove_persistent_texture(&mut self, key: &str) {
        let Some(texture) = self.persist.remove(key) else {
            return;
        };
        unsafe {
            self.gl.delete_texture(texture.tex);
        }
    }

    /// Drop writable mpv shader history while retaining immutable embedded
    /// LUTs. A live preset replacement must not let the newly selected shader
    /// sample storage written by the previous invocation/session geometry.
    pub fn clear_temporal_shader_storage(&mut self) {
        let keys: Vec<String> = self
            .persist
            .keys()
            .filter(|key| {
                key.ends_with("::GLOBAL") || key.ends_with("::LUMA") || key.ends_with("::CHROMA")
            })
            .cloned()
            .collect();
        for key in keys {
            self.remove_persistent_texture(&key);
        }
    }

    /// Upload raw uint8 pixels (RGB, tightly packed) into a pooled RGB8 tex.
    pub fn upload_rgb8(&mut self, w: i32, h: i32, data: &[u8]) -> GpuTex {
        let t = self.make_tex(w, h, 3, Dtype::U8);
        let gl = &self.gl;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(t.tex));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(data)),
            );
        }
        t
    }

    /// Upload raw RGBA16F bytes (HDR capture) into a pooled F16 texture.
    pub fn upload_rgba16f(&mut self, w: i32, h: i32, data: &[u8]) -> GpuTex {
        let t = self.make_tex(w, h, 4, Dtype::F16);
        let gl = &self.gl;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(t.tex));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                glow::RGBA,
                glow::HALF_FLOAT,
                glow::PixelUnpackData::Slice(Some(data)),
            );
        }
        t
    }

    /// Read a texture back as uint8 RGB (ONNX interchange). The GPU converts
    /// to u8 during ReadPixels — a float readback of 1080p cost ~33MB/frame
    /// plus a CPU conversion loop and was the main ONNX-path bottleneck.
    pub fn download_rgb8(&mut self, t: GpuTex) -> Vec<u8> {
        let gl = self.gl.clone();
        let (w, h) = (t.w() as usize, t.h() as usize);
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(t.tex),
                0,
            );
            let mut rgba = vec![0u8; w * h * 4];
            gl.read_pixels(
                0,
                0,
                t.w(),
                t.h(),
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut rgba)),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            use rayon::prelude::*;
            let mut out = vec![0u8; w * h * 3];
            out.par_chunks_mut(3 * 4096)
                .zip(rgba.par_chunks(4 * 4096))
                .for_each(|(o, i)| {
                    for (op, ip) in o.chunks_mut(3).zip(i.chunks(4)) {
                        op[0] = ip[0];
                        op[1] = ip[1];
                        op[2] = ip[2];
                    }
                });
            out
        }
    }

    /// Read a filtered texture as tightly packed RGBA8 into caller-owned
    /// storage. DLSSNR uses this to retain the same large allocation across
    /// frames; screenshots can keep using the allocating convenience wrapper.
    pub fn download_rgba8_into(&mut self, t: GpuTex, rgba: &mut Vec<u8>) {
        let gl = self.gl.clone();
        let (w, h) = (t.w() as usize, t.h() as usize);
        rgba.resize(w * h * 4, 0);
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(t.tex),
                0,
            );
            gl.read_pixels(
                0,
                0,
                t.w(),
                t.h(),
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(rgba.as_mut_slice())),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    /// Read the final filtered texture as tightly packed RGBA8. Used only for
    /// an explicit screenshot request; PNG encoding is performed off-thread.
    pub fn download_rgba8(&mut self, t: GpuTex) -> Vec<u8> {
        let mut rgba = Vec::new();
        self.download_rgba8_into(t, &mut rgba);
        rgba
    }

    /// Read raw f32 RGBA (diagnostics).
    pub fn download_f32(&mut self, t: GpuTex) -> Vec<f32> {
        let gl = self.gl.clone();
        let (w, h) = (t.w() as usize, t.h() as usize);
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(t.tex),
                0,
            );
            let mut buf = vec![0f32; w * h * 4];
            gl.read_pixels(
                0,
                0,
                t.w(),
                t.h(),
                glow::RGBA,
                glow::FLOAT,
                glow::PixelPackData::Slice(Some(bytemuck_cast_slice_mut(&mut buf))),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            buf
        }
    }

    /// Bind the shared FBO to render into `t` and set the viewport.
    pub fn bind_target(&self, t: GpuTex) {
        let gl = &self.gl;
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(t.tex),
                0,
            );
            gl.viewport(0, 0, t.w(), t.h());
        }
    }

    pub fn unbind_target(&self) {
        unsafe { self.gl.bind_framebuffer(glow::FRAMEBUFFER, None) };
    }

    /// Compile (or fetch cached) program for a fixed-layout quad vertex shader.
    pub fn program(&mut self, frag: &str) -> Result<glow::Program, String> {
        self.program_named(&format!("fragment-source:{frag}"), frag)
    }

    pub fn cached_program(&self, key: &str) -> Option<glow::Program> {
        self.prog_cache.get(key).copied()
    }

    /// Compile (or fetch cached) a program with caller-supplied vertex and
    /// fragment stages. Used by the clean-room Libretro Slang preset runner;
    /// ordinary mpv GLSL continues to use `program_named` unchanged.
    pub fn program_pair_named(
        &mut self,
        key: &str,
        vertex: &str,
        fragment: &str,
    ) -> Result<glow::Program, String> {
        if let Some(p) = self.prog_cache.get(key) {
            return Ok(*p);
        }
        let gl = &self.gl;
        unsafe {
            let compile = |ty: u32, src: &str, label: &str| -> Result<glow::Shader, String> {
                let sh = gl.create_shader(ty).map_err(|e| e.to_string())?;
                gl.shader_source(sh, src);
                gl.compile_shader(sh);
                if !gl.get_shader_compile_status(sh) {
                    let info = gl.get_shader_info_log(sh);
                    gl.delete_shader(sh);
                    return Err(format!("{label} shader: {info}"));
                }
                Ok(sh)
            };
            let vs = compile(glow::VERTEX_SHADER, vertex, "vertex")?;
            let fs = match compile(glow::FRAGMENT_SHADER, fragment, "fragment") {
                Ok(shader) => shader,
                Err(error) => {
                    gl.delete_shader(vs);
                    return Err(error);
                }
            };
            let prog = gl.create_program().map_err(|e| e.to_string())?;
            gl.attach_shader(prog, vs);
            gl.attach_shader(prog, fs);
            gl.link_program(prog);
            gl.delete_shader(vs);
            gl.delete_shader(fs);
            if !gl.get_program_link_status(prog) {
                let info = gl.get_program_info_log(prog);
                gl.delete_program(prog);
                return Err(format!("link: {info}"));
            }
            self.prog_cache.insert(key.to_string(), prog);
            Ok(prog)
        }
    }

    /// Draw Neo's shared fullscreen quad with the currently active program and
    /// framebuffer. This keeps Slang passes on the same VAO convention as the
    /// existing mpv GLSL renderer (location 0 = Position, 1 = TexCoord).
    pub fn draw_fullscreen(&self) {
        unsafe {
            self.gl.bind_vertex_array(Some(self.quad_vao));
            self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            self.gl.bind_vertex_array(None);
            self.gl.memory_barrier(glow::ALL_BARRIER_BITS);
        }
    }

    pub fn program_named(&mut self, key: &str, frag: &str) -> Result<glow::Program, String> {
        if let Some(p) = self.prog_cache.get(key) {
            return Ok(*p);
        }
        const VERT: &str = "#version 330\nlayout(location=0) in vec2 in_pos;\nlayout(location=1) in vec2 in_uv;\nout vec2 v_uv;\nvoid main(){ v_uv=in_uv; gl_Position=vec4(in_pos,0.0,1.0); }\n";
        let gl = &self.gl;
        unsafe {
            let compile = |ty: u32, src: &str| -> Result<glow::Shader, String> {
                let sh = gl.create_shader(ty).map_err(|e| e.to_string())?;
                gl.shader_source(sh, src);
                gl.compile_shader(sh);
                if !gl.get_shader_compile_status(sh) {
                    let log = gl.get_shader_info_log(sh);
                    gl.delete_shader(sh);
                    return Err(log);
                }
                Ok(sh)
            };
            let vs = compile(glow::VERTEX_SHADER, VERT)?;
            let fs = compile(glow::FRAGMENT_SHADER, frag).map_err(|e| {
                gl.delete_shader(vs);
                format!("fragment shader: {e}")
            })?;
            let prog = gl.create_program().map_err(|e| e.to_string())?;
            gl.attach_shader(prog, vs);
            gl.attach_shader(prog, fs);
            gl.link_program(prog);
            gl.delete_shader(vs);
            gl.delete_shader(fs);
            if !gl.get_program_link_status(prog) {
                let log = gl.get_program_info_log(prog);
                gl.delete_program(prog);
                return Err(format!("link: {log}"));
            }
            self.prog_cache.insert(key.to_string(), prog);
            Ok(prog)
        }
    }

    /// Compile (or fetch cached) OpenGL compute shader program.
    pub fn compute_program(&mut self, src: &str) -> Result<glow::Program, String> {
        self.compute_program_named(&format!("compute-source:{src}"), src)
    }

    pub fn compute_program_named(&mut self, key: &str, src: &str) -> Result<glow::Program, String> {
        if let Some(p) = self.prog_cache.get(key) {
            return Ok(*p);
        }
        let gl = &self.gl;
        unsafe {
            let sh = gl
                .create_shader(glow::COMPUTE_SHADER)
                .map_err(|e| e.to_string())?;
            gl.shader_source(sh, src);
            gl.compile_shader(sh);
            if !gl.get_shader_compile_status(sh) {
                let log = gl.get_shader_info_log(sh);
                gl.delete_shader(sh);
                return Err(format!("compute shader: {log}"));
            }
            let prog = gl.create_program().map_err(|e| e.to_string())?;
            gl.attach_shader(prog, sh);
            gl.link_program(prog);
            gl.delete_shader(sh);
            if !gl.get_program_link_status(prog) {
                let log = gl.get_program_info_log(prog);
                gl.delete_program(prog);
                return Err(format!("link: {log}"));
            }
            self.prog_cache.insert(key.to_string(), prog);
            Ok(prog)
        }
    }

    /// Get-or-upload a persistent data texture (//!TEXTURE LUT). `key` must be
    /// globally unique (shader path + texture name). Filter/wrap are set once
    /// at creation and never touched by the pool invariant.
    #[allow(clippy::too_many_arguments)]
    pub fn persistent_texture(
        &mut self,
        key: &str,
        w: i32,
        h: i32,
        d: i32,
        comps: u8,
        dtype: Dtype,
        filter_linear: bool,
        wrap: u32,
        data: &[u8],
    ) -> GpuTex {
        if let Some(t) = self.persist.get(key) {
            return *t;
        }
        let gl = &self.gl;
        let t = unsafe {
            let tex = gl.create_texture().expect("texture");
            let target = if d > 1 {
                glow::TEXTURE_3D
            } else {
                glow::TEXTURE_2D
            };
            gl.bind_texture(target, Some(tex));
            let (internal, format, ty) = formats(comps, dtype);
            if d > 1 {
                gl.tex_image_3d(
                    target,
                    0,
                    internal,
                    w,
                    h,
                    d,
                    0,
                    format,
                    ty,
                    glow::PixelUnpackData::Slice(Some(data)),
                );
            } else {
                gl.tex_image_2d(
                    target,
                    0,
                    internal,
                    w,
                    h,
                    0,
                    format,
                    ty,
                    glow::PixelUnpackData::Slice(Some(data)),
                );
            }
            let f = if filter_linear {
                glow::LINEAR
            } else {
                glow::NEAREST
            } as i32;
            gl.tex_parameter_i32(target, glow::TEXTURE_MIN_FILTER, f);
            gl.tex_parameter_i32(target, glow::TEXTURE_MAG_FILTER, f);
            gl.tex_parameter_i32(target, glow::TEXTURE_WRAP_S, wrap as i32);
            gl.tex_parameter_i32(target, glow::TEXTURE_WRAP_T, wrap as i32);
            if d > 1 {
                gl.tex_parameter_i32(target, glow::TEXTURE_WRAP_R, wrap as i32);
            }
            GpuTex {
                tex,
                key: TexKey {
                    w,
                    h,
                    d,
                    comps,
                    dtype,
                },
                offset_x: 0.0,
                offset_y: 0.0,
            }
        };
        self.persist.insert(key.to_string(), t);
        t
    }

    /// Whether `t` is a persistent //!TEXTURE (its filter must not be touched
    /// by the per-pass scaled-bind filter dance).
    pub fn is_persistent(&self, t: GpuTex) -> bool {
        self.persist.values().any(|p| p.tex == t.tex)
    }

    /// Return every texture made this frame (except `keep`) to the pool.
    pub fn release_frame(&mut self, keep: &[GpuTex]) {
        let mut keep_ids: Vec<glow::Texture> = keep.iter().map(|t| t.tex).collect();
        keep_ids.extend(self.external_neoflow_state.textures().map(|t| t.tex));
        let mut survivors = Vec::with_capacity(keep.len());
        for t in self.tracked.drain(..) {
            if keep_ids.contains(&t.tex) {
                survivors.push(t);
            } else {
                self.pool.entry(t.key).or_default().push(t);
            }
        }
        self.tracked = survivors;
    }

    /// Return a kept texture (e.g. last frame's output) to the pool.
    pub fn recycle(&mut self, t: GpuTex) {
        if self
            .capture_upload_ring
            .iter()
            .any(|capture| capture.tex == t.tex)
        {
            return;
        }
        // Kept outputs survive release_frame() in `tracked`. Remove the
        // survivor before pooling it so one GL texture cannot be handed out
        // twice through both lifecycle lists.
        if let Some(index) = self.tracked.iter().position(|tracked| tracked.tex == t.tex) {
            self.tracked.swap_remove(index);
        }
        if self
            .pool
            .get(&t.key)
            .is_some_and(|free| free.iter().any(|pooled| pooled.tex == t.tex))
        {
            return;
        }
        self.pool.entry(t.key).or_default().push(t);
    }

    /// Keep a small warm cache of transient textures across an in-session
    /// source resize. Deleting the complete pool (including persistent shader
    /// LUTs) made resolution changes behave like a cold application start and
    /// could leave x4/x5 below their previous rate until capture was restarted.
    pub fn trim_transient_pool(&mut self, max_free_per_key: usize) {
        const MAX_TRANSIENT_POOL_BYTES: usize = 256 * 1024 * 1024;

        fn texture_bytes(key: TexKey) -> usize {
            let bytes_per_component = match key.dtype {
                Dtype::U8 => 1usize,
                Dtype::F16 => 2usize,
                Dtype::U32 => 4usize,
            };
            usize::try_from(key.w.max(0))
                .unwrap_or(0)
                .saturating_mul(usize::try_from(key.h.max(0)).unwrap_or(0))
                .saturating_mul(usize::try_from(key.d.max(1)).unwrap_or(1))
                .saturating_mul(key.comps as usize)
                .saturating_mul(bytes_per_component)
        }

        let gl = &self.gl;
        let before_bytes: usize = self
            .pool
            .iter()
            .map(|(key, free)| texture_bytes(*key).saturating_mul(free.len()))
            .sum();
        let mut removed = 0usize;
        unsafe {
            for free in self.pool.values_mut() {
                while free.len() > max_free_per_key {
                    if let Some(texture) = free.pop() {
                        gl.delete_texture(texture.tex);
                        removed = removed.saturating_add(1);
                    }
                }
            }

            loop {
                let total_bytes: usize = self
                    .pool
                    .iter()
                    .map(|(key, free)| texture_bytes(*key).saturating_mul(free.len()))
                    .sum();
                if total_bytes <= MAX_TRANSIENT_POOL_BYTES {
                    break;
                }
                let largest_key = self
                    .pool
                    .iter()
                    .filter(|(_, free)| !free.is_empty())
                    .max_by_key(|(key, _)| texture_bytes(**key))
                    .map(|(key, _)| *key);
                let Some(key) = largest_key else {
                    break;
                };
                if let Some(texture) = self.pool.get_mut(&key).and_then(Vec::pop) {
                    gl.delete_texture(texture.tex);
                    removed = removed.saturating_add(1);
                }
            }
        }
        self.pool.retain(|_, free| !free.is_empty());
        let after_bytes: usize = self
            .pool
            .iter()
            .map(|(key, free)| texture_bytes(*key).saturating_mul(free.len()))
            .sum();
        if removed != 0 || before_bytes > MAX_TRANSIENT_POOL_BYTES {
            log::info!(
                "gl-texture-pool-trim: before_mb={:.1} after_mb={:.1} removed={} key_count={} cap_mb={}",
                before_bytes as f64 / (1024.0 * 1024.0),
                after_bytes as f64 / (1024.0 * 1024.0),
                removed,
                self.pool.len(),
                MAX_TRANSIENT_POOL_BYTES / (1024 * 1024)
            );
        }
    }

    /// Free all GL textures (between capture sessions).
    pub fn clear_pool(&mut self) {
        self.external_neoflow_state = ExternalNeoFlowState::default();
        let pending_fences = self
            .command_fences
            .drain()
            .map(|(_, fence)| fence)
            .collect::<Vec<_>>();
        let pending_timers = self
            .gpu_timer_queries
            .drain(..)
            .map(|(query, _)| query)
            .collect::<Vec<_>>();
        self.gpu_timer_last_sample.clear();
        self.clear_external_buffer();
        let gl = &self.gl;
        unsafe {
            for fence in pending_fences {
                gl.delete_sync(fence);
            }
            for query in pending_timers {
                gl.delete_query(query);
            }
            for t in self.capture_upload_ring.drain(..) {
                gl.delete_texture(t.tex);
            }
            self.capture_upload_ring_key = None;
            for t in self.tracked.drain(..) {
                gl.delete_texture(t.tex);
            }
            for (_, v) in self.pool.drain() {
                for t in v {
                    gl.delete_texture(t.tex);
                }
            }
            for (_, t) in self.persist.drain() {
                gl.delete_texture(t.tex);
            }
        }
    }

    pub fn take_external_neoflow_state(&mut self) -> ExternalNeoFlowState {
        std::mem::take(&mut self.external_neoflow_state)
    }

    pub fn set_external_neoflow_state(&mut self, state: ExternalNeoFlowState) {
        self.external_neoflow_state = state;
    }

    pub fn invalidate_external_neoflow_history(&mut self, mode_too: bool) {
        self.external_neoflow_state.flow_history_valid = false;
        if mode_too {
            self.external_neoflow_state.mode_history_valid = false;
        }
    }

    pub fn live_texture_count(&self) -> usize {
        self.capture_upload_ring.len()
            + self.tracked.len()
            + self.pool.values().map(|v| v.len()).sum::<usize>()
    }

    pub fn set_filter_linear(&self, t: GpuTex, linear: bool) {
        let f = if linear { glow::LINEAR } else { glow::NEAREST } as i32;
        let gl = &self.gl;
        unsafe {
            gl.bind_texture(t.target(), Some(t.tex));
            gl.tex_parameter_i32(t.target(), glow::TEXTURE_MIN_FILTER, f);
            gl.tex_parameter_i32(t.target(), glow::TEXTURE_MAG_FILTER, f);
        }
    }

    pub fn finish(&self) {
        unsafe { self.gl.finish() };
    }

    /// Begin a GPU elapsed-time query. This only inserts query commands and
    /// never waits for earlier work to finish.
    pub fn begin_gpu_timer(&mut self, label: &str) -> Option<glow::Query> {
        let now = std::time::Instant::now();
        if self
            .gpu_timer_last_sample
            .get(label)
            .is_some_and(|last| now.duration_since(*last) < std::time::Duration::from_millis(500))
        {
            return None;
        }
        unsafe {
            let query = self.gl.create_query().ok()?;
            self.gl.begin_query(glow::TIME_ELAPSED, query);
            self.gpu_timer_last_sample.insert(label.to_string(), now);
            Some(query)
        }
    }

    pub fn end_gpu_timer(&mut self, query: glow::Query, label: String) {
        unsafe {
            self.gl.end_query(glow::TIME_ELAPSED);
        }
        self.gpu_timer_queries.push_back((query, label));
    }

    /// Collect only already-completed queries. The value is the real GPU time
    /// for one sampled frame; unavailable results remain queued for later.
    pub fn poll_gpu_timers(&mut self) -> Vec<(String, f64)> {
        let mut ready = Vec::new();
        loop {
            let Some((query, _)) = self.gpu_timer_queries.front() else {
                break;
            };
            let available = unsafe {
                self.gl
                    .get_query_parameter_u32(*query, glow::QUERY_RESULT_AVAILABLE)
                    != 0
            };
            if !available {
                break;
            }
            let (query, label) = self.gpu_timer_queries.pop_front().unwrap();
            let elapsed_ns = unsafe { self.gl.get_query_parameter_u64(query, glow::QUERY_RESULT) };
            unsafe {
                self.gl.delete_query(query);
            }
            ready.push((label, elapsed_ns as f64 / 1_000_000.0));
        }
        ready
    }

    /// Discard outstanding statistics queries when the live filter chain is
    /// replaced. No wait is required; queries are owned by this GL context.
    pub fn clear_gpu_timers(&mut self) {
        let pending = self
            .gpu_timer_queries
            .drain(..)
            .map(|(query, _)| query)
            .collect::<Vec<_>>();
        unsafe {
            for query in pending {
                self.gl.delete_query(query);
            }
        }
        self.gpu_timer_last_sample.clear();
    }

    /// Submit a fence for commands issued before this point. The returned
    /// token is polled from later render-loop ticks; this function never waits.
    pub fn submit_commands_fence(&mut self) -> Result<u64, String> {
        unsafe {
            let fence = self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)?;
            // Make the fence visible to the driver without waiting for it.
            self.gl.flush();
            let token = self.next_command_fence;
            self.next_command_fence = self.next_command_fence.wrapping_add(1).max(1);
            if let Some(old) = self.command_fences.insert(token, fence) {
                self.gl.delete_sync(old);
            }
            Ok(token)
        }
    }

    /// Non-blocking poll of a previously submitted command fence.
    pub fn poll_commands_fence(&mut self, token: u64) -> Result<bool, String> {
        let Some(fence) = self.command_fences.get(&token).copied() else {
            // A consumed/cancelled token is already complete from the caller's
            // point of view.
            return Ok(true);
        };
        unsafe {
            let status = self.gl.client_wait_sync(fence, 0, 0);
            if status == glow::ALREADY_SIGNALED || status == glow::CONDITION_SATISFIED {
                self.command_fences.remove(&token);
                self.gl.delete_sync(fence);
                return Ok(true);
            }
            if status == glow::WAIT_FAILED {
                self.command_fences.remove(&token);
                self.gl.delete_sync(fence);
                return Err("OpenGL interpolation handoff fence failed".into());
            }
        }
        Ok(false)
    }

    pub fn cancel_commands_fence(&mut self, token: u64) {
        if let Some(fence) = self.command_fences.remove(&token) {
            unsafe { self.gl.delete_sync(fence) };
        }
    }

    /// Compatibility helper for diagnostic code. It
    /// has a hard deadline and therefore cannot wedge the render thread.
    pub fn wait_submitted_commands(&mut self) -> Result<(), String> {
        let token = self.submit_commands_fence()?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
        loop {
            if self.poll_commands_fence(token)? {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                self.cancel_commands_fence(token);
                return Err("OpenGL interpolation handoff fence timed out after 250ms".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

const EXTERNAL_RGBA8_BUFFER_TO_TEXTURE: &str = r#"#version 430
layout(local_size_x=16, local_size_y=8) in;
layout(std430, binding=0) readonly buffer Source { uint pixels[]; };
layout(rgba8, binding=0) uniform writeonly image2D dst;
uniform int width;
uniform int height;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= width || p.y >= height) return;
    uint rgba_word = pixels[uint(p.y * width + p.x)];
    imageStore(dst, p, unpackUnorm4x8(rgba_word));
}
"#;

const EXTERNAL_NCHW_F16_TO_RGBA8_PAIR_PIXELS: &str = r#"#version 430
layout(local_size_x=256) in;
layout(std430, binding=0) readonly buffer Source { uint words[]; };
layout(rgba8, binding=0) uniform writeonly image2D dst;
uniform int width;
uniform int height;
uniform uint pixel_count;
void main() {
    uint pair = gl_GlobalInvocationID.x;
    uint first = pair * 2u;
    if (first >= pixel_count) return;
    uint second = first + 1u;
    uint plane_words = pixel_count >> 1u;
    vec2 r = unpackHalf2x16(words[pair]);
    vec2 g = unpackHalf2x16(words[plane_words + pair]);
    vec2 b = unpackHalf2x16(words[2u * plane_words + pair]);
    ivec2 p0 = ivec2(int(first % uint(width)), int(first / uint(width)));
    ivec2 p1 = ivec2(int(second % uint(width)), int(second / uint(width)));
    imageStore(dst, p0, vec4(r.x, g.x, b.x, 1.0));
    imageStore(dst, p1, vec4(r.y, g.y, b.y, 1.0));
}
"#;

const EXTERNAL_NCHW_F16_TO_RGBA8_LEGACY: &str = r#"#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(std430, binding=0) readonly buffer Source { uint words[]; };
layout(rgba8, binding=0) uniform writeonly image2D dst;
uniform int width;
uniform int height;
uniform int plane_width;
uniform int plane_height;
float load_half(uint index) {
    vec2 pair = unpackHalf2x16(words[index >> 1u]);
    return (index & 1u) == 0u ? pair.x : pair.y;
}
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= width || p.y >= height) return;
    uint i = uint(p.y * plane_width + p.x);
    uint plane = uint(plane_width * plane_height);
    imageStore(dst, p, vec4(load_half(i), load_half(plane + i), load_half(2u * plane + i), 1.0));
}
"#;

const EXTERNAL_NCHW_F16_TO_RGBA16F: &str = r#"#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(std430, binding=0) readonly buffer Source { uint words[]; };
layout(rgba16f, binding=0) uniform writeonly image2D dst;
uniform int width;
uniform int height;
uniform int plane_width;
uniform int plane_height;
float load_half(uint index) {
    vec2 pair = unpackHalf2x16(words[index >> 1u]);
    return (index & 1u) == 0u ? pair.x : pair.y;
}
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= width || p.y >= height) return;
    uint i = uint(p.y * plane_width + p.x);
    uint plane = uint(plane_width * plane_height);
    imageStore(dst, p, vec4(load_half(i), load_half(plane + i), load_half(2u * plane + i), 1.0));
}
"#;

const EXTERNAL_NCHW_F32_TO_RGBA16F: &str = r#"#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=0) readonly buffer Source{float values[];};
layout(rgba16f,binding=0) uniform writeonly image2D dst;
uniform int width; uniform int height; uniform int plane_width; uniform int plane_height;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(p.x>=width||p.y>=height)return;uint i=uint(p.y*plane_width+p.x);uint plane=uint(plane_width*plane_height);imageStore(dst,p,vec4(values[i],values[plane+i],values[2u*plane+i],1.0));}
"#;

const EXTERNAL_NCHW_F32_TO_RGBA8: &str = r#"#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=0) readonly buffer Source{float values[];};
layout(rgba8,binding=0) uniform writeonly image2D dst;
uniform int width; uniform int height; uniform int plane_width; uniform int plane_height;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(p.x>=width||p.y>=height)return;uint i=uint(p.y*plane_width+p.x);uint plane=uint(plane_width*plane_height);imageStore(dst,p,vec4(values[i],values[plane+i],values[2u*plane+i],1.0));}
"#;

const RGBA_TEXTURE_TO_EXTERNAL_RGBA8: &str = r#"#version 430
layout(local_size_x=16,local_size_y=8) in;
layout(std430,binding=0) writeonly buffer Destination{uint words[];};
uniform sampler2D source_tex; uniform int width; uniform int height;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(p.x>=width||p.y>=height)return;uint i=uint(p.y*width+p.x);vec4 v=clamp(texelFetch(source_tex,p,0),0.0,1.0);words[i]=packUnorm4x8(v);}
"#;

const EXTERNAL_RGBA8_BUFFER_TO_TEXTURE_ALPHA: &str = r#"#version 430
layout(local_size_x=16,local_size_y=8) in;
layout(std430,binding=0) readonly buffer Source{uint words[];};
layout(rgba8,binding=0) uniform writeonly image2D dst;
uniform sampler2D alpha_tex; uniform int width; uniform int height;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(p.x>=width||p.y>=height)return;uint i=uint(p.y*width+p.x);vec4 v=unpackUnorm4x8(words[i]);v.a=texelFetch(alpha_tex,p,0).a;imageStore(dst,p,v);}
"#;

// Fast path for the overwhelmingly common even-pixel image geometry. Each
// invocation fetches two RGBA pixels once and emits one packed FP16 word per
// R/G/B plane. The previous element-major shader fetched the same source pixel
// separately for each channel (up to 3 texelFetch operations per pixel) and
// launched three times as many invocations. NeoAMD's Backend Pack already
// outruns DirectML enough that this host-side pack became measurable.
const RGBA_TEXTURE_TO_EXTERNAL_NCHW_FP16_PAIR_PIXELS: &str = r#"#version 430
layout(local_size_x=256) in;
layout(std430, binding=0) writeonly buffer Destination { uint words[]; };
uniform sampler2D source_tex;
uniform int width;
uniform int height;
uniform uint pixel_count;
void main() {
    uint pair = gl_GlobalInvocationID.x;
    uint first = pair * 2u;
    if (first >= pixel_count) return;
    uint second = first + 1u;
    ivec2 p0 = ivec2(int(first % uint(width)), int(first / uint(width)));
    ivec2 p1 = ivec2(int(second % uint(width)), int(second / uint(width)));
    vec3 a = texelFetch(source_tex, p0, 0).rgb;
    vec3 b = texelFetch(source_tex, p1, 0).rgb;
    uint plane_words = pixel_count >> 1u;
    words[pair] = packHalf2x16(vec2(a.r, b.r));
    words[plane_words + pair] = packHalf2x16(vec2(a.g, b.g));
    words[2u * plane_words + pair] = packHalf2x16(vec2(a.b, b.b));
}
"#;

// Exact fallback for odd pixel counts where an FP16 word can straddle two NCHW
// planes. Keep the long-standing element-major mapping for those rare shapes.
const RGBA_TEXTURE_TO_EXTERNAL_NCHW_FP16_LEGACY: &str = r#"#version 430
layout(local_size_x=256) in;
layout(std430, binding=0) writeonly buffer Destination { uint words[]; };
uniform sampler2D source_tex;
uniform int width;
uniform int height;
uniform uint element_count;
float load_planar(uint index) {
    uint plane = uint(width * height);
    uint channel = index / plane;
    uint pixel = index - channel * plane;
    ivec2 p = ivec2(int(pixel % uint(width)), int(pixel / uint(width)));
    vec4 value = texelFetch(source_tex, p, 0);
    return channel == 0u ? value.r : (channel == 1u ? value.g : value.b);
}
void main() {
    uint word = gl_GlobalInvocationID.x;
    uint first = word * 2u;
    if (first >= element_count) return;
    float lo = load_planar(first);
    float hi = first + 1u < element_count ? load_planar(first + 1u) : 0.0;
    words[word] = packHalf2x16(vec2(lo, hi));
}
"#;

const PACK_INTERP_RGB_F16: &str = r#"#version 430
layout(local_size_x=256) in;
layout(std430, binding=0) buffer Destination { uint words[]; };
uniform sampler2D source_tex;
uniform int source_width;
uniform int source_height;
uniform int padded_width;
uniform int padded_height;
uniform uint destination_word;
uniform uint word_count;
float value_at(uint element) {
    uint plane = uint(padded_width * padded_height);
    uint channel = element / plane;
    uint pixel = element - channel * plane;
    int x = int(pixel % uint(padded_width));
    int y = int(pixel / uint(padded_width));
    // Match fill_padded_rgb_frame_* exactly: edge replication, never black
    // padding. source_width/source_height are validated positive by caller.
    x = min(x, source_width - 1);
    y = min(y, source_height - 1);
    vec4 value = texelFetch(source_tex, ivec2(x, y), 0);
    return channel == 0u ? value.r : (channel == 1u ? value.g : value.b);
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= word_count) return;
    uint first = i * 2u;
    words[destination_word + i] = packHalf2x16(vec2(value_at(first), value_at(first + 1u)));
}
"#;

const PACK_INTERP_RGB_F32: &str = r#"#version 430
layout(local_size_x=256) in; layout(std430,binding=0) buffer Destination{float values[];};
uniform sampler2D source_tex; uniform int source_width; uniform int source_height; uniform int padded_width; uniform int padded_height; uniform uint destination_element;
void main(){uint e=gl_GlobalInvocationID.x;uint plane=uint(padded_width*padded_height);if(e>=3u*plane)return;uint c=e/plane;uint pixel=e-c*plane;int x=min(int(pixel%uint(padded_width)),source_width-1);int y=min(int(pixel/uint(padded_width)),source_height-1);vec4 p=texelFetch(source_tex,ivec2(x,y),0);float v=c==0u?p.r:(c==1u?p.g:p.b);values[destination_element+e]=v;}
"#;

const FILL_INTERP_AUX_F16: &str = r#"#version 430
layout(local_size_x=256) in;
layout(std430, binding=0) buffer Destination { uint words[]; };
uniform uint destination_word;
uniform uint word_count;
uniform int width;
uniform int height;
uniform int mode;
uniform float constant_value;
float value_at(uint pixel) {
    uint x = pixel % uint(width);
    uint y = pixel / uint(width);
    if (mode == 1) return width > 1 ? float(x) * (2.0 / float(width - 1)) - 1.0 : 0.0;
    if (mode == 2) return height > 1 ? float(y) * (2.0 / float(height - 1)) - 1.0 : 0.0;
    return constant_value;
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= word_count) return;
    uint first = i * 2u;
    words[destination_word + i] = packHalf2x16(vec2(value_at(first), value_at(first + 1u)));
}
"#;

const FILL_INTERP_AUX_F32: &str = r#"#version 430
layout(local_size_x=256) in; layout(std430,binding=0) buffer Destination{float values[];};
uniform uint destination_element; uniform int width; uniform int height; uniform int mode; uniform float constant_value;
void main(){uint i=gl_GlobalInvocationID.x;uint plane=uint(width*height);if(i>=plane)return;uint x=i%uint(width);uint y=i/uint(width);float v=constant_value;if(mode==1)v=width>1?float(x)*(2.0/float(width-1))-1.0:0.0;else if(mode==2)v=height>1?float(y)*(2.0/float(height-1))-1.0:0.0;values[destination_element+i]=v;}
"#;

const COPY_EXTERNAL_NCHW_F16: &str = r#"#version 430
layout(local_size_x=256) in;
layout(std430, binding=0) readonly buffer Source { uint source_words[]; };
layout(std430, binding=1) writeonly buffer Destination { uint destination_words[]; };
uniform uint source_word;
uniform uint destination_word;
uniform uint word_count;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < word_count) destination_words[destination_word + i] = source_words[source_word + i];
}
"#;

fn bytemuck_cast_slice(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn bytemuck_cast_slice_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
