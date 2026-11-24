//! Optional Arc ONNX executor. DirectML remains the app's default and fallback.
//! The first app integration uses host tensors; the OpenCL/D3D11 resident
//! boundary proven by the NeoArc probes can replace this transport later.

use anyhow::{Context, Result, anyhow, ensure};
use half::f16;
use libloading::{Library, os::windows as winlib};
use std::{ffi::{CStr, CString, c_char, c_void}, path::{Path, PathBuf}, ptr};
use std::os::windows::ffi::OsStrExt;
use std::sync::{Arc,Mutex,OnceLock};
use windows::{core::PCWSTR, Win32::System::LibraryLoader::{AddDllDirectory, RemoveDllDirectory}};

#[path = "neoarc_graph.rs"]
mod neoarc_graph;
#[path = "neoarc_rgb.rs"]
mod neoarc_rgb;
#[path="neoarc_compact.rs"]
mod neoarc_compact;

type Handle = *mut c_void;

#[repr(C)]
struct Shape { rank: i64, dims: *mut i64 }
#[repr(C)]
#[derive(Clone, Copy)]
struct Dimension { min: i64, max: i64 }
#[repr(C)]
#[derive(Clone, Copy)]
struct PartialShape { rank: Dimension, dims: *mut Dimension }
#[repr(C)]
struct Devices { names: *mut *mut c_char, len: usize }
#[repr(C)]
struct Property { key: *const c_char, value: *const c_void }

type ClHandle = *mut c_void;

/// v995: CL_QUEUE_PRIORITY_KHR level for the GL interpolation queue.
/// CHIDESCALER_NEOARC_QUEUE_PRIORITY=high|med|low|off.
fn neoarc_queue_priority()->Option<u64>{
    match std::env::var("CHIDESCALER_NEOARC_QUEUE_PRIORITY").ok().as_deref().map(str::trim){
        Some("high")=>Some(1),
        Some("med")=>Some(2),
        Some("low")=>Some(4),
        _=>None,
    }
}
type ClInt = i32;
#[derive(Clone, Copy)]
struct ClApi {
    platforms: unsafe extern "system" fn(u32, *mut ClHandle, *mut u32) -> ClInt,
    devices: unsafe extern "system" fn(ClHandle, u64, u32, *mut ClHandle, *mut u32) -> ClInt,
    device_info: unsafe extern "system" fn(ClHandle, u32, usize, *mut c_void, *mut usize) -> ClInt,
    context_create: unsafe extern "system" fn(*const isize, u32, *const ClHandle,
        *mut c_void, *mut c_void, *mut ClInt) -> ClHandle,
    queue_create: unsafe extern "system" fn(ClHandle, ClHandle, u64, *mut ClInt) -> ClHandle,
    buffer_create: unsafe extern "system" fn(ClHandle, u64, usize, *mut c_void, *mut ClInt) -> ClHandle,
    write: unsafe extern "system" fn(ClHandle, ClHandle, u32, usize, usize, *const c_void,
        u32, *const ClHandle, *mut ClHandle) -> ClInt,
    read: unsafe extern "system" fn(ClHandle, ClHandle, u32, usize, usize, *mut c_void,
        u32, *const ClHandle, *mut ClHandle) -> ClInt,
    buffer_free: unsafe extern "system" fn(ClHandle) -> ClInt,
    queue_free: unsafe extern "system" fn(ClHandle) -> ClInt,
    context_free: unsafe extern "system" fn(ClHandle) -> ClInt,
}

impl ClApi {
    unsafe fn load(library: &Library) -> Result<Self> {
        macro_rules! symbol { ($name:literal) => { *unsafe { library.get(concat!($name, "\0").as_bytes())? } }; }
        Ok(Self {
            platforms: symbol!("clGetPlatformIDs"), devices: symbol!("clGetDeviceIDs"),
            device_info: symbol!("clGetDeviceInfo"), context_create: symbol!("clCreateContext"),
            queue_create: symbol!("clCreateCommandQueue"), buffer_create: symbol!("clCreateBuffer"),
            write: symbol!("clEnqueueWriteBuffer"), read: symbol!("clEnqueueReadBuffer"),
            buffer_free: symbol!("clReleaseMemObject"), queue_free: symbol!("clReleaseCommandQueue"),
            context_free: symbol!("clReleaseContext"),
        })
    }
}

fn cl_check(status: ClInt, operation: &str) -> Result<()> {
    ensure!(status == 0, "NeoArc {operation}: OpenCL status {status}");
    Ok(())
}

struct CachedShape {
    size: (usize, usize),
    model: Handle,
    compiled: Handle,
    request: Handle,
    output: Handle,
}

#[derive(Clone, Copy)]
struct Api {
    last_error: unsafe extern "C" fn()->*const c_char,
    core_create: unsafe extern "C" fn(*mut Handle) -> i32,
    core_free: unsafe extern "C" fn(Handle),
    available: unsafe extern "C" fn(Handle, *mut Devices) -> i32,
    available_free: unsafe extern "C" fn(*mut Devices),
    property: unsafe extern "C" fn(Handle, *const c_char, *const c_char, *mut *mut c_char) -> i32,
    string_free: unsafe extern "C" fn(*const c_char),
    model_read: unsafe extern "C" fn(Handle, *const c_char, *const c_char, *mut Handle) -> i32,
    model_free: unsafe extern "C" fn(Handle),
    shape_create: unsafe extern "C" fn(i64, *const i64, *mut PartialShape) -> i32,
    shape_free: unsafe extern "C" fn(*mut PartialShape),
    model_reshape: unsafe extern "C" fn(Handle, PartialShape) -> i32,
    compile: unsafe extern "C" fn(Handle, Handle, *const c_char, usize, *mut Handle) -> i32,
    compile_props: Option<unsafe extern "C" fn(Handle, Handle, *const c_char, usize, *const Property, *mut Handle) -> i32>,
    context_create: unsafe extern "C" fn(Handle, *const c_char, usize, *const Property, *mut Handle) -> i32,
    context_free: unsafe extern "C" fn(Handle),
    remote_tensor_create: unsafe extern "C" fn(Handle, i32, Shape, usize, *const Property, *mut Handle) -> i32,
    compile_remote: unsafe extern "C" fn(Handle, Handle, Handle, usize, *const Property, *mut Handle) -> i32,
    compiled_free: unsafe extern "C" fn(Handle),
    request_create: unsafe extern "C" fn(Handle, *mut Handle) -> i32,
    request_free: unsafe extern "C" fn(Handle),
    tensor_from_host: unsafe extern "C" fn(i32, Shape, *mut c_void, *mut Handle) -> i32,
    tensor_create: unsafe extern "C" fn(i32, Shape, *mut Handle) -> i32,
    tensor_free: unsafe extern "C" fn(Handle),
    tensor_data: unsafe extern "C" fn(Handle, *mut Handle) -> i32,
    tensor_shape: unsafe extern "C" fn(Handle, *mut Shape) -> i32,
    static_shape_free: unsafe extern "C" fn(*mut Shape) -> i32,
    tensor_type: unsafe extern "C" fn(Handle, *mut i32) -> i32,
    get_output: unsafe extern "C" fn(Handle, usize, *mut Handle) -> i32,
    set_input: unsafe extern "C" fn(Handle, Handle) -> i32,
    set_output: unsafe extern "C" fn(Handle, usize, Handle) -> i32,
    infer: unsafe extern "C" fn(Handle) -> i32,
    start_async: unsafe extern "C" fn(Handle) -> i32,
    wait: unsafe extern "C" fn(Handle) -> i32,
}

impl Api {
    fn check_detail(&self,status:i32,operation:&str)->Result<()> {
        if status!=0 {
            let message=unsafe{(self.last_error)()};
            let detail=if message.is_null(){std::borrow::Cow::Borrowed("no runtime detail")}else{unsafe{CStr::from_ptr(message)}.to_string_lossy()};
            anyhow::bail!("NeoArc {operation}: OpenVINO status {status}: {detail}");
        }
        Ok(())
    }
    unsafe fn load(library: &Library) -> Result<Self> {
        macro_rules! symbol { ($name:literal) => { *unsafe { library.get(concat!($name, "\0").as_bytes())? } }; }
        Ok(Self {
            last_error: symbol!("ov_get_last_err_msg"),
            core_create: symbol!("ov_core_create"), core_free: symbol!("ov_core_free"),
            available: symbol!("ov_core_get_available_devices"),
            available_free: symbol!("ov_available_devices_free"),
            property: symbol!("ov_core_get_property"), string_free: symbol!("ov_free"),
            model_read: symbol!("ov_core_read_model"), model_free: symbol!("ov_model_free"),
            shape_create: symbol!("ov_partial_shape_create_static"),
            shape_free: symbol!("ov_partial_shape_free"),
            model_reshape: symbol!("ov_model_reshape_single_input"),
            compile: symbol!("ov_core_compile_model"),
            compile_props: unsafe { library.get(b"ov_core_compile_model_props\0").ok().map(|s: libloading::Symbol<unsafe extern "C" fn(Handle, Handle, *const c_char, usize, *const Property, *mut Handle) -> i32>| *s) },
            context_create: symbol!("ov_core_create_context_props"),
            context_free: symbol!("ov_remote_context_free"),
            remote_tensor_create: symbol!("ov_remote_context_create_tensor_props"),
            compile_remote: symbol!("ov_core_compile_model_with_context_props"),
            compiled_free: symbol!("ov_compiled_model_free"),
            request_create: symbol!("ov_compiled_model_create_infer_request"),
            request_free: symbol!("ov_infer_request_free"),
            tensor_from_host: symbol!("ov_tensor_create_from_host_ptr"),
            tensor_create: symbol!("ov_tensor_create"),
            tensor_free: symbol!("ov_tensor_free"), tensor_data: symbol!("ov_tensor_data"),
            tensor_shape: symbol!("ov_tensor_get_shape"),
            static_shape_free: symbol!("ov_shape_free"),
            tensor_type: symbol!("ov_tensor_get_element_type"),
            get_output: symbol!("ov_infer_request_get_output_tensor_by_index"),
            set_input: symbol!("ov_infer_request_set_input_tensor"),
            set_output: symbol!("ov_infer_request_set_output_tensor_by_index"),
            infer: symbol!("ov_infer_request_infer"),
            start_async: symbol!("ov_infer_request_start_async"),
            wait: symbol!("ov_infer_request_wait"),
        })
    }
}

fn check(status: i32, operation: &str) -> Result<()> {
    ensure!(status == 0, "NeoArc {operation}: OpenVINO status {status}");
    Ok(())
}

#[derive(Debug)]
struct RepeatDrift { max: f32, mean: f64 }
impl std::fmt::Display for RepeatDrift {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NeoArc repeat validation failed: max={} mean={}", self.max, self.mean)
    }
}
impl std::error::Error for RepeatDrift {}

// OpenVINO Core and its plugin/DLL registration have process lifetime, like
// the ORT environment. Per-model requests, tensors and CL/GL contexts still retire.
// Keeping one bounded runtime avoids repeated plugin initialization and Intel
// Core teardown waits observed when GL-sharing and byte sessions coexist.
struct SharedRuntime { api:Api, library:Arc<Library>, core:Handle, device:CString, path:PathBuf, _dll_cookie:usize }
unsafe impl Send for SharedRuntime {}
unsafe impl Sync for SharedRuntime {}
static SHARED_RUNTIME:OnceLock<Mutex<Option<Arc<SharedRuntime>>>>=OnceLock::new();
fn shared_runtime(app_dir:&Path)->Result<Arc<SharedRuntime>> {
    let runtime=app_dir.join("backends/neoarc/runtime/openvino_c.dll");
    let mut registry=SHARED_RUNTIME.get_or_init(||Mutex::new(None)).lock().map_err(|_|anyhow!("NeoArc runtime registry poisoned"))?;
    if let Some(existing)=registry.as_ref(){ensure!(existing.path==runtime,"NeoArc runtime path changed; restart required");return Ok(existing.clone());}
        ensure!(runtime.is_file(), "NeoArc runtime is not installed");
        let wide: Vec<u16> = runtime.parent().unwrap().as_os_str().encode_wide().chain([0]).collect();
        let dll_cookie = unsafe { AddDllDirectory(PCWSTR(wide.as_ptr())) };
        ensure!(!dll_cookie.is_null(), "NeoArc runtime DLL directory unavailable");
        let raw = unsafe { winlib::Library::load_with_flags(
            &runtime,
            winlib::LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | winlib::LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
                | winlib::LOAD_LIBRARY_SEARCH_USER_DIRS,
        ) }.with_context(|| format!("load {}", runtime.display()));
        let raw = match raw {
            Ok(value) => value,
            Err(error) => { unsafe { let _ = RemoveDllDirectory(dll_cookie); } return Err(error); }
        };
        let library: Library = raw.into();
        let api = unsafe { Api::load(&library)? };
        let mut core = ptr::null_mut();
        check(unsafe { (api.core_create)(&mut core) }, "create runtime")?;
        let selected = (|| -> Result<CString> {
            let mut devices = Devices { names: ptr::null_mut(), len: 0 };
            check(unsafe { (api.available)(core, &mut devices) }, "enumerate GPUs")?;
            let mut arc_names = Vec::new();
            for index in 0..devices.len {
                let name = unsafe { CStr::from_ptr(*devices.names.add(index)) };
                if !name.to_bytes().starts_with(b"GPU") { continue; }
                let mut value = ptr::null_mut();
                let status = unsafe { (api.property)(core, name.as_ptr(),
                    c"FULL_DEVICE_NAME".as_ptr(), &mut value) };
                if status == 0 && !value.is_null() {
                    let full = unsafe { CStr::from_ptr(value) }.to_string_lossy();
                    if full.contains("Arc") { arc_names.push(name.to_owned()); }
                    unsafe { (api.string_free)(value) };
                }
            }
            unsafe { (api.available_free)(&mut devices) };
            ensure!(arc_names.len() == 1,
                "NeoArc needs one unambiguous Intel Arc GPU; found {}", arc_names.len());
            Ok(arc_names.remove(0))
        })();
        let device = match selected {
            Ok(value) => value,
            Err(error) => { unsafe { (api.core_free)(core); let _ = RemoveDllDirectory(dll_cookie); } return Err(error); }
        };

    let value=Arc::new(SharedRuntime{api,library:Arc::new(library),core,device,path:runtime,_dll_cookie:dll_cookie as usize});
    *registry=Some(value.clone());
    log::info!("neoarc-runtime-shared: device={} lifetime=process model_resources=per-session",value.device.to_string_lossy());
    Ok(value)
}

pub(crate) struct NeoArcSession {
    api: Api,
    _library: Arc<Library>,
    _runtime: Arc<SharedRuntime>,
    core: Handle,
    model: Handle,
    compiled: Handle,
    request: Handle,
    output: Handle,
    shape: Option<(usize, usize)>,
    previous: Option<CachedShape>,
    model_path: PathBuf,
    device: CString,
    remote: Option<RemoteArc>,
    remote_disabled: bool,
    pair_reuse: bool,
    image_mode: bool,
    input_channels: usize,
    rgb: Option<neoarc_rgb::RgbIo>,
    rgb_disabled: bool,
    rife_rgb:Option<neoarc_rgb::RgbIo>,
    rife_rgb_shape:Option<(usize,usize)>,
    rife_rgb_rejected:std::collections::HashSet<(usize,usize)>,
    shape_fp32:std::collections::HashSet<(usize,usize)>,
    disk_cache_allowed:bool,
    gl_rgb: Option<neoarc_rgb::RgbIo>,
    gl_rgb_key: Option<(isize,isize)>,
    gl_rgb_disabled: bool,
    image_merged_model:Option<PathBuf>,
    compact_plan:Option<Arc<neoarc_graph::CompactPlan>>,
    compact_library:PathBuf,
    channel_select_plan:Option<neoarc_graph::ChannelSelectPlan>,
    channel_select_shape:Option<(usize,usize)>,
    gl_rgb_reference:Option<neoarc_rgb::RgbIo>,
    image_merge_rejected:std::collections::HashSet<(usize,usize)>,
    sisr_speed_qualified:std::collections::HashSet<(usize,usize,bool,usize)>,
    gl_rgb_shape_key: Option<(usize,usize)>,
    precision_f32: bool,
    stable_model: Option<PathBuf>,
    stable_cache: PathBuf,
    model_identity:PathBuf,
    validated_rife_shapes: std::collections::HashSet<(usize, usize)>,
    rife_boost_source: Option<(PathBuf,PathBuf)>,
    // v983: the boost head is the plain Lite final stage (one-kernel candidate).
    rife_boost_native_head: bool,
}

pub(crate) enum ImageOutput<'a> { F16(&'a [f16]), F32(&'a [f32]) }

/// v974: canonical-route TemporalFix output (planar RGB, f32) for comparisons.
pub fn diagnostic_temporal_output(app_dir:&Path,model:&Path,frames:usize,w:usize,h:usize,packed:&[f16],repeats:usize)->Result<(Vec<f32>,f64)> {
    let mut session=NeoArcSession::load_temporal(app_dir,model,frames)?;
    let mut collect=|session:&mut NeoArcSession|session.run_image_with_output(w,h,packed,|ow,oh,o|{
        ensure!((ow,oh)==(w,h),"temporal diagnostic extent");
        Ok(match o{ImageOutput::F16(v)=>v.iter().map(|x|x.to_f32()).collect::<Vec<f32>>(),ImageOutput::F32(v)=>v.to_vec()})
    });
    let mut out=collect(&mut session)?;for _ in 0..2{out=collect(&mut session)?;}
    let started=std::time::Instant::now();for _ in 0..repeats.max(1){out=collect(&mut session)?;}
    Ok((out,started.elapsed().as_secs_f64()*1000.0/repeats.max(1) as f64))
}

pub fn diagnostic_profile_model(app_dir:&Path,model:&Path,w:usize,h:usize,channels:usize,frames:&[&[u8]],profile:bool)->Result<serde_json::Value> {
    NeoArcSession::load(app_dir,model)?.diagnostic_remote_profile(w,h,channels,frames,profile)
}

// Owns only OpenCL resources; the renderer owns and fences all GL textures.
pub(crate) struct GlInterpIo { io: neoarc_rgb::RgbIo }
impl GlInterpIo {
    pub(crate) fn begin_pair(&mut self){self.io.begin_interp_pair();}
    pub(crate) fn prepare_pair(&mut self,inputs:&[(u32,u64)],outputs:&[(u32,u64)])->Result<()>{self.io.prepare_gl_pair(inputs,outputs)}
    /// v985: endpoints as GL pixel buffers (RGBA FP16) instead of GL images.
    pub(crate) fn prepare_pair_buffers(&mut self,buffers:&[u32],outputs:&[(u32,u64)])->Result<()>{self.io.prepare_gl_pair_buffers(buffers,outputs)}
    pub(crate) fn supports_buffer_input(&self)->bool{self.io.supports_buffer_input()}
    /// v985b: GL owner thread, after the copy fence: move the endpoints into CL memory.
    pub(crate) fn stage_pair_buffers(&mut self)->Result<bool>{self.io.stage_pair_buffers()}
    pub(crate) fn run(&mut self,frames:&[u32],output:u32,phase:f32)->Result<[f64;3]> {
        self.io.run_gl_interp(frames,output,phase)
    }
    /// v963: choose the boost graphs for the next prepared pair (if built).
    pub(crate) fn set_boost(&mut self,on:bool){self.io.set_boost(on);}
    pub(crate) fn has_boost(&self)->bool{self.io.has_boost()}
}

struct RemoteArc {
    api: Api,
    cl: ClApi,
    _cl_library: Library,
    context: ClHandle,
    queue: ClHandle,
    remote_context: Handle,
    model: Handle,
    compiled: Handle,
    request: Handle,
    input_mem: ClHandle,
    output_mem: ClHandle,
    input_tensor: Handle,
    output_tensor: Handle,
    output: Vec<f16>,
    shape: Option<(usize, usize)>,
    run_count: u64,
    pair_input_ready: bool,
}

// The stage owns this request and invokes it serially on its worker thread.
unsafe impl Send for NeoArcSession {}

impl RemoteArc {
    fn new(api: Api, core: Handle, device: &CStr) -> Result<Self> {
        Self::new_sharing(api,core,device,None)
    }
    fn new_sharing(api: Api, core: Handle, device: &CStr, gl:Option<(isize,isize)>) -> Result<Self> {
        let cl_library = unsafe { Library::new("OpenCL.dll") }?;
        let cl = unsafe { ClApi::load(&cl_library)? };
        let mut platform_count = 0u32;
        cl_check(unsafe { (cl.platforms)(0, ptr::null_mut(), &mut platform_count) },
            "enumerate OpenCL platforms")?;
        let mut platforms = vec![ptr::null_mut(); platform_count as usize];
        cl_check(unsafe { (cl.platforms)(platform_count, platforms.as_mut_ptr(), ptr::null_mut()) },
            "read OpenCL platforms")?;
        let mut arc_devices = Vec::new();
        let mut arc_platform = ptr::null_mut();
        for platform in platforms {
            let mut count = 0u32;
            if unsafe { (cl.devices)(platform, 4, 0, ptr::null_mut(), &mut count) } != 0 {
                continue;
            }
            let mut devices = vec![ptr::null_mut(); count as usize];
            cl_check(unsafe { (cl.devices)(platform, 4, count, devices.as_mut_ptr(),
                ptr::null_mut()) }, "read OpenCL GPUs")?;
            for device in devices {
                let mut size = 0usize;
                if unsafe { (cl.device_info)(device, 0x102B, 0, ptr::null_mut(), &mut size) } != 0 {
                    continue;
                }
                let mut name = vec![0u8; size];
                if unsafe { (cl.device_info)(device, 0x102B, size,
                    name.as_mut_ptr().cast(), ptr::null_mut()) } == 0
                    && String::from_utf8_lossy(&name).contains("Arc") {
                    arc_devices.push(device);
                    arc_platform = platform;
                }
            }
        }
        ensure!(arc_devices.len() == 1,
            "NeoArc remote path needs one Arc OpenCL GPU, found {}", arc_devices.len());
        let mut props=Vec::new();
        if let Some((rc,dc))=gl {
            props=vec![0x1084,arc_platform as isize,0x2008,rc,0x200b,dc,0];
            let extension:unsafe extern "system" fn(ClHandle,*const c_char)->*mut c_void=unsafe{*cl_library.get(b"clGetExtensionFunctionAddressForPlatform\0")?};
            let address=unsafe{extension(arc_platform,c"clGetGLContextInfoKHR".as_ptr())};
            ensure!(!address.is_null(),"CL/GL query unavailable");
            let info:unsafe extern "system" fn(*const isize,u32,usize,*mut c_void,*mut usize)->i32=unsafe{std::mem::transmute(address)};
            let mut actual:ClHandle=ptr::null_mut();
            cl_check(unsafe{info(props.as_ptr(),0x2006,std::mem::size_of::<ClHandle>(),(&mut actual as *mut ClHandle).cast(),ptr::null_mut())},"query GL-sharing device")?;
            ensure!(actual==arc_devices[0],"current OpenGL context is not on the selected Arc GPU");
        }
        let mut status = 0;
        let context = unsafe { (cl.context_create)(if props.is_empty(){ptr::null()}else{props.as_ptr()}, 1,
            arc_devices.as_ptr(), ptr::null_mut(), ptr::null_mut(), &mut status) };
        cl_check(status, "create Arc OpenCL context")?;
        ensure!(!context.is_null(), "NeoArc OpenCL context is null");
        let mut result = Self { api, cl, _cl_library: cl_library, context,
            queue: ptr::null_mut(), remote_context: ptr::null_mut(),
            model: ptr::null_mut(), compiled: ptr::null_mut(), request: ptr::null_mut(),
            input_mem: ptr::null_mut(), output_mem: ptr::null_mut(),
            input_tensor: ptr::null_mut(), output_tensor: ptr::null_mut(),
            output: Vec::new(), shape: None, run_count: 0, pair_input_ready: false };
        let init = (|| -> Result<()> {
            let profile=std::env::var("CHIDESCALER_NEOARC_PROFILE").ok().as_deref()==Some("on");
            // v995: the GL interpolation queue may ask for a high scheduling
            // priority (cl_khr_priority_hints) so the shared motion pass is
            // not delayed by other GPU clients (browser decode, DWM). Any
            // failure keeps the ordinary queue.
            let priority=if gl.is_some(){neoarc_queue_priority()}else{None};
            if let Some(level)=priority {
                let create:Option<unsafe extern "system" fn(ClHandle,ClHandle,*const u64,*mut i32)->ClHandle>=
                    unsafe{result._cl_library.get(b"clCreateCommandQueueWithProperties\0").ok().map(|s:libloading::Symbol<unsafe extern "system" fn(ClHandle,ClHandle,*const u64,*mut i32)->ClHandle>|*s)};
                if let Some(create)=create {
                    let mut props:Vec<u64>=vec![0x1096,level];
                    if profile{props.extend_from_slice(&[0x1093,2]);}
                    props.push(0);
                    let mut s=0;let queue=unsafe{create(result.context,arc_devices[0],props.as_ptr(),&mut s)};
                    if s==0&&!queue.is_null(){result.queue=queue;log::info!("neoarc-queue-priority: level={} result=applied",level);}
                    else{log::info!("neoarc-queue-priority: level={} result=unsupported status={} action=default-queue",level,s);}
                }
            }
            if result.queue.is_null(){
            result.queue = unsafe { (result.cl.queue_create)(result.context,
                arc_devices[0], if profile{2}else{0}, &mut status) };
            }
            cl_check(status, "create Arc OpenCL queue")?;
            ensure!(!result.queue.is_null(), "NeoArc OpenCL queue is null");
            let properties = [
                Property { key: c"CONTEXT_TYPE".as_ptr(), value: c"OCL".as_ptr().cast() },
                Property { key: c"OCL_CONTEXT".as_ptr(), value: result.context.cast() },
                Property { key: c"OCL_QUEUE".as_ptr(), value: result.queue.cast() },
            ];
            check(unsafe { (result.api.context_create)(core, device.as_ptr(),
                properties.len(), properties.as_ptr(), &mut result.remote_context) },
                "create Arc OpenVINO remote context")?;
            Ok(())
        })();
        init?;
        Ok(result)
    }

    fn retire_shape(&mut self) {
        unsafe {
            if !self.request.is_null() { (self.api.request_free)(self.request); self.request = ptr::null_mut(); }
            if !self.input_tensor.is_null() { (self.api.tensor_free)(self.input_tensor); self.input_tensor = ptr::null_mut(); }
            if !self.output_tensor.is_null() { (self.api.tensor_free)(self.output_tensor); self.output_tensor = ptr::null_mut(); }
            if !self.compiled.is_null() { (self.api.compiled_free)(self.compiled); self.compiled = ptr::null_mut(); }
            if !self.model.is_null() { (self.api.model_free)(self.model); self.model = ptr::null_mut(); }
            if !self.input_mem.is_null() { let _ = (self.cl.buffer_free)(self.input_mem); self.input_mem = ptr::null_mut(); }
            if !self.output_mem.is_null() { let _ = (self.cl.buffer_free)(self.output_mem); self.output_mem = ptr::null_mut(); }
        }
        self.output.clear();
        self.shape = None;
        self.pair_input_ready = false;
    }

    fn prepare_shape(&mut self, model_path: &Path, width: usize, height: usize, core: Handle) -> Result<()> {
        if self.shape == Some((width, height)) { return Ok(()); }
        self.retire_shape();
        let plane = width.checked_mul(height).ok_or_else(|| anyhow!("NeoArc plane overflow"))?;
        let mut status = 0;
        self.input_mem = unsafe { (self.cl.buffer_create)(self.context, 1,
            7 * plane * 2, ptr::null_mut(), &mut status) };
        cl_check(status, "create Arc input buffer")?;
        self.output_mem = unsafe { (self.cl.buffer_create)(self.context, 1,
            3 * plane * 2, ptr::null_mut(), &mut status) };
        cl_check(status, "create Arc output buffer")?;
        let tensor_props = |memory: ClHandle| [
            Property { key: c"SHARED_MEM_TYPE".as_ptr(), value: c"OCL_BUFFER".as_ptr().cast() },
            Property { key: c"MEM_HANDLE".as_ptr(), value: memory.cast() },
        ];
        let mut in_dims = [1i64, 7, height as i64, width as i64];
        let props = tensor_props(self.input_mem);
        check(unsafe { (self.api.remote_tensor_create)(self.remote_context, 3,
            Shape { rank: 4, dims: in_dims.as_mut_ptr() }, props.len(), props.as_ptr(),
            &mut self.input_tensor) }, "wrap Arc input buffer")?;
        let mut out_dims = [1i64, 3, height as i64, width as i64];
        let props = tensor_props(self.output_mem);
        check(unsafe { (self.api.remote_tensor_create)(self.remote_context, 3,
            Shape { rank: 4, dims: out_dims.as_mut_ptr() }, props.len(), props.as_ptr(),
            &mut self.output_tensor) }, "wrap Arc output buffer")?;
        let path = CString::new(model_path.to_string_lossy().as_bytes())?;
        check(unsafe { (self.api.model_read)(core, path.as_ptr(), ptr::null(),
            &mut self.model) }, "read Arc remote model")?;
        let mut shape = PartialShape { rank: Dimension { min: 0, max: 0 }, dims: ptr::null_mut() };
        check(unsafe { (self.api.shape_create)(4, in_dims.as_ptr(), &mut shape) },
            "create Arc remote shape")?;
        let reshape_status = unsafe { (self.api.model_reshape)(self.model, shape) };
        unsafe { (self.api.shape_free)(&mut shape) };
        check(reshape_status, "specialize Arc remote model")?;
        let hint = Property { key: c"PERFORMANCE_HINT".as_ptr(), value: c"LATENCY".as_ptr().cast() };
        check(unsafe { (self.api.compile_remote)(core, self.model, self.remote_context, 1,
            &hint, &mut self.compiled) }, "compile Arc remote model")?;
        check(unsafe { (self.api.request_create)(self.compiled, &mut self.request) },
            "create Arc remote request")?;
        check(unsafe { (self.api.set_input)(self.request, self.input_tensor) },
            "bind Arc remote input")?;
        check(unsafe { (self.api.set_output)(self.request, 0, self.output_tensor) },
            "bind Arc remote output")?;
        self.output.resize(3 * plane, f16::ZERO);
        self.shape = Some((width, height));
        self.run_count = 0;
        self.pair_input_ready = false;
        Ok(())
    }

    fn run(&mut self, model_path: &Path, core: Handle, width: usize, height: usize,
        input: &[f16], pair_reuse: bool) -> Result<&[f16]> {
        self.prepare_shape(model_path, width, height, core)?;
        let plane = width * height;
        let started = std::time::Instant::now();
        let (offset, count) = if pair_reuse && self.pair_input_ready {
            (6 * plane, plane)
        } else {
            (0, 7 * plane)
        };
        cl_check(unsafe { (self.cl.write)(self.queue, self.input_mem, 1, offset * 2,
            count * 2, input[offset..].as_ptr().cast(), 0, ptr::null(), ptr::null_mut()) },
            "upload Arc input")?;
        self.pair_input_ready = pair_reuse;
        let upload_ms = started.elapsed().as_secs_f64() * 1000.0;
        check(unsafe { (self.api.infer)(self.request) }, "infer Arc remote model")?;
        let infer_ms = started.elapsed().as_secs_f64() * 1000.0 - upload_ms;
        cl_check(unsafe { (self.cl.read)(self.queue, self.output_mem, 1, 0,
            3 * plane * 2, self.output.as_mut_ptr().cast(), 0,
            ptr::null(), ptr::null_mut()) }, "download Arc output")?;
        self.run_count += 1;
        if self.run_count <= 12 || self.run_count % 120 == 0 {
            log::info!("neoarc-remote-profile: shape={}x{} upload_ms={:.3} infer_ms={:.3} download_ms={:.3} reuse_rgb={}",
                width, height, upload_ms, infer_ms,
                started.elapsed().as_secs_f64() * 1000.0 - upload_ms - infer_ms,
                offset != 0);
        }
        Ok(&self.output)
    }
}

impl Drop for RemoteArc {
    fn drop(&mut self) {
        self.retire_shape();
        unsafe {
            if !self.remote_context.is_null() { (self.api.context_free)(self.remote_context); }
            if !self.queue.is_null() { let _ = (self.cl.queue_free)(self.queue); }
            if !self.context.is_null() { let _ = (self.cl.context_free)(self.context); }
        }
    }
}

type WarmGlKey=(usize,PathBuf,Option<PathBuf>,usize,usize,isize,isize,bool);
struct WarmGl{key:WarmGlKey,io:GlInterpIo}
fn warm_gl_cache()->&'static std::sync::Mutex<std::collections::VecDeque<WarmGl>>{
    static CACHE:std::sync::OnceLock<std::sync::Mutex<std::collections::VecDeque<WarmGl>>>=std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}
/// v987: last OpenCL driver version seen on the GL interop device (process copy
/// of `cache/neoarc/validated-v1/driver.txt`).
fn known_driver_cache()->&'static std::sync::Mutex<Option<String>>{
    static DRIVER:std::sync::OnceLock<std::sync::Mutex<Option<String>>>=std::sync::OnceLock::new();
    DRIVER.get_or_init(Default::default)
}
pub(crate) fn warm_gl_budget()->bool {
    use windows::{core::Interface,Win32::Graphics::Dxgi::{CreateDXGIFactory1,IDXGIFactory1,IDXGIAdapter3,DXGI_MEMORY_SEGMENT_GROUP_LOCAL}};
    let result=(||->windows::core::Result<bool>{unsafe{
        let factory:IDXGIFactory1=CreateDXGIFactory1()?;let mut usage=Vec::new();let mut index=0;
        while let Ok(adapter)=factory.EnumAdapters1(index){index+=1;let desc=adapter.GetDesc1()?;
            if desc.VendorId!=0x8086||desc.DedicatedVideoMemory<1024*1024*1024{continue;}let adapter:IDXGIAdapter3=adapter.cast()?;
            let mut info=windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();adapter.QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&mut info)?;
            if desc.DedicatedVideoMemory>0{usage.push((info.CurrentUsage,info.Budget));}
        }
        log::info!("neoarc-warm-memory-budget: discrete_intel_adapters={} usage_and_budget_bytes={usage:?}",usage.len());
        Ok(usage.len()==1&&usage[0].1>0&&usage[0].0<usage[0].1*65/100)
    }})();result.unwrap_or(false)
}
fn compilation_cache(root:&Path)->Option<CString> {
    if std::env::var("CHIDESCALER_NEOARC_COMPILE_CACHE").ok().as_deref()==Some("off"){return None;}
    let path=std::env::var_os("CHIDESCALER_NEOARC_COMPILE_CACHE_DIR").map(PathBuf::from)
        .unwrap_or_else(||root.parent().unwrap_or(root).join("compiled-v1"));
    match std::fs::create_dir_all(&path).map_err(anyhow::Error::from)
        .and_then(|_|CString::new(path.to_string_lossy().as_bytes()).map_err(Into::into)) {
        Ok(path)=>Some(path),Err(error)=>{log::warn!("neoarc-compile-cache-unavailable: reason={error:#} action=uncached");None}
    }
}

/// v973: carry a model's FP16 stabilization decision over to a derived graph.
/// When `original` already needed the stabilized graph (its cache entry
/// exists), the same treatment is prepared for `derived`, so it never starts
/// on the plain FP16 graph that was found to drift. Returns whether it did.
pub(super) fn inherit_stable_fp16(app_dir: &Path, original: &Path, derived: &Path) -> Result<bool> {
    let cache = app_dir.join("cache/neoarc/stable-fp16-v1");
    if !neoarc_graph::cache_path(&cache, original)?.is_file() {
        return Ok(false);
    }
    neoarc_graph::prepare(&cache, derived)?;
    Ok(true)
}

pub(super) fn cache_rife_bridge(app_dir: &Path, original: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let cache = app_dir.join("cache/neoarc/rife-half-bridge-v1");
    let derived = neoarc_graph::cache_path(&cache, original)?;
    neoarc_graph::write_derived(&derived, bytes)?;
    Ok(derived)
}

impl NeoArcSession {
    fn retire_rife_rgb(&mut self){self.rife_rgb.take();self.rife_rgb_shape=None;self.rife_rgb_rejected.clear();}
    pub(crate) fn reset_rife_validation(&mut self) {
        if !self.image_mode&&self.input_channels==7 {self.validated_rife_shapes.clear();self.rife_rgb_shape=None;self.rife_rgb_rejected.clear();}
    }
    pub(crate) fn set_pair_reuse(&mut self, enabled: bool) {
        self.pair_reuse = enabled;
        if let Some(remote) = self.remote.as_mut() {
            remote.pair_input_ready = false;
        }
    }

    /// v963: RIFE boost graphs (built by the host from the original ONNX) for
    /// the next GL interpolation bridge. Written next to the stable cache.
    pub(crate) fn set_rife_boost_graphs(&mut self,w:usize,h:usize,trunk:&[u8],head:&[u8],native_head:bool)->Result<()> {
        let dir=self.stable_cache.parent().unwrap_or(&self.stable_cache).join("rife-boost-v1");
        std::fs::create_dir_all(&dir)?;
        let stem=self.model_identity.file_stem().map(|s|s.to_string_lossy().into_owned()).unwrap_or_else(||"model".into());
        let mut paths=Vec::new();
        for(part,bytes)in[("trunk",trunk),("head",head)]{
            let path=dir.join(format!("{stem}-{w}x{h}-{part}.onnx"));
            if std::fs::read(&path).ok().as_deref()!=Some(bytes){std::fs::write(&path,bytes)?;}
            paths.push(path);
        }
        self.rife_boost_source=Some((paths.remove(0),paths.remove(0)));self.rife_boost_native_head=native_head;Ok(())
    }
    pub(crate) fn clear_rife_boost(&mut self){self.rife_boost_source=None;self.rife_boost_native_head=false;}

    pub(crate) fn load(app_dir: &Path, model_path: &Path) -> Result<Self> {
        let owner=shared_runtime(app_dir)?;
        let api=owner.api;let core=owner.core;let device=owner.device.clone();let library=owner.library.clone();
        let stable_cache = app_dir.join("cache/neoarc/stable-fp16-v1");
        let disk_cache_allowed=match neoarc_graph::requires_uncached_gpu_model(model_path){Ok(required)=>!required,Err(error)=>{log::warn!("neoarc-model-cache-inspection: reason={error:#} action=uncached-model");false}};
        if !disk_cache_allowed{log::info!("neoarc-model-cache-policy: model={} disk_cache=false reason=reduction-or-function-graph precision=unchanged",model_path.display());}
        let precision_f32 = std::env::var("CHIDESCALER_NEOARC_PRECISION").ok().as_deref() == Some("f32");
        let stable_model = if precision_f32 { None } else {
            neoarc_graph::cache_path(&stable_cache, model_path).ok().filter(|path| path.is_file())
        };
        if stable_model.is_some() { log::info!("neoarc-stable-fp16-cache-hit: model={}", model_path.display()); }
        Ok(Self { api, _library: library, _runtime: owner, core, model: ptr::null_mut(),
            compiled: ptr::null_mut(), request: ptr::null_mut(), output: ptr::null_mut(),
            shape: None, previous: None, model_path: model_path.to_owned(), device,
            remote: None, remote_disabled: false, pair_reuse: false, image_mode: false, input_channels: 7,
            precision_f32, stable_model,model_identity:neoarc_graph::cache_path(&stable_cache,model_path)?, stable_cache,
            rgb: None, rgb_disabled: false,
            rife_rgb:None,rife_rgb_shape:None,rife_rgb_rejected:std::collections::HashSet::new(),shape_fp32:std::collections::HashSet::new(),
            compact_plan:None,compact_library:app_dir.join("backends/neoarc/runtime/neoarc_compact.dll"),image_merged_model:None,channel_select_plan:None,channel_select_shape:None,gl_rgb_reference:None,image_merge_rejected:std::collections::HashSet::new(),sisr_speed_qualified:std::collections::HashSet::new(),
            disk_cache_allowed,gl_rgb: None, gl_rgb_key: None, gl_rgb_disabled: false, gl_rgb_shape_key: None,
            validated_rife_shapes: std::collections::HashSet::new(), rife_boost_source: None, rife_boost_native_head: false })
    }

    pub(crate) fn load_image(app_dir: &Path, model_path: &Path) -> Result<Self> {
        let mut session = Self::load(app_dir, model_path)?;
        session.image_mode = true;
        session.input_channels = 3;
        if !session.precision_f32&&session.stable_model.is_none()&&session.compact_library.is_file()&&std::env::var("CHIDESCALER_NEOARC_COMPACT_NATIVE").ok().as_deref()!=Some("off"){
            match neoarc_graph::discover_compact(model_path){Ok(plan)=>session.compact_plan=plan.map(Arc::new),Err(e)=>log::warn!("neoarc-compact-discovery-fallback: reason={e:#} action=original-gpu-graph")}
        }
        if !session.precision_f32&&session.stable_model.is_none(){
            let cache=session.stable_cache.parent().unwrap_or(&session.stable_cache);
            if std::env::var("CHIDESCALER_NEOARC_DW_MERGE").ok().as_deref()!=Some("off"){
                match neoarc_graph::prepare_depthwise_branches(cache,model_path){
                    Ok(path)=>session.image_merged_model=path,
                    Err(error)=>log::warn!("neoarc-sisr-optimized-graph-fallback: reason={error:#} action=original-gpu-graph"),
                }
            }
            if session.image_merged_model.is_none()&&std::env::var("CHIDESCALER_NEOARC_CONCAT_PREFIX").ok().as_deref()!=Some("off"){
                match neoarc_graph::prepare_concat_prefix(cache,model_path){
                    Ok(path)=>session.image_merged_model=path,
                    Err(error)=>log::warn!("neoarc-concat-prefix-fallback: reason={error:#} action=original-gpu-graph"),
                }
            }
            if session.image_merged_model.is_none()&&std::env::var("CHIDESCALER_NEOARC_SPAN_GATE").ok().as_deref()!=Some("off"){
                match neoarc_graph::prepare_tanh_gates(cache,model_path){
                    Ok(path)=>session.image_merged_model=path,
                    Err(error)=>log::warn!("neoarc-sisr-optimized-graph-fallback: reason={error:#} action=original-gpu-graph"),
                }
            }
        }
        Ok(session)
    }

    pub(crate) fn load_temporal(app_dir: &Path, model_path: &Path, frames: usize) -> Result<Self> {
        ensure!(frames >= 3, "NeoArc temporal history must contain at least three frames");
        // v974: flow channel split/merge removed (same arithmetic per element).
        // TemporalFix S2 640x480 16.6 -> 10.0 ms, 1280x720 73 -> 46 ms.
        let derived = if std::env::var("CHIDESCALER_NEOARC_FLOW_CHANNELS").ok().as_deref() == Some("off") {
            None
        } else {
            match neoarc_graph::prepare_flow_channels(&app_dir.join("cache/neoarc"), model_path) {
                Ok(path) => path,
                Err(error) => { log::warn!("neoarc-flow-channels-fallback: model={} reason={error:#} action=original-gpu-graph", model_path.display()); None }
            }
        };
        let mut session = match derived.as_deref() {
            Some(path) => match Self::load(app_dir, path) {
                Ok(session) => session,
                Err(error) => {
                    log::warn!("neoarc-flow-channels-fallback: model={} reason={error:#} action=original-gpu-graph", model_path.display());
                    Self::load(app_dir, model_path)?
                }
            },
            None => Self::load(app_dir, model_path)?,
        };
        session.image_mode = true;
        session.input_channels = frames.checked_mul(3).ok_or_else(|| anyhow!("temporal channel overflow"))?;
        if !session.precision_f32&&session.stable_model.is_none()&&std::env::var("CHIDESCALER_NEOARC_TEMPORAL_GATHER").ok().as_deref()==Some("on"){
            match neoarc_graph::ChannelSelectPlan::discover(session.stable_cache.parent().unwrap_or(&session.stable_cache),model_path){
                Ok(plan)=>session.channel_select_plan=plan,
                Err(error)=>log::warn!("neoarc-channel-select-analysis-fallback: reason={error:#} action=original-gpu-graph"),
            }
        }
        Ok(session)
    }

    pub(crate) fn load_drba(app_dir: &Path, model_path: &Path) -> Result<Self> {
        let mut session = Self::load(app_dir, model_path)?;
        session.input_channels = 14;
        Ok(session)
    }

    fn retire_shape(&mut self) {
        unsafe {
            if !self.output.is_null() { (self.api.tensor_free)(self.output); self.output = ptr::null_mut(); }
            if !self.request.is_null() { (self.api.request_free)(self.request); self.request = ptr::null_mut(); }
            if !self.compiled.is_null() { (self.api.compiled_free)(self.compiled); self.compiled = ptr::null_mut(); }
            if !self.model.is_null() { (self.api.model_free)(self.model); self.model = ptr::null_mut(); }
        }
        self.shape = None;
    }

    fn retire_previous(&mut self) {
        if let Some(previous) = self.previous.take() {
            unsafe {
                (self.api.tensor_free)(previous.output);
                (self.api.request_free)(previous.request);
                (self.api.compiled_free)(previous.compiled);
                (self.api.model_free)(previous.model);
            }
        }
    }

    fn prepare_shape(&mut self, width: usize, height: usize) -> Result<()> {
        if self.shape == Some((width, height)) { return Ok(()); }
        if self.previous.as_ref().is_some_and(|previous| previous.size == (width, height)) {
            let previous = self.previous.take().unwrap();
            let old = self.shape.map(|size| CachedShape { size, model: self.model,
                compiled: self.compiled, request: self.request, output: self.output });
            self.model = previous.model;
            self.compiled = previous.compiled;
            self.request = previous.request;
            self.output = previous.output;
            self.shape = Some(previous.size);
            self.previous = old;
            return Ok(());
        }
        self.retire_previous();
        if let Some(size) = self.shape.take() {
            self.previous = Some(CachedShape { size, model: self.model,
                compiled: self.compiled, request: self.request, output: self.output });
            self.model = ptr::null_mut();
            self.compiled = ptr::null_mut();
            self.request = ptr::null_mut();
            self.output = ptr::null_mut();
        } else {
            self.retire_shape();
        }
        let path = CString::new((if self.precision_f32{&self.model_path}else{self.stable_model.as_deref().unwrap_or(&self.model_path)}).to_string_lossy().as_bytes())?;
        check(unsafe { (self.api.model_read)(self.core, path.as_ptr(), ptr::null(),
            &mut self.model) }, "read ONNX model")?;
        let dims = [1i64, self.input_channels as i64, height as i64, width as i64];
        let mut shape = PartialShape { rank: Dimension { min: 0, max: 0 }, dims: ptr::null_mut() };
        check(unsafe { (self.api.shape_create)(4, dims.as_ptr(), &mut shape) }, "shape input")?;
        let status = unsafe { (self.api.model_reshape)(self.model, shape) };
        unsafe { (self.api.shape_free)(&mut shape) };
        self.api.check_detail(status, "specialize ONNX shape")?;
        let hint = std::env::var("CHIDESCALER_NEOARC_HINT").unwrap_or_default();
        let mut properties = Vec::new();
        if matches!(hint.as_str(), "LATENCY" | "THROUGHPUT") {
            properties.push(Property { key: c"PERFORMANCE_HINT".as_ptr(),
                value: if hint == "LATENCY" { c"LATENCY".as_ptr() } else { c"THROUGHPUT".as_ptr() }.cast() });
        }
        if self.precision_f32 {
            properties.push(Property { key: c"INFERENCE_PRECISION_HINT".as_ptr(), value: c"f32".as_ptr().cast() });
        }
        let cache=if self.disk_cache_allowed{compilation_cache(&self.stable_cache)}else{None};
        if self.api.compile_props.is_some(){if let Some(cache)=cache.as_ref(){properties.push(Property{key:c"CACHE_DIR".as_ptr(),value:cache.as_ptr().cast()});}}
        let compile_started=std::time::Instant::now();
        let status = if let Some(compile_props) = self.api.compile_props {
            unsafe { compile_props(self.core, self.model, self.device.as_ptr(), properties.len(),
                properties.as_ptr(), &mut self.compiled) }
        } else {
            ensure!(!self.precision_f32, "NeoArc runtime lacks precision recovery properties");
            unsafe { (self.api.compile)(self.core, self.model, self.device.as_ptr(), 0, &mut self.compiled) }
        };
        let status=if status!=0&&cache.is_some()&&self.api.compile_props.is_some(){
            if !self.compiled.is_null(){unsafe{(self.api.compiled_free)(self.compiled)};self.compiled=ptr::null_mut();}
            properties.retain(|p|unsafe{CStr::from_ptr(p.key)}!=c"CACHE_DIR");
            log::warn!("neoarc-compile-cache-retry: route=canonical action=uncached");
            unsafe{self.api.compile_props.unwrap()(self.core,self.model,self.device.as_ptr(),properties.len(),properties.as_ptr(),&mut self.compiled)}
        }else{status};
        check(status, "compile Arc model")?;
        log::info!("neoarc-compile: route=canonical shape={}x{} cache={} compile_ms={:.3}",width,height,cache.is_some(),compile_started.elapsed().as_secs_f64()*1000.0);
        check(unsafe { (self.api.request_create)(self.compiled, &mut self.request) },
            "create inference request")?;
        if self.image_mode {
            self.shape = Some((width, height));
            return Ok(());
        }
        let mut out_dims = [1i64, 3, height as i64, width as i64];
        check(unsafe { (self.api.tensor_create)(3, Shape { rank: 4,
            dims: out_dims.as_mut_ptr() }, &mut self.output) }, "allocate FP16 output")?;
        check(unsafe { (self.api.set_output)(self.request, 0, self.output) }, "bind output")?;
        self.shape = Some((width, height));
        Ok(())
    }

    fn requires_shape_fp32(&self,w:usize,h:usize)->bool {
        self.shape_fp32.contains(&(w,h))||(!self.image_mode&&self.input_channels==7&&self.stable_model.is_none()&&(w,h)==(1024,640))
    }
    pub(crate) fn precision_for_shape(&self,w:usize,h:usize)->&'static str {
        if self.precision_f32||self.requires_shape_fp32(w,h){"f32"}else if self.stable_model.is_some(){"stable-fp16"}else{"fp16"}
    }
    pub(crate) fn run_with_output<R>(&mut self,width:usize,height:usize,input:&[f16],consume:impl FnOnce(&[f16])->Result<R>)->Result<R>{
        let scoped=self.requires_shape_fp32(width,height)&&!self.precision_f32;
        if scoped {self.precision_f32=true;}
        let result=self.run_with_output_impl(width,height,input,consume);
        if scoped {self.precision_f32=false;}
        result
    }

    fn run_with_output_impl<R>(
        &mut self, width: usize, height: usize, input: &[f16],
        consume: impl FnOnce(&[f16]) -> Result<R>,
    ) -> Result<R> {
        let mut consume = Some(consume);
        let mut result = self.run_with_output_once(width, height, input, |output| consume.take().unwrap()(output));
        let unstable = |result: &Result<R>| result.as_ref().err()
            .is_some_and(|error| error.downcast_ref::<RepeatDrift>().is_some());
        if unstable(&result) && !self.precision_f32 && self.stable_model.is_none() {
            match neoarc_graph::prepare(&self.stable_cache, &self.model_path) {
                Ok(path) => {
                    log::warn!("neoarc-precision-recovery: model={} shape={}x{} action=retry-stable-fp16-on-same-gpu", self.model_path.display(), width, height);
                    self.retire_previous(); self.retire_shape();
                    self.remote.take(); self.remote_disabled = true;
                    self.retire_rife_rgb();
                    self.stable_model = Some(path);
                    self.validated_rife_shapes.clear();
                    result = self.run_with_output_once(width, height, input, |output| consume.take().unwrap()(output));
                }
                Err(error) => log::warn!("neoarc-stable-fp16-unavailable: model={} reason={error:#}", self.model_path.display()),
            }
        }
        let recover = result.as_ref().err().is_some_and(|error| error.downcast_ref::<RepeatDrift>().is_some())
            && !self.precision_f32 && self.api.compile_props.is_some();
        if recover {
            log::warn!("neoarc-precision-recovery: model={} shape={}x{} reason={} action=retry-f32-on-same-gpu",
                self.model_path.display(), width, height, result.as_ref().err().unwrap());
            self.retire_previous();
            self.retire_shape();
            self.remote.take();
            self.remote_disabled = true;
            self.retire_rife_rgb();
            self.precision_f32 = true;
            self.stable_model = None;
            self.validated_rife_shapes.clear();
            return self.run_with_output_once(width, height, input, |output| consume.take().unwrap()(output));
        }
        result
    }

    fn validate_phase_repeats(&self,width:usize,height:usize,input:&[f16],original:Handle)->Result<()> {
        let plane=width*height;let mut probe=input.to_vec();
        let mut dims=[1,7,height as i64,width as i64];let mut tensor=ptr::null_mut();
        check(unsafe{(self.api.tensor_from_host)(3,Shape{rank:4,dims:dims.as_mut_ptr()},probe.as_mut_ptr().cast(),&mut tensor)},"wrap phase validation input")?;
        let result=(||->Result<()> {
            check(unsafe{(self.api.set_input)(self.request,tensor)},"bind phase validation input")?;
            for phase in [0.2f32,0.5,0.8] {
                probe[6*plane..].fill(f16::from_f32(phase));
                let mut first=Vec::new();let mut max=0.0f32;let mut sum=0.0f64;
                for iteration in 0..2 {
                    check(unsafe{(self.api.infer)(self.request)},"infer phase validation")?;
                    let mut data=ptr::null_mut();check(unsafe{(self.api.tensor_data)(self.output,&mut data)},"read phase validation")?;
                    ensure!(!data.is_null(),"null phase validation output");
                    let values=unsafe{std::slice::from_raw_parts(data.cast::<f16>(),3*plane)};
                    if iteration==0 {first=values.to_vec();continue;}
                    for(a,b)in first.iter().zip(values){let(a,b)=(a.to_f32(),b.to_f32());ensure!(a.is_finite()&&b.is_finite(),"nonfinite phase validation output");let d=(a-b).abs();max=max.max(d);sum+=d as f64;}
                }
                let mean=sum/(3*plane)as f64;
                log::info!("neoarc-phase-validation: model={} shape={}x{} t={} repeat_max={:.6} repeat_mean={:.8}",self.model_path.display(),width,height,phase,max,mean);
                if !(max<0.25&&mean<0.001){return Err(RepeatDrift{max,mean}.into());}
            }
            Ok(())
        })();
        // Restore the real request input even on rejection, before probe storage dies.
        let restore=check(unsafe{(self.api.set_input)(self.request,original)},"restore requested phase input");
        unsafe{(self.api.tensor_free)(tensor)};
        restore?;result
    }

    fn run_with_output_once<R>(
        &mut self,
        width: usize,
        height: usize,
        input: &[f16],
        consume: impl FnOnce(&[f16]) -> Result<R>,
    ) -> Result<R> {
        let plane = width.checked_mul(height).ok_or_else(|| anyhow!("NeoArc size overflow"))?;
        ensure!(input.len() == self.input_channels * plane, "NeoArc input channel count differs from session");
        self.prepare_shape(width, height)?;
        let mut dims = [1i64, self.input_channels as i64, height as i64, width as i64];
        let mut tensor = ptr::null_mut();
        check(unsafe { (self.api.tensor_from_host)(3,
            Shape { rank: 4, dims: dims.as_mut_ptr() },
            input.as_ptr().cast_mut().cast(), &mut tensor) }, "wrap packed input")?;
        let result = (|| -> Result<R> {
            check(unsafe { (self.api.set_input)(self.request, tensor) }, "bind input")?;
            check(unsafe { (self.api.infer)(self.request) }, "infer")?;
            let mut data = ptr::null_mut();
            check(unsafe { (self.api.tensor_data)(self.output, &mut data) }, "read output")?;
            ensure!(!data.is_null(), "NeoArc returned a null output");
            if !self.validated_rife_shapes.contains(&(width, height)) {
                let mut shape = Shape { rank: 0, dims: ptr::null_mut() };
                check(unsafe { (self.api.tensor_shape)(self.output, &mut shape) }, "get RIFE output shape")?;
                let valid = shape.rank == 4 && !shape.dims.is_null()
                    && unsafe { std::slice::from_raw_parts(shape.dims, 4) } == [1, 3, height as i64, width as i64];
                unsafe { let _ = (self.api.static_shape_free)(&mut shape); }
                ensure!(valid, "NeoArc RIFE output dimensions differ from requested geometry");
                let first = unsafe { std::slice::from_raw_parts(data.cast::<f16>(), 3 * plane) }.to_vec();
                check(unsafe { (self.api.infer)(self.request) }, "validate repeated RIFE inference")?;
                check(unsafe { (self.api.tensor_data)(self.output, &mut data) }, "read repeated RIFE output")?;
                ensure!(!data.is_null(), "NeoArc repeated output is null");
                let repeated = unsafe { std::slice::from_raw_parts(data.cast::<f16>(), 3 * plane) };
                let mut max_delta = 0.0f32;
                let mut sum_delta = 0.0f64;
                for (a, b) in first.iter().zip(repeated) {
                    let (a, b) = (a.to_f32(), b.to_f32());
                    ensure!(a.is_finite() && b.is_finite(), "NeoArc RIFE nonfinite output");
                    max_delta = max_delta.max((a-b).abs());
                    sum_delta += (a-b).abs() as f64;
                }
                let mean_delta = sum_delta / first.len() as f64;
                // FP16 warp boundaries can differ at a few pixels. Reject
                // frame-wide drift as well as gross local errors, rather
                // than blacklisting a shape from a single outlier pixel.
                if !(max_delta < 0.25 && mean_delta < 0.001) {
                    return Err(RepeatDrift { max: max_delta, mean: mean_delta }.into());
                }
                if self.input_channels==7&&!self.precision_f32&&self.stable_model.is_none() {
                    self.validate_phase_repeats(width,height,input,tensor)?;
                    // Sweeps must never return their last synthetic phase to playback.
                    check(unsafe{(self.api.infer)(self.request)},"restore requested phase output")?;
                    check(unsafe{(self.api.tensor_data)(self.output,&mut data)},"read requested phase output")?;
                    ensure!(!data.is_null(),"restored phase output is null");
                }
                self.validated_rife_shapes.insert((width, height));
                log::info!("neoarc-rife-shape-validation: model={} shape={}x{} repeat_max_delta={:.6} repeat_mean_delta={:.8} result=accepted precision={} input_channels={}",
                    self.model_path.display(), width, height, max_delta, mean_delta,
                    if self.precision_f32 { "f32" } else if self.stable_model.is_some() { "stable-fp16" } else { "default" }, self.input_channels);
            }
            consume(unsafe { std::slice::from_raw_parts(data.cast::<f16>(), 3 * plane) })
        })();
        unsafe { (self.api.tensor_free)(tensor) };
        result
    }

    pub(crate) fn run(&mut self, width: usize, height: usize, input: &[f16]) -> Result<Vec<f16>> {
        self.run_with_output(width, height, input, |output| Ok(output.to_vec()))
    }

    fn rife_rgb_enabled(&self)->bool {
        self.input_channels==7&&!self.image_mode&&!self.precision_f32
            && std::env::var("CHIDESCALER_NEOARC_RIFE_RGB_IO").ok().as_deref()!=Some("off")
            && std::env::var("CHIDESCALER_NEOARC_REMOTE").ok().as_deref()!=Some("off")
    }
    pub(crate) fn run_rife_rgb(&mut self,w:usize,h:usize,frames:&[&[u8]],stride:usize,phase:f32)->Result<Option<(Vec<u8>,[f64;3])>> {
        if self.rife_rgb.as_ref().is_some_and(|io|!io.matches_shape(w,h,7)) {
            self.rife_rgb.take();self.rife_rgb_shape=None;
        }
        if !self.rife_rgb_enabled()||self.requires_shape_fp32(w,h)||self.rife_rgb_shape!=Some((w,h)){return Ok(None);}
        let Some(io)=self.rife_rgb.as_mut()else{return Ok(None)};
        match io.run_rife(w,h,frames,stride,phase){Ok(v)=>Ok(Some(v)),Err(error)=>{
            log::warn!("neoarc-rife-rgb-fallback: shape={}x{} reason={error:#}",w,h);
            self.rife_rgb.take();self.rife_rgb_shape=None;self.rife_rgb_rejected.insert((w,h));Ok(None)
        }}
    }
    pub(crate) fn prepare_rife_rgb(&mut self,w:usize,h:usize,input:&[f16],frames:&[&[u8]],stride:usize) {
        if !self.rife_rgb_enabled()||self.requires_shape_fp32(w,h)||self.rife_rgb_shape==Some((w,h))||self.rife_rgb_rejected.contains(&(w,h)){return;}
        let existing=self.rife_rgb.take();self.rife_rgb_shape=None;let started=std::time::Instant::now();
        let result=(||->Result<neoarc_rgb::RgbIo>{
            let n=w*h;let mut probe=input.to_vec();let mut references=Vec::new();
            for phase in [0.2f32,0.5,0.8]{probe[6*n..].fill(f16::from_f32(phase));references.push(self.run_with_output(w,h,&probe,|v|Ok(v.to_vec()))?);}
            ensure!(!self.precision_f32,"canonical model requires FP32");
            let mut io=if let Some(io)=existing{io}else{neoarc_rgb::RgbIo::new(self.api,self.core,&self.device,&self._library,&self.stable_cache,self.disk_cache_allowed)?};
            io.prepare(self.core,self.stable_model.as_deref().unwrap_or(&self.model_path),7,w,h,false)?;
            for(phase,reference)in [0.2f32,0.5,0.8].into_iter().zip(&references){
                let mut previous:Option<Vec<f16>>=None;
                for _ in 0..2 {
                    io.run_rife(w,h,frames,stride,phase)?;let output=io.read_rife_output()?;
                    for(kind,expected)in [("canonical",Some(reference.as_slice())),("repeat",previous.as_deref())]{if let Some(expected)=expected{
                        ensure!(expected.len()==output.len(),"RIFE RGB validation length mismatch");let mut max=0.0f32;let mut sum=0.0f64;
                        for(a,b)in output.iter().zip(expected){let(a,b)=(a.to_f32(),b.to_f32());ensure!(a.is_finite()&&b.is_finite(),"nonfinite RIFE RGB output");let d=(a-b).abs();max=max.max(d);sum+=d as f64;}
                        let mean=sum/output.len()as f64;log::info!("neoarc-rife-rgb-validation: model={} shape={}x{} t={} kind={} max={:.6} mean={:.8}",self.model_path.display(),w,h,phase,kind,max,mean);
                        if !(max<0.25&&mean<0.001){return Err(RepeatDrift{max,mean}.into());}
                    }}
                    previous=Some(output);
                }
            }
            Ok(io)
        })();
        match result {Ok(io)=>{self.retire_previous();self.retire_shape();self.validated_rife_shapes.clear();self.rife_rgb=Some(io);self.rife_rgb_shape=Some((w,h));log::info!("neoarc-rife-rgb-ready: model={} shape={}x{} validation_ms={:.3} host_fp16_bytes=0 host_rgb_io=true canonical_resources=retired",self.model_path.display(),w,h,started.elapsed().as_secs_f64()*1000.0);},Err(error)=>{
            if error.downcast_ref::<RepeatDrift>().is_some(){
                self.retire_previous();self.retire_shape();self.validated_rife_shapes.clear();self.shape_fp32.insert((w,h));
                log::warn!("neoarc-shape-precision-fallback: model={} shape={}x{} precision=f32 scope=shape reason=remote-validation",self.model_path.display(),w,h);
            }
            self.rife_rgb_rejected.insert((w,h));log::warn!("neoarc-rife-rgb-rejected: model={} shape={}x{} reason={error:#} action=canonical-host",self.model_path.display(),w,h);}}
    }

    fn warm_gl_key(&self,w:usize,h:usize,rc:isize,dc:isize)->WarmGlKey{
        (self.core as usize,self.model_identity.clone(),self.stable_model.clone(),w,h,rc,dc,self.precision_f32||self.requires_shape_fp32(w,h))
    }
    pub(crate) fn park_gl_interp(&self,w:usize,h:usize,rc:isize,dc:isize,mut io:GlInterpIo){
        io.io.park();
        let mut cache=warm_gl_cache().lock().unwrap_or_else(|e|e.into_inner());
        if std::env::var("CHIDESCALER_NEOARC_WARM_GRAPH").ok().as_deref()==Some("off")||!warm_gl_budget(){cache.clear();return;}
        let key=self.warm_gl_key(w,h,rc,dc);cache.retain(|entry|entry.key!=key);
        cache.push_back(WarmGl{key,io});while cache.len()>5{cache.pop_front();}
        log::info!("neoarc-warm-graph-park: entries={} limit=5 request=freed input_output_buffers=freed gl_images=released budget_guard=pass",cache.len());
    }
    pub(crate) fn create_gl_interp(&mut self,w:usize,h:usize,rc:isize,dc:isize,
        input:&[f16],textures:&[u32],output:u32)->Result<GlInterpIo> {
        let key=self.warm_gl_key(w,h,rc,dc);
        let warm={let mut cache=warm_gl_cache().lock().unwrap_or_else(|e|e.into_inner());
            if std::env::var("CHIDESCALER_NEOARC_WARM_GRAPH").ok().as_deref()==Some("off")||!warm_gl_budget(){cache.clear();None}
            else{
                // v985: a parked bridge keeps its compiled and validated boost
                // graphs (and the native-head verdict). Reuse it when it holds
                // the boost graphs requested now; otherwise drop it and rebuild.
                match cache.iter().position(|entry|entry.key==key){
                    Some(index) if self.rife_boost_source.is_none()
                        ||(cache[index].io.io.has_boost()&&cache[index].io.io.boost_source()==self.rife_boost_source.as_ref())=>cache.remove(index),
                    Some(index)=>{cache.remove(index);None}
                    None=>None,
                }
            }};
        if let Some(mut warm)=warm {
            let started=std::time::Instant::now();
            let result=(||->Result<()>{
                let path=if key.7{&self.model_path}else{self.stable_model.as_deref().unwrap_or(&self.model_path)};
                warm.io.io.prepare(self.core,path,self.input_channels,w,h,key.7)?;
                // The exact compiled graph retains its original nine-phase
                // canonical proof. New bindings still repeat-check all phases.
                // v993: three phases bound repeat stability of a reused bridge
                // (nine added ~0.2 s to every return to a parked model/geometry).
                for phase in [0.2f32,0.5,0.8]{
                    warm.io.io.run_gl_interp(textures,output,phase)?;let reference=warm.io.io.read_rife_output()?;
                    warm.io.io.run_gl_interp(textures,output,phase)?;let values=warm.io.io.read_rife_output()?;
                    ensure!(reference.len()==values.len(),"warm graph output extent drift");let(mut max,mut sum)=(0.0f32,0.0f64);
                    for(a,b)in reference.iter().zip(&values){let(a,b)=(a.to_f32(),b.to_f32());ensure!(a.is_finite()&&b.is_finite(),"warm graph nonfinite output");let d=(a-b).abs();max=max.max(d);sum+=d as f64;}
                    let mean=sum/values.len()as f64;if !(max<0.25&&mean<0.001){return Err(RepeatDrift{max,mean}.into());}
                }Ok(())
            })();
            match result{Ok(())=>{log::info!("neoarc-warm-graph-ready: shape={}x{} canonical_proof=retained phase_repeat=3 boost_graphs={} native_head={} validation_ms={:.3}",w,h,warm.io.io.has_boost(),warm.io.io.native_head_active(),started.elapsed().as_secs_f64()*1000.0);return Ok(warm.io);},Err(error)=>log::warn!("neoarc-warm-graph-rejected: reason={error:#} action=full-canonical-compile-validation")}
        }
        let mut result=self.create_gl_interp_once(w,h,rc,dc,input,textures,output);
        let unstable=|r:&Result<GlInterpIo>|r.as_ref().err().is_some_and(|e|e.downcast_ref::<RepeatDrift>().is_some());
        if unstable(&result)&&!self.precision_f32&&!self.requires_shape_fp32(w,h)&&self.stable_model.is_none() {
            if let Ok(path)=neoarc_graph::prepare(&self.stable_cache,&self.model_path) {
                self.retire_previous();self.retire_shape();self.validated_rife_shapes.clear();self.stable_model=Some(path);
                log::warn!("neoarc-gl-interp-precision-retry: shape={}x{} precision=stable-fp16 reason=phase-validation",w,h);
                result=self.create_gl_interp_once(w,h,rc,dc,input,textures,output);
            }
        }
        if unstable(&result)&&!self.precision_f32&&!self.requires_shape_fp32(w,h) {
            self.shape_fp32.insert((w,h));self.retire_previous();self.retire_shape();self.validated_rife_shapes.clear();
            log::warn!("neoarc-gl-interp-precision-retry: shape={}x{} precision=f32 reason=phase-validation",w,h);
            return self.create_gl_interp_once(w,h,rc,dc,input,textures,output);
        }
        result
    }
    /// v987: key of a completed canonical validation on this machine: model
    /// content, stable-FP16 choice, geometry, channels, OpenVINO device and the
    /// OpenCL driver version. A driver update or another model never matches.
    fn validation_marker(&self,w:usize,h:usize,driver:&str)->PathBuf{
        use sha2::{Digest,Sha256};
        let key=format!("neoarc-gl-validated-v1|{}|stable={}|{}x{}|ch={}|dev={}|drv={}",self.model_identity.display(),
            self.stable_model.as_ref().map(|p|p.display().to_string()).unwrap_or_default(),w,h,self.input_channels,self.device.to_string_lossy(),driver);
        self.stable_cache.parent().unwrap_or(&self.stable_cache).join("validated-v1").join(format!("{:x}.ok",Sha256::digest(key.as_bytes())))
    }
    /// v993: model-level marker (any geometry passed a full canonical check
    /// with this model, stable-FP16 choice, device and driver).
    fn validation_marker_model(&self,driver:&str)->PathBuf{
        use sha2::{Digest,Sha256};
        let key=format!("neoarc-gl-validated-model-v1|{}|stable={}|ch={}|dev={}|drv={}",self.model_identity.display(),
            self.stable_model.as_ref().map(|p|p.display().to_string()).unwrap_or_default(),self.input_channels,self.device.to_string_lossy(),driver);
        self.stable_cache.parent().unwrap_or(&self.stable_cache).join("validated-v1").join(format!("model-{:x}.ok",Sha256::digest(key.as_bytes())))
    }
    fn known_driver(&self)->Option<String>{
        let mut known=known_driver_cache().lock().unwrap_or_else(|e|e.into_inner());
        if known.is_none(){
            let path=self.stable_cache.parent().unwrap_or(&self.stable_cache).join("validated-v1").join("driver.txt");
            *known=std::fs::read_to_string(path).ok().map(|s|s.trim().to_string()).filter(|s|!s.is_empty());
        }
        known.clone()
    }
    fn remember_driver(&self,driver:&str){
        let mut known=known_driver_cache().lock().unwrap_or_else(|e|e.into_inner());
        if known.as_deref()!=Some(driver){
            *known=Some(driver.to_string());
            let dir=self.stable_cache.parent().unwrap_or(&self.stable_cache).join("validated-v1");
            if std::fs::create_dir_all(&dir).is_ok(){let _=std::fs::write(dir.join("driver.txt"),driver);}
        }
    }
    fn create_gl_interp_once(&mut self,w:usize,h:usize,rc:isize,dc:isize,
        input:&[f16],textures:&[u32],output:u32)->Result<GlInterpIo> {
        self.create_gl_interp_validated(w,h,rc,dc,input,textures,output,
            std::env::var("CHIDESCALER_NEOARC_VALIDATION_CACHE").ok().as_deref()!=Some("off"))
    }
    #[allow(clippy::too_many_arguments)]
    fn create_gl_interp_validated(&mut self,w:usize,h:usize,rc:isize,dc:isize,
        input:&[f16],textures:&[u32],output:u32,allow_marker:bool)->Result<GlInterpIo> {
        ensure!(matches!((textures.len(),self.input_channels),(2,7)|(4,14)),"unsupported shared interpolation layout");
        let n=w*h;let aux=textures.len()*3;let mut probe=input.to_vec();let mut references=Vec::new();
        let phases=[0.2f32,0.25,1.0/3.0,0.4,0.5,0.6,2.0/3.0,0.75,0.8];
        // v987: a geometry/model/driver already proven against the canonical
        // graph skips the CPU-visible reference compile and its nine runs.
        // Repeat-stability checks (which catch FP16 drift) still run every time.
        let assumed_driver=if allow_marker&&!(self.precision_f32||self.requires_shape_fp32(w,h)){self.known_driver()}else{None};
        // v993: a new geometry of a model already proven at another geometry
        // (same stable-FP16 choice, device, driver) also skips the reference
        // compile (8.9 s for a first 960x540). Repeat checks still run.
        let skip_references=assumed_driver.as_deref().is_some_and(|driver|self.validation_marker(w,h,driver).is_file()
            ||(std::env::var("CHIDESCALER_NEOARC_VALIDATION_CACHE").ok().as_deref()!=Some("shape")&&self.validation_marker_model(driver).is_file()));
        if !skip_references{
            for phase in phases {
                probe[aux*n..(aux+1)*n].fill(f16::from_f32(phase));
                references.push(self.run_with_output(w,h,&probe,|v|Ok(v.to_vec()))?);
            }
        }
        let f32=self.precision_f32||self.requires_shape_fp32(w,h);
        let mut io=neoarc_rgb::RgbIo::new_gl(self.api,self.core,&self.device,&self._library,rc,dc,&self.stable_cache,self.disk_cache_allowed)?;
        let actual_driver=io.driver_version();
        if let Some(driver)=actual_driver.as_deref(){self.remember_driver(driver);}
        if skip_references&&actual_driver!=assumed_driver{
            log::info!("neoarc-validation-cache: shape={}x{} result=driver-changed action=full-validation",w,h);
            drop(io);return self.create_gl_interp_validated(w,h,rc,dc,input,textures,output,false);
        }
        if skip_references{
            log::info!("neoarc-validation-cache: shape={}x{} result=hit canonical_compile=skipped repeat_checks=kept",w,h);
            // A per-shape proof is also a model-level proof (v993).
            if let Some(driver)=actual_driver.as_deref(){let model=self.validation_marker_model(driver);if self.validation_marker(w,h,driver).is_file()&&!model.is_file(){let _=std::fs::write(model,b"ok");}}
        }
        let path=if f32{&self.model_path}else{self.stable_model.as_deref().unwrap_or(&self.model_path)};
        io.set_boost_source(if f32{None}else{self.rife_boost_source.clone()});
        io.set_native_head_candidate(!f32&&self.rife_boost_native_head);
        io.prepare(self.core,path,self.input_channels,w,h,f32)?;
        // Native FP16 XS has a measurable repeated-run variance even with the
        // cache disabled. Keep its original full-pack transaction; enable the
        // within-pair shortcut only for stabilized FP16 or GPU FP32 execution.
        if !f32&&self.stable_model.is_none(){io.disable_interp_pair_cache();}
        ensure!(io.output_size()==(w,h),"shared interpolation output extent differs");
        // v963: validate the boost graphs against the same canonical references
        // first (one pair: trunk once, nine heads). Any drift -> original graph.
        if io.has_boost(){
            io.set_boost_force(Some(true));io.begin_interp_pair();
            let boost_check=(||->Result<()>{
                for (index, phase) in phases.into_iter().enumerate() {
                    // v993: three phases (0.2/0.5/0.8) bound repeat stability; the
                    // other six added ~0.4 s to every bridge build.
                    if !matches!(index, 0 | 4 | 8) { continue; }
                    // v987: without a canonical reference (validation cache hit)
                    // the diagnostic comparison uses the run itself (zero diff).
                    let mut previous: Option<Vec<f16>> = None;
                    for repeat in 0..2 {
                        io.run_gl_interp(textures, output, phase)?;
                        ensure!(io.has_boost(), "boost request fell back during validation");
                        let values = io.read_rife_output()?;
                        let reference = references.get(index).unwrap_or(&values);
                        ensure!(values.len() == reference.len() && !values.is_empty(), "boost validation extent mismatch");
                        let mut max = 0.0f32;
                        let mut sum = 0.0f64;
                        let mut noticeable = 0;
                        let mut severe = 0;
                        let mut repeat_max = 0.0f32;
                        let mut repeat_sum = 0.0f64;
                        for (index, (a, b)) in values.iter().zip(reference).enumerate() {
                            let (a, b) = (a.to_f32(), b.to_f32());
                            ensure!(a.is_finite() && b.is_finite(), "nonfinite boost output");
                            // Compare displayed RGB; clipping is also applied by the output kernel.
                            let d = (a.clamp(0.0, 1.0) - b.clamp(0.0, 1.0)).abs();
                            max = max.max(d);
                            sum += f64::from(d);
                            noticeable += usize::from(d > 0.10);
                            severe += usize::from(d > 0.25);
                            if let Some(previous) = &previous {
                                let d = (a - previous[index].to_f32()).abs();
                                repeat_max = repeat_max.max(d);
                                repeat_sum += f64::from(d);
                            }
                        }
                        let n = values.len() as f64;
                        let mean = sum / n;
                        let noticeable = noticeable as f64 / n;
                        let severe = severe as f64 / n;
                        let repeat_mean = repeat_sum / n;
                        log::info!("neoarc-rife-boost-validation: shape={}x{} t={} repeat={} max={:.6} mean={:.8} noticeable_ratio={:.6} severe_ratio={:.6} repeat_max={:.6} repeat_mean={:.8}",
                            w, h, phase, repeat, max, mean, noticeable, severe, repeat_max, repeat_mean);
                        // Explicit Boost selects approximate motion reuse. The
                        // comparison with the full graph is diagnostic only.
                        ensure!(repeat_max < 0.25 && repeat_mean < 0.001,
                            "unstable boost t={phase} repeat_max={repeat_max} repeat_mean={repeat_mean}");
                        previous = Some(values);
                    }
                }
                Ok(())
            })();
            if let Err(error)=boost_check{io.disable_boost();log::warn!("neoarc-rife-boost-fallback: shape={}x{} reason={error:#} stage=validation action=original-gpu-graph",w,h);}
            // v983: one-kernel Lite head. Proven against the ONNX head on the
            // same pair (trunk once, nine phases); otherwise the ONNX head stays.
            if io.has_boost()&&io.native_head_available(){
                // v985: two independent proofs. (1) Exactness: the kernel output
                // must match the same formula recomputed on the CPU in f64 (catches
                // any layout/indexing bug). (2) Semantics: against the ONNX head
                // the mean and the share of visibly different pixels must stay
                // small; per-pixel maxima are not used because the ONNX head runs
                // in FP16 (grid coordinates drift up to ~0.3 px at 1280 wide).
                let native_check=(||->Result<(f32,f64,f64,f64,f64,f64)>{
                    // v993: three phases and one exact CPU check (was nine and two).
                    let check_phases=[phases[0],phases[4],phases[8]];
                    io.set_native_head(false);io.begin_interp_pair();
                    let(mut expected,mut graph_ms)=(Vec::new(),0.0);
                    for phase in check_phases{let times=io.run_gl_interp(textures,output,phase)?;if phase!=check_phases[0]{graph_ms+=times[1]+times[2];}expected.push(io.read_rife_output()?);}
                    io.set_native_head(true);ensure!(io.native_head_active(),"native head unavailable");io.begin_interp_pair();
                    let(mut cpu_max,mut worst_mean,mut worst_visible,mut worst_severe,mut native_ms)=(0.0f32,0.0f64,0.0f64,0.0f64,0.0);
                    for(index,(phase,reference))in check_phases.into_iter().zip(&expected).enumerate(){
                        let times=io.run_gl_interp(textures,output,phase)?;if phase!=check_phases[0]{native_ms+=times[1]+times[2];}
                        if index==1{
                            let(max,mean)=io.native_head_cpu_check(phase)?;cpu_max=cpu_max.max(max);
                            ensure!(max<0.004&&mean<0.0005,"native head is not exact t={phase} cpu_max={max} cpu_mean={mean}");
                        }
                        let values=io.read_rife_output()?;ensure!(values.len()==reference.len()&&!values.is_empty(),"native head extent mismatch");
                        let(mut sum,mut visible,mut severe)=(0.0f64,0usize,0usize);
                        for(a,b)in values.iter().zip(reference){
                            let(a,b)=(a.to_f32(),b.to_f32());ensure!(a.is_finite(),"nonfinite native head output");
                            let d=(a.clamp(0.0,1.0)-b.clamp(0.0,1.0)).abs();sum+=f64::from(d);visible+=usize::from(d>0.10);severe+=usize::from(d>0.25);
                        }
                        let count=values.len()as f64;let(mean,visible,severe)=(sum/count,visible as f64/count,severe as f64/count);
                        worst_mean=worst_mean.max(mean);worst_visible=worst_visible.max(visible);worst_severe=worst_severe.max(severe);
                        ensure!(mean<0.003&&visible<0.02&&severe<0.004,"native head differs from the ONNX head t={phase} mean={mean} visible_ratio={visible} severe_ratio={severe}");
                    }
                    let heads=(check_phases.len()-1)as f64;Ok((cpu_max,worst_mean,worst_visible,worst_severe,graph_ms/heads,native_ms/heads))
                })();
                match native_check{
                    Ok((cpu_max,mean,visible,severe,graph_ms,native_ms))=>log::info!("neoarc-rife-boost-native-head: shape={}x{} active=true cpu_exact_max={:.6} onnx_mean={:.8} onnx_visible_ratio={:.6} onnx_severe_ratio={:.6} onnx_head_ms={:.3} native_head_ms={:.3} (validation run, head+output per phase)",w,h,cpu_max,mean,visible,severe,graph_ms,native_ms),
                    Err(error)=>{io.set_native_head(false);log::warn!("neoarc-rife-boost-native-head: shape={}x{} active=false reason={error:#} action=onnx-head",w,h);}
                }
            }
            io.set_boost_force(Some(false));
        }
        for attempt in 0..2 {
        io.begin_interp_pair();
        let validated=(||->Result<()>{
        for(index,phase)in phases.into_iter().enumerate() {
            // v987: on a validation-cache hit only the repeat check runs
            // (v993: on three phases; a first full proof still covers all nine).
            if references.is_empty()&&!matches!(index,0|4|8){continue;}
            let reference=references.get(index);
            let mut previous:Option<Vec<f16>>=None;
            for _ in 0..2 {
                io.run_gl_interp(textures,output,phase)?;let values=io.read_rife_output()?;
                for(kind,expected)in [("canonical",reference.map(|r|r.as_slice())),("repeat",previous.as_deref())] {
                    if let Some(expected)=expected {
                        ensure!(values.len()==expected.len(),"shared interpolation validation extent mismatch");
                        let(mut max,mut sum)=(0.0f32,0.0f64);
                        for(a,b)in values.iter().zip(expected) {
                            let(a,b)=(a.to_f32(),b.to_f32());ensure!(a.is_finite()&&b.is_finite(),"nonfinite shared interpolation output");
                            let d=(a-b).abs();max=max.max(d);sum+=d as f64;
                        }
                        let mean=sum/values.len()as f64;
                        log::info!("neoarc-gl-interp-validation: shape={}x{} t={} kind={} max={:.6} mean={:.8}",w,h,phase,kind,max,mean);
                        if !(max<0.25&&mean<0.001){return Err(RepeatDrift{max,mean}.into());}
                    }
                }
                previous=Some(values);
            }
        }
        Ok(())})();
        if let Err(error)=validated {
            if attempt==0&&io.has_encoder_cache(){
                log::warn!("neoarc-encoder-cache-validation-rejected: shape={}x{} reason={error:#} action=original-gpu-graph",w,h);
                io.disable_encoder_cache();continue;
            }
            return Err(error);
        }
        break;
        }
        io.set_boost_force(None);
        // v987: remember a full canonical pass for this model/geometry/driver.
        if !skip_references&&!f32&&!references.is_empty()&&allow_marker{
            if let Some(driver)=actual_driver.as_deref(){
                let marker=self.validation_marker(w,h,driver);
                if let Some(dir)=marker.parent(){if std::fs::create_dir_all(dir).is_ok()&&std::fs::write(&marker,b"ok").is_ok(){
                    let _=std::fs::write(self.validation_marker_model(driver),b"ok");
                    log::info!("neoarc-validation-cache: shape={}x{} result=stored model_level=true",w,h);
                }}
            }
        }
        if !warm_gl_budget(){warm_gl_cache().lock().unwrap_or_else(|e|e.into_inner()).clear();}
        self.rife_rgb.take();self.rife_rgb_shape=None;self.retire_previous();self.retire_shape();self.validated_rife_shapes.clear();
        log::info!("neoarc-gl-interp-ready: input={}x{} channels={} precision={} host_input_bytes=0 host_output_bytes=0 validation=phase-repeat-canonical boost_graphs={}",w,h,self.input_channels,if f32{"f32"}else{"fp16"},io.has_boost());
        Ok(GlInterpIo{io})
    }

    pub fn diagnostic_remote_profile(&self,w:usize,h:usize,channels:usize,frames:&[&[u8]],profile:bool)->Result<serde_json::Value> {
        ensure!(matches!(channels,3|7|21),"diagnostic channels must be 3, 7 or 21");
        let mut io=neoarc_rgb::RgbIo::new(self.api,self.core,&self.device,&self._library,&self.stable_cache,self.disk_cache_allowed)?;
        let path=self.stable_model.as_deref().unwrap_or(&self.model_path);
        let f32=self.precision_f32||self.requires_shape_fp32(w,h);
        if channels==7 {io.prepare(self.core,path,channels,w,h,f32)?;io.run_rife(w,h,frames,3,0.5)?;}
        else{ensure!(!frames.is_empty(),"missing diagnostic image");io.run(self.core,path,channels,w,h,frames[0],3,3,true,f32)?;}
        let mut result=io.diagnostic_infer_profile(&self._library,profile)?;
        result["model"]=serde_json::json!(path.to_string_lossy());result["shape"]=serde_json::json!([1,channels,h,w]);
        result["precision"]=serde_json::json!(self.precision_for_shape(w,h));result["device"]=serde_json::json!(self.device.to_string_lossy());Ok(result)
    }

    pub(crate) fn retire_gl_rgb(&mut self) {
        self.gl_rgb.take();self.gl_rgb_reference.take();self.gl_rgb_key=None;self.gl_rgb_disabled=false;self.gl_rgb_shape_key=None;
    }
    fn channel_select_model(&self,plan:&neoarc_graph::ChannelSelectPlan,w:usize,h:usize)->Result<Option<PathBuf>>{
        let get_port:unsafe extern "C" fn(Handle,usize,*mut Handle)->i32=unsafe{*self._library.get(b"ov_model_const_output_by_index\0")?};
        let get_shape:unsafe extern "C" fn(Handle,*mut Shape)->i32=unsafe{*self._library.get(b"ov_const_port_get_shape\0")?};
        let get_kind:unsafe extern "C" fn(Handle,*mut i32)->i32=unsafe{*self._library.get(b"ov_port_get_element_type\0")?};
        let free_port:unsafe extern "C" fn(Handle)=unsafe{*self._library.get(b"ov_output_const_port_free\0")?};
        let mut model=ptr::null_mut();let path=CString::new(plan.analysis.to_string_lossy().as_bytes())?;
        let status=unsafe{(self.api.model_read)(self.core,path.as_ptr(),ptr::null(),&mut model)};
        if status!=0{if !model.is_null(){unsafe{(self.api.model_free)(model)}}check(status,"read temporal port analysis")?;}
        let result=(||->Result<Option<PathBuf>>{
            let dims=[1,self.input_channels as i64,h as i64,w as i64];let mut shape=PartialShape{rank:Dimension{min:0,max:0},dims:ptr::null_mut()};
            check(unsafe{(self.api.shape_create)(4,dims.as_ptr(),&mut shape)},"create temporal analysis shape")?;
            let status=unsafe{(self.api.model_reshape)(model,shape)};unsafe{(self.api.shape_free)(&mut shape)};check(status,"reshape temporal port analysis")?;
            let mut metadata=Vec::new();
            for index in 0..plan.names.len(){
                let mut port=ptr::null_mut();check(unsafe{get_port(model,index,&mut port)},"get temporal analysis port")?;
                let mut shape=Shape{rank:0,dims:ptr::null_mut()};let status=unsafe{get_shape(port,&mut shape)};let mut kind=0;
                let item=if status==0&&shape.rank==4&&!shape.dims.is_null()&&unsafe{get_kind(port,&mut kind)}==0{
                    let dims=unsafe{std::slice::from_raw_parts(shape.dims,4)};
                    if matches!(kind,3|4)&&matches!(dims[1],2|4){Some((if kind==3{10}else{1},dims[1]as usize))}else{None}
                }else{None};
                if status==0{unsafe{let _=(self.api.static_shape_free)(&mut shape);}}
                unsafe{free_port(port)};metadata.push(item);
            }
            plan.materialize(self.stable_cache.parent().unwrap_or(&self.stable_cache),w,h,&metadata)
        })();unsafe{(self.api.model_free)(model)};result
    }
    pub(crate) fn prepare_gl_rgb(&mut self,w:usize,h:usize,rc:isize,dc:isize)->Result<Option<(usize,usize)>> {
        if !self.image_mode || rc==0 || dc==0
            || std::env::var("CHIDESCALER_NEOARC_GL_IO").ok().as_deref()==Some("off") {return Ok(None);}
        if !self.precision_f32&&self.stable_model.is_none()&&self.channel_select_shape!=Some((w,h)){
            if let Some(plan)=self.channel_select_plan.as_ref(){
                self.channel_select_shape=Some((w,h));
                // Conservative rollout: larger shapes showed pre-existing
                // FP16 recompilation drift even with this optimization OFF.
                // Preserve their original path until independently qualified.
                self.image_merged_model=if (w,h)==(640,480)&&plan.qualified_480p(){match self.channel_select_model(plan,w,h){Ok(path)=>path,Err(error)=>{log::warn!("neoarc-channel-select-analysis-fallback: reason={error:#} action=original-gpu-graph");None}}}else{None};
            }
        }
        // A later canonical recovery must never retain an optimized FP16 graph.
        if (self.precision_f32||self.stable_model.is_some())&&self.image_merged_model.take_if(|_|true).is_some(){
            self.gl_rgb.take();self.gl_rgb_reference.take();
        }
        if (self.precision_f32||self.stable_model.is_some())&&self.compact_plan.take_if(|_|true).is_some(){self.gl_rgb.take();self.gl_rgb_reference.take();}
        if self.gl_rgb_key!=Some((rc,dc)) {self.retire_gl_rgb();self.gl_rgb_key=Some((rc,dc));}
        if self.gl_rgb_disabled&&self.gl_rgb_shape_key!=Some((w,h)){self.gl_rgb_disabled=false;}
        if self.gl_rgb_disabled {return Ok(None);}
        self.gl_rgb_shape_key=Some((w,h));
        let native=self.compact_plan.is_some()&&!self.precision_f32&&self.stable_model.is_none()&&!self.image_merge_rejected.contains(&(w,h));
        let merged=(self.image_merged_model.is_some()||native)&&!self.image_merge_rejected.contains(&(w,h));
        let result:Result<(usize,usize)>=(|| {
            let reshape=self.gl_rgb.as_ref().is_none_or(|io|!io.matches_shape(w,h,self.input_channels));
            if self.gl_rgb.is_none(){self.gl_rgb=Some(neoarc_rgb::RgbIo::new_gl(self.api,self.core,&self.device,&self._library,rc,dc,&self.stable_cache,self.disk_cache_allowed)?);}
            let native_resume=native&&self.gl_rgb.as_ref().is_some_and(|io|io.can_resume_native(w,h));
            if reshape{
                self.gl_rgb_reference.take();
                if merged&&!native_resume{
                    let mut reference=neoarc_rgb::RgbIo::new_gl(self.api,self.core,&self.device,&self._library,rc,dc,&self.stable_cache,self.disk_cache_allowed)?;
                    reference.prepare(self.core,&self.model_path,self.input_channels,w,h,self.precision_f32)?;
                    self.gl_rgb_reference=Some(reference);
                }
            }
            let path=if merged&&!native{self.image_merged_model.as_ref().unwrap()}else{self.stable_model.as_ref().unwrap_or(&self.model_path)};
            let io=self.gl_rgb.as_mut().unwrap();
            if native{io.prepare_native(self.compact_plan.as_deref().unwrap(),&self.compact_library,w,h)?;}else{io.prepare(self.core,path,self.input_channels,w,h,self.precision_f32)?;}
            if let Some(reference)=self.gl_rgb_reference.as_ref(){ensure!(io.output_size()==reference.output_size(),"merged SISR output extent mismatch");}
            Ok(io.output_size())
        })();
        match result {Ok(size)=>Ok(Some(size)),Err(error)=>{
            if merged{
                self.image_merge_rejected.insert((w,h));self.gl_rgb.take();
                if let Some(reference)=self.gl_rgb_reference.take(){let size=reference.output_size();self.gl_rgb=Some(reference);
                    log::warn!("neoarc-sisr-optimized-compile-fallback: shape={}x{} reason={error:#} action=original-gpu-graph",w,h);return Ok(Some(size));}
                log::warn!("neoarc-sisr-optimized-compile-retry: shape={}x{} reason={error:#} action=original-gpu-graph",w,h);
                return self.prepare_gl_rgb(w,h,rc,dc);
            }
            log::warn!("neoarc-gl-io-fallback: model={} reason={error:#} action=byte-io",self.model_path.display());
            self.gl_rgb.take();self.gl_rgb_reference.take();self.gl_rgb_disabled=true;Ok(None)
        }}
    }
    pub(crate) fn run_gl_rgb(&mut self,input:u32,output:u32,reset:bool,input_id:u64,output_id:u64)->Result<Option<[f64;3]>> {
        if let Some(mut reference)=self.gl_rgb_reference.take(){
            let started=std::time::Instant::now();
            let key=self.gl_rgb_shape_key.and_then(|(w,h)|self.gl_rgb.as_ref().and_then(|io|io.device_identity().map(|device|(w,h,io.is_native(),device))));
            let speed_cached=std::env::var("CHIDESCALER_NEOARC_SPEED_CHECK").ok().as_deref()!=Some("legacy")&&std::env::var("CHIDESCALER_NEOARC_SPEED_CACHE").ok().as_deref()!=Some("off")&&key.is_some_and(|k|self.sisr_speed_qualified.contains(&k));
            let validate=(||->Result<[f64;3]>{
                let io=self.gl_rgb.as_mut().ok_or_else(||anyhow!("merged GL model missing"))?;
                let legacy=std::env::var("CHIDESCALER_NEOARC_SPEED_CHECK").ok().as_deref()==Some("legacy");
                let(mut reference_ms,mut candidate_ms)=(0.,0.);let(mut reference_p95,mut candidate_p95)=(0.,0.);
                if speed_cached{
                    // Re-check canonical pixels and repeatability below in the
                    // new GL context. Only the already-proven speed trial is reused.
                    for _ in 0..2{reference.run_gl(input,output,true,input_id,output_id)?;io.run_gl(input,output,true,input_id,output_id)?;}
                }else if legacy{
                    for _ in 0..8{let start=std::time::Instant::now();reference.run_gl(input,output,true,input_id,output_id)?;reference_ms=start.elapsed().as_secs_f64()*1000.;}
                    for _ in 0..8{let start=std::time::Instant::now();io.run_gl(input,output,true,input_id,output_id)?;candidate_ms=start.elapsed().as_secs_f64()*1000.;}
                }else{
                    // Alternate execution order under the same clocks/load. A
                    // single last sample used to reject a faster native path.
                    for _ in 0..2{reference.run_gl(input,output,true,input_id,output_id)?;io.run_gl(input,output,true,input_id,output_id)?;}
                    let(mut original,mut candidate)=(Vec::with_capacity(8),Vec::with_capacity(8));
                    for pair in 0..8{for mode in if pair%2==0{[0,1]}else{[1,0]}{let start=std::time::Instant::now();if mode==0{reference.run_gl(input,output,true,input_id,output_id)?;}else{io.run_gl(input,output,true,input_id,output_id)?;}let ms=start.elapsed().as_secs_f64()*1000.;if mode==0{original.push(ms);}else{candidate.push(ms);}}}
                    original.sort_by(f64::total_cmp);candidate.sort_by(f64::total_cmp);reference_ms=(original[3]+original[4])*0.5;candidate_ms=(candidate[3]+candidate[4])*0.5;reference_p95=original[7];candidate_p95=candidate[7];
                }
                let expected=reference.read_rife_output()?;
                let mut last=None;let mut times=[0.;3];
                for _ in 0..2{
                    times=io.run_gl(input,output,true,input_id,output_id)?;let actual=io.read_rife_output()?;
                    ensure!(actual.len()==expected.len()&&!actual.is_empty(),"merged SISR output length mismatch");
                    for other in std::iter::once(expected.as_slice()).chain(last.as_deref()){
                        let mut maximum=0u8;let mut total=0u64;
                        for(a,b)in actual.iter().zip(other){let(a,b)=(a.to_f32(),b.to_f32());ensure!(a.is_finite()&&b.is_finite(),"non-finite merged SISR result");
                            let byte=|x:f32|(x.clamp(0.,1.)*255.+0.5).floor()as u8;let delta=byte(a).abs_diff(byte(b));maximum=maximum.max(delta);total+=delta as u64;
                        }
                        let mean=total as f64/actual.len()as f64;ensure!(maximum<=4&&mean<0.5,"merged SISR drift: max_byte={maximum} mean_byte={mean}");
                    }
                    last=Some(actual);
                }
                ensure!(speed_cached||(candidate_ms<=reference_ms*if legacy{0.9}else{0.95}&&(legacy||candidate_p95<=reference_p95*1.02)),"merged SISR gain below rollout threshold: original_ms={reference_ms:.3} candidate_ms={candidate_ms:.3} original_p95_ms={reference_p95:.3} candidate_p95_ms={candidate_p95:.3} paired={}",!legacy);
                if speed_cached{log::info!("neoarc-sisr-speed-cache-reused: shape={:?} scope=same-session-shape-implementation canonical_repeat_retained=true",self.gl_rgb_shape_key);}else{                log::info!("neoarc-sisr-optimized-speed-check: original_ms={reference_ms:.3} candidate_ms={candidate_ms:.3} original_p95_ms={reference_p95:.3} candidate_p95_ms={candidate_p95:.3} paired={} scope=warm-gl-input-to-completed-gl-output",!legacy);}
                Ok(times)
            })();
            match validate{
                Ok(times)=>{if let Some(key)=key{if self.sisr_speed_qualified.len()>=32{self.sisr_speed_qualified.clear();}self.sisr_speed_qualified.insert(key);}log::info!("neoarc-sisr-optimized-validation: model={} shape={:?} canonical=pass repeat=pass validation_ms={:.3} steady_state_host_bytes=0 implementation={}",self.model_path.display(),self.gl_rgb_shape_key,started.elapsed().as_secs_f64()*1000.,if self.gl_rgb.as_ref().is_some_and(|io|io.is_native()){"native-xmx-compact"}else{"optimized-openvino-graph"});return Ok(Some(times));}
                Err(error)=>{
                    if let Some(key)=key{self.sisr_speed_qualified.remove(&key);}
                    if let Some(shape)=self.gl_rgb_shape_key{self.image_merge_rejected.insert(shape);}
                    self.gl_rgb=Some(reference);log::warn!("neoarc-sisr-optimized-validation-fallback: model={} reason={error:#} action=original-gpu-graph",self.model_path.display());
                }
            }
        }
        let Some(io)=self.gl_rgb.as_mut()else{return Ok(None);};
        match io.run_gl(input,output,reset,input_id,output_id) {Ok(times)=>Ok(Some(times)),Err(error)=>{
            if self.image_merged_model.is_some()||self.compact_plan.is_some(){
                if let Some(shape)=self.gl_rgb_shape_key{
                    if self.image_merge_rejected.insert(shape){
                        self.gl_rgb.take();log::warn!("neoarc-sisr-optimized-runtime-fallback: reason={error:#} action=original-gpu-graph");
                        if let Some((rc,dc))=self.gl_rgb_key{
                            if self.prepare_gl_rgb(shape.0,shape.1,rc,dc)?.is_some(){return self.run_gl_rgb(input,output,true,input_id,output_id);}
                        }
                    }
                }
            }
            log::warn!("neoarc-gl-io-fallback: model={} reason={error:#} action=byte-io history=reset",self.model_path.display());
            self.gl_rgb.take();self.gl_rgb_disabled=true;Ok(None)
        }}
    }
    pub(crate) fn gl_rgb_import_stats(&self)->Option<(bool,usize,u64,u64,usize)>{
        self.gl_rgb.as_ref().map(|io|io.image_cache_stats())
    }

    pub(crate) fn run_rgb_image(&mut self, width: usize, height: usize,
        pixels: &[u8], stride: usize, output_stride: usize, reset: bool,
    ) -> Result<Option<(usize, usize, Vec<u8>, [f64;3])>> {
        if self.rgb_disabled || !self.image_mode
            || std::env::var("CHIDESCALER_NEOARC_RGB_IO").ok().as_deref() == Some("off") { return Ok(None); }
        let result = (|| {
            if self.rgb.is_none() {
                self.rgb = Some(neoarc_rgb::RgbIo::new(self.api, self.core, &self.device, &self._library,&self.stable_cache,self.disk_cache_allowed)?);
            }
            self.rgb.as_mut().unwrap().run(self.core,
                self.stable_model.as_deref().unwrap_or(&self.model_path), self.input_channels,
                width, height, pixels, stride, output_stride, reset, self.precision_f32)
        })();
        match result {
            Ok(result) => Ok(Some(result)),
            Err(error) => {
                log::warn!("neoarc-rgb-io-fallback: model={} reason={error:#} action=host-tensors", self.model_path.display());
                self.rgb.take(); self.rgb_disabled = true;
                Ok(None)
            }
        }
    }

    pub(crate) fn run_image_with_output<R>(&mut self, width: usize, height: usize,
        input: &[f16], consume: impl FnOnce(usize, usize, ImageOutput<'_>) -> Result<R>) -> Result<R> {
        ensure!(self.image_mode, "NeoArc image session required");
        let count = width.checked_mul(height).and_then(|v| v.checked_mul(self.input_channels))
            .ok_or_else(|| anyhow!("NeoArc image input overflow"))?;
        ensure!(width > 0 && height > 0 && input.len() == count, "NeoArc image input invalid");
        self.prepare_shape(width, height)?;
        let mut dims = [1i64, self.input_channels as i64, height as i64, width as i64];
        let mut tensor = ptr::null_mut();
        let mut output = ptr::null_mut();
        check(unsafe { (self.api.tensor_from_host)(3, Shape { rank: 4, dims: dims.as_mut_ptr() },
            input.as_ptr().cast_mut().cast(), &mut tensor) }, "wrap SISR input")?;
        let result = (|| {
            check(unsafe { (self.api.set_input)(self.request, tensor) }, "bind SISR input")?;
            check(unsafe { (self.api.infer)(self.request) }, "infer SISR")?;
            check(unsafe { (self.api.get_output)(self.request, 0, &mut output) }, "get SISR output")?;
            let mut shape = Shape { rank: 0, dims: ptr::null_mut() };
            check(unsafe { (self.api.tensor_shape)(output, &mut shape) }, "get SISR shape")?;
            let dimensions = if shape.rank == 4 && !shape.dims.is_null() {
                Some(unsafe { std::slice::from_raw_parts(shape.dims, 4) }.to_vec())
            } else { None };
            unsafe { let _ = (self.api.static_shape_free)(&mut shape); }
            let dimensions = dimensions.ok_or_else(|| anyhow!("NeoArc SISR output must be NCHW rank 4"))?;
            ensure!(dimensions[0] == 1 && dimensions[1] == 3 && dimensions[2] > 0 && dimensions[3] > 0,
                "NeoArc SISR output shape unsupported: {dimensions:?}");
            let (ow, oh) = (usize::try_from(dimensions[3])?, usize::try_from(dimensions[2])?);
            let len = ow.checked_mul(oh).and_then(|v| v.checked_mul(3))
                .filter(|v| *v <= isize::MAX as usize / 4)
                .ok_or_else(|| anyhow!("NeoArc SISR output overflow"))?;
            let mut element_type = 0;
            check(unsafe { (self.api.tensor_type)(output, &mut element_type) }, "get SISR output type")?;
            let mut data = ptr::null_mut();
            check(unsafe { (self.api.tensor_data)(output, &mut data) }, "read SISR output")?;
            ensure!(!data.is_null(), "NeoArc SISR output null");
            match element_type {
                3 => consume(ow, oh, ImageOutput::F16(unsafe { std::slice::from_raw_parts(data.cast(), len) })),
                4 => consume(ow, oh, ImageOutput::F32(unsafe { std::slice::from_raw_parts(data.cast(), len) })),
                other => anyhow::bail!("NeoArc SISR output type unsupported: {other}"),
            }
        })();
        unsafe {
            if !output.is_null() { (self.api.tensor_free)(output); }
            (self.api.tensor_free)(tensor);
        }
        result
    }

    /// Experimental phase-throughput probe. Every request owns its input and
    /// output until completion; outputs are collected in timestep order.
    pub(crate) fn run_many_async_with_output<R>(
        &mut self, width: usize, height: usize, input: &[f16], ts: &[f32],
        mut consume: impl FnMut(&[f16]) -> Result<R>,
    ) -> Result<Vec<R>> {
        let plane = width.checked_mul(height).ok_or_else(|| anyhow!("NeoArc size overflow"))?;
        ensure!(input.len() == 7 * plane && !ts.is_empty() && ts.len() <= 4,
            "NeoArc async phase input invalid");
        self.prepare_shape(width, height)?;
        struct Slot { request: Handle, tensor: Handle, output: Handle, input: Vec<f16>, started: bool }
        let mut slots: Vec<Slot> = Vec::with_capacity(ts.len());
        let result = (|| -> Result<Vec<R>> {
            for &t in ts {
                slots.push(Slot { request: ptr::null_mut(), tensor: ptr::null_mut(),
                    output: ptr::null_mut(), input: input.to_vec(), started: false });
                let slot = slots.last_mut().unwrap();
                slot.input[6 * plane..].fill(f16::from_f32(t));
                check(unsafe { (self.api.request_create)(self.compiled, &mut slot.request) },
                    "create async request")?;
                let mut in_dims = [1i64, 7, height as i64, width as i64];
                check(unsafe { (self.api.tensor_from_host)(3, Shape { rank: 4,
                    dims: in_dims.as_mut_ptr() }, slot.input.as_mut_ptr().cast(),
                    &mut slot.tensor) }, "wrap async input")?;
                let mut out_dims = [1i64, 3, height as i64, width as i64];
                check(unsafe { (self.api.tensor_create)(3, Shape { rank: 4,
                    dims: out_dims.as_mut_ptr() }, &mut slot.output) }, "create async output")?;
                check(unsafe { (self.api.set_input)(slot.request, slot.tensor) }, "bind async input")?;
                check(unsafe { (self.api.set_output)(slot.request, 0, slot.output) }, "bind async output")?;
                check(unsafe { (self.api.start_async)(slot.request) }, "start async inference")?;
                slot.started = true;
            }
            let mut outputs = Vec::with_capacity(slots.len());
            for slot in &mut slots {
                check(unsafe { (self.api.wait)(slot.request) }, "wait async inference")?;
                slot.started = false;
                let mut data = ptr::null_mut();
                check(unsafe { (self.api.tensor_data)(slot.output, &mut data) }, "read async output")?;
                ensure!(!data.is_null(), "NeoArc async output is null");
                outputs.push(consume(unsafe {
                    std::slice::from_raw_parts(data.cast::<f16>(), 3 * plane)
                })?);
            }
            Ok(outputs)
        })();
        for slot in &mut slots {
            unsafe {
                if slot.started { let _ = (self.api.wait)(slot.request); }
                if !slot.tensor.is_null() { (self.api.tensor_free)(slot.tensor); }
                if !slot.output.is_null() { (self.api.tensor_free)(slot.output); }
                if !slot.request.is_null() { (self.api.request_free)(slot.request); }
            }
        }
        result
    }

    pub(crate) fn run_many_async(
        &mut self, width: usize, height: usize, input: &[f16], ts: &[f32],
    ) -> Result<Vec<Vec<f16>>> {
        self.run_many_async_with_output(width, height, input, ts, |output| Ok(output.to_vec()))
    }
}

impl Drop for NeoArcSession {
    fn drop(&mut self) {
        log::debug!("neoarc-retire: phase=gl model={}",self.model_path.display());
        self.gl_rgb.take();self.gl_rgb_reference.take();
        log::debug!("neoarc-retire: phase=rgb model={}",self.model_path.display());
        self.rgb.take();
        self.rife_rgb.take();
        log::debug!("neoarc-retire: phase=remote model={}",self.model_path.display());
        self.remote.take();
        log::debug!("neoarc-retire: phase=host model={}",self.model_path.display());
        self.retire_shape();
        self.retire_previous();
        log::debug!("neoarc-retire: phase=core model={}",self.model_path.display());
        // Shared Core stays alive; all per-session model/CL objects are gone.
        log::debug!("neoarc-retire: phase=complete model={}",self.model_path.display());
    }
}
