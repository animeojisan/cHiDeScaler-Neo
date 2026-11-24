//! Optional GPU-only Compact runtime. All frame data remains in shared OpenCL buffers.
use super::{Handle,neoarc_graph::CompactPlan};
use anyhow::{Result,anyhow,ensure,Context};
use libloading::Library;
use std::{ffi::{c_char,CStr},path::{Path,PathBuf},ptr,sync::{Arc,Mutex,OnceLock}};
#[repr(C)]
struct Layer {ic:i32,oc:i32,activation:i32,weights:*const u16,bias:*const u16,slope:*const f32}
struct Api {_library:Library,path:PathBuf,create:unsafe extern "C" fn(Handle,Handle,Handle,i32,i32,*const Layer,i32)->Handle,run:unsafe extern "C" fn(Handle,Handle,Handle)->i32,destroy:unsafe extern "C" fn(Handle),error:unsafe extern "C" fn()->*const c_char}
static API:OnceLock<Mutex<Option<Arc<Api>>>>=OnceLock::new();
fn api(path:&Path)->Result<Arc<Api>>{
 let mut state=API.get_or_init(||Mutex::new(None)).lock().map_err(|_|anyhow!("native runtime registry poisoned"))?;
 if let Some(api)=state.as_ref(){ensure!(api.path==path,"native runtime path changed; restart required");return Ok(api.clone());}
 let raw=unsafe{libloading::os::windows::Library::load_with_flags(path,libloading::os::windows::LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|libloading::os::windows::LOAD_LIBRARY_SEARCH_DEFAULT_DIRS|libloading::os::windows::LOAD_LIBRARY_SEARCH_USER_DIRS)}.with_context(||format!("load {}",path.display()))?;
 let library:Library=raw.into();let abi:unsafe extern "C" fn()->u32=unsafe{*library.get(b"nc_abi_version\0")?};ensure!(unsafe{abi()}==2,"unsupported Compact native ABI");
 let api=Arc::new(Api{create:unsafe{*library.get(b"nc_create\0")?},run:unsafe{*library.get(b"nc_run\0")?},destroy:unsafe{*library.get(b"nc_destroy\0")?},error:unsafe{*library.get(b"nc_last_error\0")?},_library:library,path:path.to_owned()});*state=Some(api.clone());Ok(api)
}
pub(super) struct NativeCompact {api:Arc<Api>,handle:Handle}
impl NativeCompact {
 pub(super) fn create(plan:&CompactPlan,path:&Path,context:Handle,queue:Handle,w:usize,h:usize)->Result<Self>{
  let api=api(path)?;let layers:Vec<Layer>=plan.layers.iter().map(|l|Layer{ic:l.ic as i32,oc:l.oc as i32,activation:i32::from(!l.slope.is_empty()),weights:l.weights.as_ptr(),bias:l.bias.as_ptr(),slope:if l.slope.is_empty(){ptr::null()}else{l.slope.as_ptr()}}).collect();
  let handle=unsafe{(api.create)(context,ptr::null_mut(),queue,w as i32,h as i32,layers.as_ptr(),layers.len()as i32)};
  ensure!(!handle.is_null(),"Compact GPU creation failed: {}",error(&api));log::info!("neoarc-compact-native-ready: shape={}x{} scale={} layers={} source_sha={} input=shared-opencl output=shared-opencl layout=nhwc precision=fp16-acc-fp32",w,h,plan.scale,plan.layers.len(),plan.digest);Ok(Self{api,handle})
 }
 pub(super) fn run(&mut self,input:Handle,output:Handle)->Result<()>{ensure!(unsafe{(self.api.run)(self.handle,input,output)}==0,"Compact GPU execution failed: {}",error(&self.api));Ok(())}
}
fn error(api:&Api)->String{let p=unsafe{(api.error)()};if p.is_null(){"unknown native error".to_owned()}else{unsafe{CStr::from_ptr(p)}.to_string_lossy().into_owned()}}
impl Drop for NativeCompact{fn drop(&mut self){unsafe{(self.api.destroy)(self.handle)};}}
