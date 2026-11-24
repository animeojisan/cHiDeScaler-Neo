//! GPU output conversion and render-thread-owned D3D11/WGL registrations.
use anyhow::{Result,anyhow,ensure};
use std::{ffi::{c_void,CStr},rc::Rc};
use windows::{core::Interface,Win32::Graphics::{Direct3D11::{ID3D11Texture2D,ID3D11Device},Direct3D12::{ID3D12Device,ID3D12Resource},OpenGL::wglGetProcAddress}};
use glow::HasContext;
use super::gl::{GlContext,GpuTex,Dtype};
type Handle=*mut c_void;
type Assemble=unsafe extern "C" fn(Handle,*const Handle,u32,Handle,u32,u32,u32)->i32;
type Pack=unsafe extern "C" fn(Handle,Handle)->i32;
type Create=unsafe extern "C" fn(Handle,u32,u32)->Handle;type Convert=unsafe extern "C" fn(Handle,Handle,i32)->i32;type Get=unsafe extern "C" fn(Handle)->Handle;type Destroy=unsafe extern "C" fn(Handle);type Error=unsafe extern "C" fn()->*const i8;
struct Api{_library:libloading::Library,create:Create,convert:Convert,assemble:Assemble,pack:Pack,get:Get,destroy:Destroy,error:Error}
fn api()->Result<std::sync::Arc<Api>>{
 static API:std::sync::OnceLock<std::sync::Arc<Api>>=std::sync::OnceLock::new();
 if let Some(api)=API.get(){return Ok(api.clone());}
 static INIT:std::sync::Mutex<()>=std::sync::Mutex::new(());let _guard=INIT.lock().map_err(|_|anyhow!("DML helper initialization poisoned"))?;if let Some(api)=API.get(){return Ok(api.clone());}
 // Embedded optional helper: missing VC runtime never prevents Neo startup.
 // Exact bytes are checked before loading from the per-user cache.
 let bytes=include_bytes!(env!("NEO_DML_BRIDGE_DLL"));use sha2::{Digest,Sha256};
 let digest=format!("{:x}",Sha256::digest(bytes));let directory=std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir).join("cHiDeScaler-Neo/cache/dml-wgl").join(digest);std::fs::create_dir_all(&directory)?;
 let path=directory.join("neo_dml_gl_bridge.dll");
 if std::fs::read(&path).ok().as_deref()!=Some(bytes.as_slice()){
  let temp=directory.join(format!("{}.tmp",std::process::id()));std::fs::write(&temp,bytes)?;
  if let Err(error)=std::fs::rename(&temp,&path){let _=std::fs::remove_file(&temp);if std::fs::read(&path).ok().as_deref()!=Some(bytes.as_slice()){return Err(error.into());}}
 }
 let library:libloading::Library=unsafe{libloading::os::windows::Library::load_with_flags(&path,libloading::os::windows::LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|libloading::os::windows::LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)}?.into();
 let api=std::sync::Arc::new(Api{create:unsafe{*library.get(b"neo_dml_texture_create\0")?},convert:unsafe{*library.get(b"neo_dml_texture_convert\0")?},assemble:unsafe{*library.get(b"neo_dml_interp_assemble\0")?},pack:unsafe{*library.get(b"neo_dml_texture_pack\0")?},get:unsafe{*library.get(b"neo_dml_texture_get\0")?},destroy:unsafe{*library.get(b"neo_dml_texture_destroy\0")?},error:unsafe{*library.get(b"neo_dml_texture_error\0")?},_library:library});
 let _=API.set(api.clone());Ok(API.get().cloned().unwrap_or(api))
}
pub(super) struct Converter{api:std::sync::Arc<Api>,handle:Handle,pub size:(i32,i32),device:usize,validated_context:usize}
unsafe impl Send for Converter{}
impl Converter{
 pub fn new(device:&ID3D12Device,w:i32,h:i32)->Result<Self>{ensure!(surface_budget(device,w,h),"insufficient DXGI memory headroom for GPU output surface");let api=api()?;let handle=unsafe{(api.create)(device.as_raw(),w as u32,h as u32)};ensure!(!handle.is_null(),"DML shared texture creation: {}",error(&api));Ok(Self{api,handle,size:(w,h),device:device.as_raw()as usize,validated_context:0})}
 pub fn input_texture(&self)->Result<ID3D11Texture2D>{let raw=unsafe{(self.api.get)(self.handle)};Ok(unsafe{ID3D11Texture2D::from_raw_borrowed(&raw)}.ok_or_else(||anyhow!("missing DML input texture"))?.clone())}
 pub fn pack(&mut self,destination:&ID3D12Resource)->Result<()>{ensure!(unsafe{(self.api.pack)(self.handle,destination.as_raw())}==0,"DML GPU input pack: {}",error(&self.api));Ok(())}
 pub fn assemble(&mut self,sources:&[ID3D12Resource],target:&ID3D12Resource,padded:(i32,i32),phase:f32)->Result<()>{let raw:Vec<Handle>=sources.iter().map(|s|s.as_raw()).collect();let bits=half::f16::from_f32(phase.clamp(0.,1.)).to_bits()as u32;ensure!(unsafe{(self.api.assemble)(self.handle,raw.as_ptr(),raw.len()as u32,target.as_raw(),padded.0 as u32,padded.1 as u32,bits)}==0,"DML GPU interpolation pack: {}",error(&self.api));Ok(())}
 pub fn validated_current_context(&self)->bool{let current=unsafe{windows::Win32::Graphics::OpenGL::wglGetCurrentContext()}.0 as usize;current!=0&&self.validated_context==current}
 pub fn mark_validated(&mut self){self.validated_context=unsafe{windows::Win32::Graphics::OpenGL::wglGetCurrentContext()}.0 as usize;}
 pub fn matches(&self,device:&ID3D12Device,size:(i32,i32))->bool{self.size==size&&self.device==device.as_raw()as usize}
 pub fn convert(&mut self,source:&ID3D12Resource,uav:bool)->Result<ID3D11Texture2D>{ensure!(unsafe{(self.api.convert)(self.handle,source.as_raw(),i32::from(uav))}==0,"DML GPU conversion: {}",error(&self.api));let raw=unsafe{(self.api.get)(self.handle)};Ok(unsafe{ID3D11Texture2D::from_raw_borrowed(&raw)}.ok_or_else(||anyhow!("missing DML texture"))?.clone())}
}
impl Drop for Converter{fn drop(&mut self){unsafe{(self.api.destroy)(self.handle)}}}
fn error(api:&Api)->String{unsafe{CStr::from_ptr((api.error)())}.to_string_lossy().into_owned()}
fn surface_budget(device:&ID3D12Device,w:i32,h:i32)->bool{
 use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1,IDXGIFactory4,IDXGIAdapter3,DXGI_QUERY_VIDEO_MEMORY_INFO,DXGI_MEMORY_SEGMENT_GROUP_LOCAL};
 let result=(||->windows::core::Result<bool>{unsafe{let factory:IDXGIFactory4=CreateDXGIFactory1()?;let adapter:IDXGIAdapter3=factory.EnumAdapterByLuid(device.GetAdapterLuid())?;let mut info=DXGI_QUERY_VIDEO_MEMORY_INFO::default();adapter.QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&mut info)?;let reserve=w as u64*h as u64*8+64*1024*1024;let allowed=info.Budget>0&&info.CurrentUsage.saturating_add(reserve)<info.Budget/100*85;log::info!("directml-wgl-surface-budget: output={}x{} usage_bytes={} budget_bytes={} reserve_bytes={} allowed={}",w,h,info.CurrentUsage,info.Budget,reserve,allowed);Ok(allowed)}})();result.unwrap_or(false)
}
type Open=unsafe extern "system" fn(Handle)->Handle;type Close=unsafe extern "system" fn(Handle)->i32;type Register=unsafe extern "system" fn(Handle,Handle,u32,u32,u32)->Handle;type Unregister=unsafe extern "system" fn(Handle,Handle)->i32;type Lock=unsafe extern "system" fn(Handle,i32,*const Handle)->i32;
pub(super) struct SharedImage{gl:Rc<glow::Context>,_device:ID3D11Device,source:ID3D11Texture2D,texture:glow::Texture,device:Handle,object:Handle,close:Close,unregister:Unregister,lock:Lock,unlock:Lock,pub bytes:usize,locked:bool}
impl SharedImage{
 pub fn matches(&self,source:&ID3D11Texture2D)->bool{self.source.as_raw()==source.as_raw()}
 pub fn new(gl:Rc<glow::Context>,source:&ID3D11Texture2D,w:i32,h:i32)->Result<Self>{
  fn load(name:&CStr)->Result<*const c_void>{let f=unsafe{wglGetProcAddress(windows::core::PCSTR(name.as_ptr().cast()))}.ok_or_else(||anyhow!("WGL interop unavailable"))?;let p=f as *const c_void;ensure!((p as usize)>3&&p as usize!=usize::MAX,"invalid WGL entry");Ok(p)}
  let(open,close,register,unregister,lock,unlock):(Open,Close,Register,Unregister,Lock,Lock)=unsafe{(std::mem::transmute(load(c"wglDXOpenDeviceNV")?),std::mem::transmute(load(c"wglDXCloseDeviceNV")?),std::mem::transmute(load(c"wglDXRegisterObjectNV")?),std::mem::transmute(load(c"wglDXUnregisterObjectNV")?),std::mem::transmute(load(c"wglDXLockObjectsNV")?),std::mem::transmute(load(c"wglDXUnlockObjectsNV")?))};
  let device11=unsafe{source.GetDevice()?};let device=unsafe{open(device11.as_raw())};ensure!(!device.is_null(),"WGL DML device rejected");let texture=match unsafe{gl.create_texture()}{Ok(t)=>t,Err(e)=>{unsafe{close(device);}return Err(anyhow!(e));}};
  let object=unsafe{register(device,source.as_raw(),texture.0.get(),glow::TEXTURE_2D,1)};if object.is_null(){unsafe{gl.delete_texture(texture);close(device);}return Err(anyhow!("WGL DML texture registration rejected"));}
  Ok(Self{gl,_device:device11,source:source.clone(),texture,device,object,close,unregister,lock,unlock,bytes:w as usize*h as usize*4,locked:false})
 }
 pub fn upload(&mut self,gc:&mut GlContext,input:GpuTex)->Result<()>{
  ensure!(unsafe{(self.lock)(self.device,1,&self.object)}!=0,"WGL input lock failed");self.locked=true;
  let result=(||->Result<()>{let program=gc.program("#version 330\nout vec4 frag;uniform sampler2D tex;void main(){frag=texelFetch(tex,ivec2(gl_FragCoord.xy),0);}").map_err(anyhow::Error::msg)?;
   let target=GpuTex{tex:self.texture,..input};gc.bind_target(target);
   unsafe{let gl=&gc.gl;ensure!(gl.check_framebuffer_status(glow::FRAMEBUFFER)==glow::FRAMEBUFFER_COMPLETE,"WGL input framebuffer incomplete");gl.use_program(Some(program));gl.active_texture(glow::TEXTURE0);gl.bind_texture(glow::TEXTURE_2D,Some(input.tex));gl.uniform_1_i32(gl.get_uniform_location(program,"tex").as_ref(),0);gl.bind_vertex_array(Some(gc.quad_vao));gl.draw_arrays(glow::TRIANGLE_STRIP,0,4);gl.bind_vertex_array(None);}Ok(())})();
  gc.unbind_target();gc.finish();let unlocked=unsafe{(self.unlock)(self.device,1,&self.object)}!=0;if unlocked{self.locked=false;}ensure!(unlocked,"WGL input unlock failed");result
 }
 pub fn copy(&mut self,gc:&mut GlContext,w:i32,h:i32)->Result<GpuTex>{
  ensure!(unsafe{(self.lock)(self.device,1,&self.object)}!=0,"WGL DML lock failed");self.locked=true;
  let result=(||->Result<GpuTex>{let program=gc.program("#version 330\nout vec4 frag;uniform sampler2D tex;void main(){frag=texelFetch(tex,ivec2(gl_FragCoord.xy),0);}").map_err(anyhow::Error::msg)?;let out=gc.make_tex(w,h,4,Dtype::U8);gc.bind_target(out);unsafe{let gl=&gc.gl;gl.use_program(Some(program));gl.active_texture(glow::TEXTURE0);gl.bind_texture(glow::TEXTURE_2D,Some(self.texture));gl.tex_parameter_i32(glow::TEXTURE_2D,glow::TEXTURE_MIN_FILTER,glow::NEAREST as i32);gl.tex_parameter_i32(glow::TEXTURE_2D,glow::TEXTURE_MAG_FILTER,glow::NEAREST as i32);gl.uniform_1_i32(gl.get_uniform_location(program,"tex").as_ref(),0);gl.bind_vertex_array(Some(gc.quad_vao));gl.draw_arrays(glow::TRIANGLE_STRIP,0,4);gl.bind_vertex_array(None);}gc.unbind_target();gc.finish();Ok(out)})();
  let unlocked=unsafe{(self.unlock)(self.device,1,&self.object)}!=0;if unlocked{self.locked=false;}ensure!(unlocked,"WGL DML unlock failed");result
 }
}
impl Drop for SharedImage{fn drop(&mut self){unsafe{self.gl.finish();if self.locked{(self.unlock)(self.device,1,&self.object);}(self.unregister)(self.device,self.object);self.gl.delete_texture(self.texture);(self.close)(self.device);}}}
