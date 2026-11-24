//! WGC D3D11 -> WGL owned-copy handoff. Never maps capture pixels.
use anyhow::{Result,anyhow,ensure};
use super::gl::{GlContext,GpuTex,Dtype};
use crate::capture::wgc_gpu::{View,Snapshot};
use std::{ffi::c_void,rc::Rc,sync::Weak};
use glow::HasContext;
use windows::{core::Interface,Win32::Graphics::OpenGL::wglGetProcAddress};
type Handle=*mut c_void;
type Open=unsafe extern "system" fn(Handle)->Handle;
type Close=unsafe extern "system" fn(Handle)->i32;
type Register=unsafe extern "system" fn(Handle,Handle,u32,u32,u32)->Handle;
type Unregister=unsafe extern "system" fn(Handle,Handle)->i32;
type Lock=unsafe extern "system" fn(Handle,i32,*const Handle)->i32;
struct Image { snapshot:Weak<Snapshot>,texture:glow::Texture,object:Handle }
pub struct Bridge {gl:Rc<glow::Context>,device:windows::Win32::Graphics::Direct3D11::ID3D11Device,handle:Handle,close:Close,register:Register,unregister:Unregister,lock:Lock,unlock:Lock,images:Vec<Image>,
 /// v998: copies still executing on the GL queue. Each keeps its capture
 /// snapshot alive (the D3D11 pool only reuses a snapshot nobody references),
 /// so the copy no longer needs glFinish (which waited for every earlier GL
 /// command, e.g. 2-3 ms of Anime4K post shaders, on the render thread).
 pending:Vec<(std::sync::Arc<Snapshot>,glow::Fence)>}
// The fence variant gave no measurable gain (the cost is in wglDXLockObjectsNV,
// which drains the GL queue) and unlocking with GL reads still queued is a
// possible source of block artifacts on Intel, so glFinish stays the default.
// CHIDESCALER_CAPTURE_COPY_SYNC=fence opts in to the experiment.
fn capture_copy_finish()->bool{static V:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*V.get_or_init(||std::env::var("CHIDESCALER_CAPTURE_COPY_SYNC").ok().as_deref()!=Some("fence"))}
impl Bridge {
 pub fn new(gl:Rc<glow::Context>,view:&View)->Result<Self> {
  fn load(name:&std::ffi::CStr)->Result<*const c_void>{let p=unsafe{wglGetProcAddress(windows::core::PCSTR(name.as_ptr().cast()))}.ok_or_else(||anyhow!("WGL sharing entry unavailable"))?;let p=p as *const c_void;ensure!((p as usize)>3&&p as usize!=usize::MAX,"invalid WGL sharing entry");Ok(p)}
  let open:Open=unsafe{std::mem::transmute(load(c"wglDXOpenDeviceNV")?)};
  let close:Close=unsafe{std::mem::transmute(load(c"wglDXCloseDeviceNV")?)};
  let register:Register=unsafe{std::mem::transmute(load(c"wglDXRegisterObjectNV")?)};
  let unregister:Unregister=unsafe{std::mem::transmute(load(c"wglDXUnregisterObjectNV")?)};
  let lock:Lock=unsafe{std::mem::transmute(load(c"wglDXLockObjectsNV")?)};
  let unlock:Lock=unsafe{std::mem::transmute(load(c"wglDXUnlockObjectsNV")?)};
  let device=view.snapshot.device.clone();let handle=unsafe{open(device.as_raw())};ensure!(!handle.is_null(),"WGL capture device sharing rejected (adapter/driver mismatch)");
  Ok(Self{gl,device,handle,close,register,unregister,lock,unlock,images:Vec::new(),pending:Vec::new()})
 }
 pub fn matches(&self,view:&View)->bool{self.device.as_raw()==view.snapshot.device.as_raw()}
 fn retire_pending(&mut self){
  // Bounded: at most 6 copies in flight; the oldest is waited for (normally
  // long finished) only when that limit is exceeded.
  let mut index=0;
  while index<self.pending.len(){
   let must_wait=self.pending.len()>6&&index==0;
   let status=unsafe{self.gl.client_wait_sync(self.pending[index].1,0,if must_wait{1_000_000_000}else{0})};
   if status==glow::ALREADY_SIGNALED||status==glow::CONDITION_SATISFIED||status==glow::WAIT_FAILED{
    let(_snapshot,fence)=self.pending.remove(index);unsafe{self.gl.delete_sync(fence);}
   }else{index+=1;}
  }
 }
 pub fn copy(&mut self,gc:&mut GlContext,view:&View,w:i32,h:i32)->Result<GpuTex> {
  self.retire_pending();
  let mut index=0;while index<self.images.len(){if self.images[index].snapshot.strong_count()==0{let old=self.images.remove(index);unsafe{(self.unregister)(self.handle,old.object);self.gl.delete_texture(old.texture);}}else{index+=1;}}
  let index=if let Some(index)=self.images.iter().position(|v|v.snapshot.ptr_eq(&std::sync::Arc::downgrade(&view.snapshot))){index}else {
   ensure!(self.images.len()<24,"too many capture GL registrations");
   let texture=unsafe{self.gl.create_texture().map_err(|e|anyhow!(e))?};
   let object=unsafe{(self.register)(self.handle,view.snapshot.texture.as_raw(),texture.0.get(),glow::TEXTURE_2D,0)};
   if object.is_null(){unsafe{self.gl.delete_texture(texture);}return Err(anyhow!("WGL capture texture registration rejected"));}
   self.images.push(Image{snapshot:std::sync::Arc::downgrade(&view.snapshot),texture,object});self.images.len()-1
  };
  let image=&self.images[index];let object=image.object;let texture=image.texture;
  let t_lock=std::time::Instant::now();
  ensure!(unsafe{(self.lock)(self.handle,1,&object)}!=0,"WGL capture lock failed");
  let lock_ms=t_lock.elapsed().as_secs_f64()*1000.0;
  let result=(||->Result<GpuTex>{
   const FRAG:&str="#version 330\nin vec2 v_uv;out vec4 frag;uniform sampler2D tex;uniform ivec2 origin;uniform ivec2 content;uniform ivec2 blackExtent;\nvoid main(){ivec2 q=ivec2(gl_FragCoord.xy);if(any(greaterThanEqual(q,blackExtent))){frag=vec4(0,0,0,1);return;}ivec2 p=origin+min(q,content-ivec2(1));frag=texelFetch(tex,p,0);}";
   let program=gc.program(FRAG).map_err(|e|anyhow!(e))?;let out=gc.make_tex(w,h,4,Dtype::U8);gc.bind_target(out);
   unsafe{let gl=&gc.gl;gl.use_program(Some(program));gl.active_texture(glow::TEXTURE0);gl.bind_texture(glow::TEXTURE_2D,Some(texture));
    gl.tex_parameter_i32(glow::TEXTURE_2D,glow::TEXTURE_MIN_FILTER,glow::NEAREST as i32);gl.tex_parameter_i32(glow::TEXTURE_2D,glow::TEXTURE_MAG_FILTER,glow::NEAREST as i32);
    gl.uniform_1_i32(gl.get_uniform_location(program,"tex").as_ref(),0);
    gl.uniform_2_i32(gl.get_uniform_location(program,"origin").as_ref(),view.left as i32,view.top as i32);
    gl.uniform_2_i32(gl.get_uniform_location(program,"content").as_ref(),view.content_w as i32,view.content_h as i32);
    let(bw,bh)=view.black_extent.unwrap_or((i32::MAX as u32,i32::MAX as u32));
    gl.uniform_2_i32(gl.get_uniform_location(program,"blackExtent").as_ref(),bw as i32,bh as i32);
    gl.bind_vertex_array(Some(gc.quad_vao));gl.draw_arrays(glow::TRIANGLE_STRIP,0,4);gl.bind_vertex_array(None);
   }gc.unbind_target();
   if capture_copy_finish(){gc.finish();}
   else{
    match unsafe{gc.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE,0)}{
     Ok(fence)=>{unsafe{gc.gl.flush();}self.pending.push((view.snapshot.clone(),fence));}
     Err(_)=>gc.finish(),
    }
   }
   Ok(out)
  })();
  let t_unlock=std::time::Instant::now();
  let unlocked=unsafe{(self.unlock)(self.handle,1,&object)}!=0;
  if log::log_enabled!(log::Level::Debug){
   static N:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
   let n=N.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
   if n%48==0{log::debug!("capture-gl-copy-split: lock_ms={:.3} body_ms={:.3} unlock_ms={:.3} pending={}",lock_ms,t_unlock.duration_since(t_lock).as_secs_f64()*1000.0-lock_ms,t_unlock.elapsed().as_secs_f64()*1000.0,self.pending.len());}
  }
  ensure!(unlocked,"WGL capture unlock failed");result
 }
}
impl Drop for Bridge {fn drop(&mut self){unsafe{self.gl.finish();for(_,fence)in self.pending.drain(..){self.gl.delete_sync(fence);}for image in self.images.drain(..){(self.unregister)(self.handle,image.object);self.gl.delete_texture(image.texture);}(self.close)(self.handle);}}}
