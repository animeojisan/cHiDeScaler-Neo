//! Owned immutable D3D11 capture snapshots. Pixel readback is lazy fallback only.
use anyhow::{Result,ensure,anyhow};
use std::sync::Arc;
use windows::{core::Interface,Win32::Graphics::Direct3D11::*};
use windows_capture::frame::Frame;
#[derive(Clone)]
pub struct View {pub snapshot:Arc<Snapshot>,pub left:u32,pub top:u32,pub content_w:u32,pub content_h:u32,pub black_extent:Option<(u32,u32)>}
pub struct Snapshot {pub texture:ID3D11Texture2D,pub device:ID3D11Device,pub context:ID3D11DeviceContext,pub w:u32,pub h:u32}
// v998 experiment: waiting here did not shorten wglDXLockObjectsNV (it drains
// the GL queue, not D3D11), so it is opt-in only (=on).
fn capture_d3d_wait_enabled()->bool{static V:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*V.get_or_init(||std::env::var("CHIDESCALER_CAPTURE_D3D_WAIT").ok().as_deref()==Some("on"))}
impl Snapshot {
 pub fn copy(frame:&Frame<'_>,rect:(u32,u32,u32,u32),pool:&mut Vec<Arc<Self>>)->Result<Arc<Self>> {
  let mut desc=D3D11_TEXTURE2D_DESC::default();unsafe{frame.as_raw_texture().GetDesc(&mut desc);}
  Self::copy_texture(frame.device(),frame.device_context(),frame.as_raw_texture(),desc,rect,pool)
 }
 pub fn copy_texture(device:&ID3D11Device,context:&ID3D11DeviceContext,source:&ID3D11Texture2D,source_desc:D3D11_TEXTURE2D_DESC,rect:(u32,u32,u32,u32),pool:&mut Vec<Arc<Self>>)->Result<Arc<Self>> {
  ensure!(source_desc.Format==windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,"GPU capture format requires RGBA8");
  ensure!(rect.0.checked_add(rect.2).is_some_and(|v|v<=source_desc.Width)&&rect.1.checked_add(rect.3).is_some_and(|v|v<=source_desc.Height),"capture crop exceeds source");
  let(x,y,w,h)=rect;ensure!(w>0&&h>0,"empty GPU capture");
  let protected:ID3D11Multithread=context.cast()?;
  unsafe{let _=protected.SetMultithreadProtected(true);}
  let slot=pool.iter().find(|p|Arc::strong_count(p)==1&&p.w==w&&p.h==h).cloned();
  let slot=if let Some(slot)=slot{slot}else {
   pool.retain(|p|Arc::strong_count(p)>1||p.w==w&&p.h==h);
   ensure!(pool.len()<24,"GPU capture pool exhausted");
   let mut desc=source_desc;desc.Width=w;desc.Height=h;desc.MipLevels=1;desc.ArraySize=1;
   desc.Usage=D3D11_USAGE_DEFAULT;desc.CPUAccessFlags=0;desc.MiscFlags=D3D11_RESOURCE_MISC_SHARED.0 as u32;
   desc.BindFlags=(D3D11_BIND_SHADER_RESOURCE.0|D3D11_BIND_RENDER_TARGET.0)as u32;
   let mut texture=None;unsafe{device.CreateTexture2D(&desc,None,Some(&mut texture))?;}
   let slot=Arc::new(Self{texture:texture.ok_or_else(||anyhow!("null GPU capture texture"))?,device:device.clone(),context:context.clone(),w,h});
   pool.push(slot.clone());slot
  };
  let region=D3D11_BOX{left:x,top:y,front:0,right:x+w,bottom:y+h,back:1};
  unsafe{context.CopySubresourceRegion(&slot.texture,0,0,0,0,source,0,Some(&region));context.Flush();}
  // v998: wait (on this capture thread, bounded) until the copy has executed.
  // Otherwise the render thread's wglDXLockObjectsNV waited ~3 ms for it, as
  // the D3D11 copy queued behind the GL post-shader work on the GPU.
  // CHIDESCALER_CAPTURE_D3D_WAIT=off restores the previous behaviour.
  if capture_d3d_wait_enabled(){
   let desc=D3D11_QUERY_DESC{Query:D3D11_QUERY_EVENT,MiscFlags:0};
   let mut query=None;
   if unsafe{device.CreateQuery(&desc,Some(&mut query))}.is_ok(){
    if let Some(query)=query{
     unsafe{context.End(&query);}
     let started=std::time::Instant::now();
     loop{
      let mut done:i32=0;
      let status=unsafe{context.GetData(&query,Some((&mut done as *mut i32).cast()),4,0)};
      // S_FALSE (not yet) is also Ok(()); `done` turns non-zero when executed.
      if status.is_err()||done!=0||started.elapsed()>std::time::Duration::from_millis(30){break;}
      std::thread::sleep(std::time::Duration::from_micros(200));
     }
    }
   }
  }
  Ok(slot)
 }
 pub fn read_rgba(&self)->Result<Vec<u8>> {
  let mut desc=D3D11_TEXTURE2D_DESC::default();unsafe{self.texture.GetDesc(&mut desc);}
  desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ.0 as u32;desc.BindFlags=0;desc.MiscFlags=0;
  let mut staging=None;unsafe{self.device.CreateTexture2D(&desc,None,Some(&mut staging))?;}
  let staging=staging.ok_or_else(||anyhow!("null capture readback texture"))?;
  let mut mapped=D3D11_MAPPED_SUBRESOURCE::default();unsafe{self.context.CopyResource(&staging,&self.texture);self.context.Map(&staging,0,D3D11_MAP_READ,0,Some(&mut mapped))?;}
  let mut pixels=vec![0;self.w as usize*self.h as usize*4];
  for y in 0..self.h as usize {unsafe{std::ptr::copy_nonoverlapping(mapped.pData.cast::<u8>().add(y*mapped.RowPitch as usize),pixels.as_mut_ptr().add(y*self.w as usize*4),self.w as usize*4);}}
  unsafe{self.context.Unmap(&staging,0);}
  Ok(pixels)
 }
}
impl View {
 pub fn read_rgba(&self,w:i32,h:i32)->Result<Vec<u8>> {
  let raw=self.snapshot.read_rgba()?;let mut out=vec![0;w as usize*h as usize*4];
  for y in 0..h as usize {for x in 0..w as usize {
   if self.black_extent.is_some_and(|(bw,bh)|x>=bw as usize||y>=bh as usize){out[(y*w as usize+x)*4+3]=255;continue;}

   let sy=self.top as usize+y.min(self.content_h.saturating_sub(1)as usize);let sx=self.left as usize+x.min(self.content_w.saturating_sub(1)as usize);
   let src=(sy*self.snapshot.w as usize+sx)*4;let dst=(y*w as usize+x)*4;
   out[dst..dst+4].copy_from_slice(&raw[src..src+4]);
  }}Ok(out)
 }
}
