//! OpenCL-owned FP16 input/history and GPU RGB output. Host RGB transport remains.
use super::{Api, RemoteArc, Handle, Shape, PartialShape, Dimension, Property, check, cl_check};
use anyhow::{Result, ensure, anyhow};
use libloading::Library;
use std::{ffi::{CStr,CString,c_char,c_void}, path::Path, ptr, time::Instant};

/// v973: OpenVINO GPU compile properties. Diagnostic overrides come from
/// CHIDESCALER_NEOARC_GPU_PROPS="KEY=VALUE;KEY=VALUE" (kept alive for the
/// process, appended after the defaults so a later key wins).
pub(super) fn gpu_extra_props()->&'static [(CString,CString)]{
    static PROPS:std::sync::OnceLock<Vec<(CString,CString)>>=std::sync::OnceLock::new();
    PROPS.get_or_init(||{
        let mut props:Vec<(CString,CString)>=Vec::new();
        if let Ok(text)=std::env::var("CHIDESCALER_NEOARC_GPU_PROPS"){
            for item in text.split(';').filter(|s|!s.trim().is_empty()){
                if let Some((k,v))=item.split_once('='){
                    if let(Ok(k),Ok(v))=(CString::new(k.trim()),CString::new(v.trim())){props.retain(|(old,_)|old!=&k);props.push((k,v));}
                }
            }
            log::info!("neoarc-gpu-props: {}",props.iter().map(|(k,v)|format!("{}={}",k.to_string_lossy(),v.to_string_lossy())).collect::<Vec<_>>().join(","));
        }
        props
    })
}
pub(super) fn push_gpu_props(props:&mut Vec<Property>){
    for(k,v)in gpu_extra_props(){props.push(Property{key:k.as_ptr(),value:v.as_ptr().cast()});}
}

#[derive(Clone,Copy)]
struct Extra {
    program: unsafe extern "system" fn(Handle,u32,*const *const c_char,*const usize,*mut i32)->Handle,
    build: unsafe extern "system" fn(Handle,u32,*const Handle,*const c_char,*mut c_void,*mut c_void)->i32,
    kernel: unsafe extern "system" fn(Handle,*const c_char,*mut i32)->Handle,
    argument: unsafe extern "system" fn(Handle,u32,usize,*const c_void)->i32,
    enqueue: unsafe extern "system" fn(Handle,Handle,u32,*const usize,*const usize,*const usize,u32,*const Handle,*mut Handle)->i32,
    finish: unsafe extern "system" fn(Handle)->i32,
    barrier: unsafe extern "system" fn(Handle,u32,*const Handle,*mut Handle)->i32,
    free_program: unsafe extern "system" fn(Handle)->i32,
    free_kernel: unsafe extern "system" fn(Handle)->i32,
    outputs_size:unsafe extern "C" fn(Handle,*mut usize)->i32,
    output_port: unsafe extern "C" fn(Handle,usize,*mut Handle)->i32,
    port_shape: unsafe extern "C" fn(Handle,*mut Shape)->i32,
    port_type: unsafe extern "C" fn(Handle,*mut i32)->i32,
    free_port: unsafe extern "C" fn(Handle),
    set_input_index:unsafe extern "C" fn(Handle,usize,Handle)->i32,
    import_gl:Option<unsafe extern "system" fn(Handle,u64,u32,i32,u32,*mut i32)->Handle>,
    acquire_gl:Option<unsafe extern "system" fn(Handle,u32,*const Handle,u32,*const Handle,*mut Handle)->i32>,
    release_gl:Option<unsafe extern "system" fn(Handle,u32,*const Handle,u32,*const Handle,*mut Handle)->i32>,
    import_gl_buffer:Option<unsafe extern "system" fn(Handle,u64,u32,*mut i32)->Handle>,
    copy_buffer:Option<unsafe extern "system" fn(Handle,Handle,Handle,usize,usize,usize,u32,*const Handle,*mut Handle)->i32>,
    flush:Option<unsafe extern "system" fn(Handle)->i32>,
}

pub(super) struct RgbIo {
    base: RemoteArc,
    native:Option<super::neoarc_compact::NativeCompact>,
    extra: Extra,
    program: Handle,
    pack: Handle,
    compact16: Handle,
    compact32: Handle,
    rife_pack: Handle,
    gl_pack: Handle,
    gl_interp_pack: Handle,
    gl_interp_vector: Handle,
    gl_image_vector: Handle,
    gl_interp_time: Handle,
    vector_pack: bool,
    input_audit: bool,
    pair_cache: bool,
    compile_cache:Option<CString>,
    parked_shape:Option<(usize,usize,usize)>,
    pair_images:Vec<(u32,Handle,bool)>,
    image_cache:std::collections::VecDeque<SharedImage>,
    image_cache_enabled:bool,
    image_cache_hits:u64,
    image_cache_misses:u64,
    pair_packed: bool,
    gl_compact16: Handle,
    gl_compact32: Handle,
    gl_mode: bool,
    queued_pack: bool,
    source: Handle,
    packed_rgb: Handle,
    input: [Handle;2],
    tensors: [Handle;2],
    aux_outputs:Vec<(Handle,Handle)>,
    encoder:Option<EncoderGraphs>,
    encoder_cache:std::path::PathBuf,
    encoder_valid:bool,
    // v963: RIFE boost (trunk once per pair -> state, head per phase).
    boost:Option<BoostGraphs>,
    boost_source:Option<(std::path::PathBuf,std::path::PathBuf)>,
    boost_enabled:bool,
    boost_valid:bool,
    boost_force:Option<bool>,
    // v983: boost scene-cut gate (judged once per pair on the packed input).
    scene_thumb:Handle,
    cut_copy16:Handle,
    cut_copy32:Handle,
    thumb_mem:Handle,
    scene_gate:bool,
    pair_cut:Option<bool>,
    scene_stats:SceneStats,
    // v983: one-kernel boost head for the Lite RIFE layout. [state f16/f32][out f16/f32].
    boost_heads:[Handle;4],
    native_head_candidate:bool,
    native_head:bool,
    // v985: endpoint input through GL pixel buffers (persistent imports by name).
    gl_interp_buffer:Handle,
    buffer_imports:Vec<(u32,usize,Handle)>,
    pair_buffers:Vec<Handle>,
    // v985b: endpoints copied (on the GL owner thread) into CL-owned memory.
    gl_interp_staged:Handle,
    endpoint:Handle,
    endpoint_bytes:usize,
    staged:bool,
    phase_count:usize,
    shape: Option<(usize,usize,usize)>,
    output_size: (usize,usize),
    output_type: i32,
    current: usize,
    valid: bool,
}
struct SharedImage {allocation:u64,name:u32,flags:u64,memory:Handle,bytes:usize}
unsafe impl Send for RgbIo {}

/// v983: thumbnail geometry and thresholds of the boost scene-cut gate. Same
/// test as NeoAMD v133 (`boost_scene_cut`) and neo_rife_motion::scene_gate_host.
pub(super) const SCENE_THUMB_W:usize=64;
pub(super) const SCENE_THUMB_H:usize=36;
const SCENE_THUMB_PLANE:usize=SCENE_THUMB_W*SCENE_THUMB_H;
const SCENE_MAD:f32=0.17;
const SCENE_MEAN_DIFF:f32=0.03;
const SCENE_CORR:f32=0.65;

/// Statistics of one pair: (mean abs diff, mean abs channel-mean diff, centered
/// correlation). `thumb` holds frame 0 RGB planes then frame 1 RGB planes.
pub(super) fn scene_cut_metrics(thumb:&[f32])->(f32,f32,f32){
    let plane=SCENE_THUMB_PLANE;debug_assert_eq!(thumb.len(),6*plane);
    let(r0,r1)=thumb.split_at(3*plane);
    let(mut mean0,mut mean1,mut mad)=([0.0f32;3],[0.0f32;3],0.0f32);
    for c in 0..3{for k in c*plane..(c+1)*plane{mean0[c]+=r0[k];mean1[c]+=r1[k];mad+=(r0[k]-r1[k]).abs();}}
    for c in 0..3{mean0[c]/=plane as f32;mean1[c]/=plane as f32;}
    let count=(3*plane)as f32;mad/=count;
    let md=((mean0[0]-mean1[0]).abs()+(mean0[1]-mean1[1]).abs()+(mean0[2]-mean1[2]).abs())/3.0;
    let(mut num,mut d0,mut d1)=(0.0f32,0.0f32,0.0f32);
    for c in 0..3{for k in c*plane..(c+1)*plane{let(a,b)=(r0[k]-mean0[c],r1[k]-mean1[c]);num+=a*b;d0+=a*a;d1+=b*b;}}
    let den=((d0/count)*(d1/count)).sqrt();
    let corr=if den>0.0{(num/count)/den}else{f32::NAN};
    (mad,md,corr)
}
/// NaN correlation (a flat frame) never counts as a cut, exactly like NeoAMD.
pub(super) fn scene_cut_decision(mad:f32,md:f32,corr:f32)->bool{mad>SCENE_MAD&&md>SCENE_MEAN_DIFF&&corr<SCENE_CORR}

#[derive(Default)]
struct SceneStats{pairs:u64,cuts:u64,judge_ms:f64}

struct EncoderGraph { model:Handle,compiled:Handle,request:Handle }
struct EncoderGraphs {
    api:Api,
    free_buffer:unsafe extern "system" fn(Handle)->i32,
    graphs:Vec<EncoderGraph>,
    features:Vec<(Handle,Handle)>,
    shapes:Vec<Vec<i64>>,
    first_output:usize,
}
impl EncoderGraphs {
    fn compile(&mut self,io:&RgbIo,core:Handle,path:&Path,shape:Option<(usize,usize)>)->Result<()> {
        let index=self.graphs.len();self.graphs.push(EncoderGraph{model:ptr::null_mut(),compiled:ptr::null_mut(),request:ptr::null_mut()});
        let graph=&mut self.graphs[index];let path=CString::new(path.to_string_lossy().as_bytes())?;
        check(unsafe{(self.api.model_read)(core,path.as_ptr(),ptr::null(),&mut graph.model)},"read encoder graph")?;
        if let Some((w,h))=shape{
            let dims=[1,7,h as i64,w as i64];let mut shape=PartialShape{rank:Dimension{min:0,max:0},dims:ptr::null_mut()};
            check(unsafe{(self.api.shape_create)(4,dims.as_ptr(),&mut shape)},"create encoder input shape")?;
            let status=unsafe{(self.api.model_reshape)(graph.model,shape)};unsafe{(self.api.shape_free)(&mut shape)};check(status,"reshape encoder graph")?;
        }
        let mut props=vec![Property{key:c"PERFORMANCE_HINT".as_ptr(),value:c"LATENCY".as_ptr().cast()}];push_gpu_props(&mut props);
        if let Some(cache)=io.compile_cache.as_ref(){props.push(Property{key:c"CACHE_DIR".as_ptr(),value:cache.as_ptr().cast()});}
        let started=Instant::now();
        let mut status=unsafe{(self.api.compile_remote)(core,graph.model,io.base.remote_context,props.len(),props.as_ptr(),&mut graph.compiled)};
        if status!=0&&io.compile_cache.is_some(){
            if !graph.compiled.is_null(){unsafe{(self.api.compiled_free)(graph.compiled)};graph.compiled=ptr::null_mut();}
            props.truncate(1);status=unsafe{(self.api.compile_remote)(core,graph.model,io.base.remote_context,props.len(),props.as_ptr(),&mut graph.compiled)};
        }
        check(status,"compile encoder graph")?;
        log::info!("neoarc-encoder-compile: part={} compile_ms={:.3} cache={}",index,started.elapsed().as_secs_f64()*1000.0,io.compile_cache.is_some());Ok(())
    }
    fn create(io:&RgbIo,core:Handle,path:&Path,w:usize,h:usize)->Result<Option<Self>> {
        let Some(plan)=super::neoarc_graph::EncoderPlan::discover(&io.encoder_cache,path,w,h)? else{return Ok(None)};
        let mut result=Self{api:io.base.api,free_buffer:io.base.cl.buffer_free,graphs:Vec::new(),features:Vec::new(),shapes:Vec::new(),first_output:plan.output_count};
        result.compile(io,core,&plan.full()?,Some((w,h)))?;
        let mut total=0usize;
        for index in 0..plan.features.len(){
            let mut port=ptr::null_mut();check(unsafe{(io.extra.output_port)(result.graphs[0].compiled,plan.output_count+index,&mut port)},"get encoder output port")?;
            let details=(||->Result<Vec<i64>>{
                let mut shape=Shape{rank:0,dims:ptr::null_mut()};check(unsafe{(io.extra.port_shape)(port,&mut shape)},"read encoder output shape")?;
                let dims=if shape.rank==4&&!shape.dims.is_null(){unsafe{std::slice::from_raw_parts(shape.dims,4)}.to_vec()}else{Vec::new()};unsafe{let _=(io.base.api.static_shape_free)(&mut shape);}
                let mut kind=0;check(unsafe{(io.extra.port_type)(port,&mut kind)},"read encoder output precision")?;
                ensure!(kind==3&&dims.len()==4&&(1..=2).contains(&dims[0])&&dims[1]==4&&dims[2]>0&&dims[3]>0,"encoder feature shape/type unsupported");
                Ok(dims)
            })();unsafe{(io.extra.free_port)(port)};let dims=details?;
            let bytes=dims.iter().try_fold(2usize,|n,d|usize::try_from(*d).ok().and_then(|v|n.checked_mul(v))).ok_or_else(||anyhow!("encoder size overflow"))?;
            total=total.checked_add(bytes).filter(|n|*n<=256*1024*1024).ok_or_else(||anyhow!("encoder feature budget exceeded"))?;result.shapes.push(dims);
        }
        result.compile(io,core,&plan.tail(w,h,&result.shapes)?,None)?;result.bind(io)?;
        log::info!("neoarc-encoder-cache-ready: shape={}x{} branches={} feature_bytes={} reuse=within-pair-only x2=original-graph host_feature_bytes=0",w,h,result.shapes.len(),total);
        Ok(Some(result))
    }
    fn park(&mut self){
        for graph in &mut self.graphs{if !graph.request.is_null(){unsafe{(self.api.request_free)(graph.request)};graph.request=ptr::null_mut();}}
        for(memory,tensor)in self.features.drain(..){unsafe{(self.api.tensor_free)(tensor);let _=(self.free_buffer)(memory);}}
    }
    fn bind(&mut self,io:&RgbIo)->Result<()> {
        self.park();
        for dims in &self.shapes{
            let count=dims.iter().try_fold(2usize,|n,d|usize::try_from(*d).ok().and_then(|v|n.checked_mul(v))).ok_or_else(||anyhow!("feature size overflow"))?;
            let memory=io.buffer(count)?;let mut dims=dims.clone();let tensor=match io.tensor(memory,3,&mut dims){Ok(t)=>t,Err(e)=>{unsafe{let _=(self.free_buffer)(memory);}return Err(e)}};self.features.push((memory,tensor));
        }
        for graph in &mut self.graphs{
            check(unsafe{(self.api.request_create)(graph.compiled,&mut graph.request)},"create encoder request")?;
            check(unsafe{(self.api.set_output)(graph.request,0,io.base.output_tensor)},"bind encoder RGB output")?;
            ensure!(self.first_output==io.aux_outputs.len()+1,"encoder side output layout differs");
            for(index,(_,tensor))in io.aux_outputs.iter().enumerate(){check(unsafe{(self.api.set_output)(graph.request,index+1,*tensor)},"bind encoder side output")?;}
        }
        for(index,(_,tensor))in self.features.iter().enumerate(){
            check(unsafe{(self.api.set_output)(self.graphs[0].request,self.first_output+index,*tensor)},"bind shared encoder output")?;
            check(unsafe{(io.extra.set_input_index)(self.graphs[1].request,index+1,*tensor)},"bind shared encoder feature")?;
        }
        Ok(())
    }
}
impl Drop for EncoderGraphs {
    fn drop(&mut self){self.park();for graph in self.graphs.drain(..){unsafe{if !graph.compiled.is_null(){(self.api.compiled_free)(graph.compiled);}if !graph.model.is_null(){(self.api.model_free)(graph.model);}}}}
}

/// v993: everything a fixed-graph compile needs, detached from `RgbIo` so it
/// can run on a helper thread (OpenVINO compile_model is thread-safe).
struct RawCompile{api:Api,core:Handle,remote:Handle,cache:Option<CString>}
unsafe impl Send for RawCompile {}
/// Two compiled boost graphs travelling back from the helper thread.
struct SendGraphs(EncoderGraph,EncoderGraph);
unsafe impl Send for SendGraphs {}
fn free_graph(api:&Api,graph:&EncoderGraph){
    unsafe{if !graph.request.is_null(){(api.request_free)(graph.request);}if !graph.compiled.is_null(){(api.compiled_free)(graph.compiled);}if !graph.model.is_null(){(api.model_free)(graph.model);}}
}
/// v993: trunk/head compile running in parallel with the main model compile.
/// Dropping an unclaimed job joins it and frees its graphs (no leak on `?`).
struct BoostCompileJob{api:Api,handle:Option<std::thread::JoinHandle<Result<SendGraphs>>>}
impl BoostCompileJob{
    fn spawn(raw:RawCompile,trunk:std::path::PathBuf,head:std::path::PathBuf)->Self{
        let api=raw.api;
        let handle=std::thread::Builder::new().name("neoarc-boost-compile".into()).spawn(move||->Result<SendGraphs>{
            let raw=raw;
            let trunk=compile_fixed_graph_raw(&raw,&trunk,"trunk")?;
            match compile_fixed_graph_raw(&raw,&head,"head"){Ok(head)=>Ok(SendGraphs(trunk,head)),Err(error)=>{free_graph(&raw.api,&trunk);Err(error)}}
        }).ok();
        Self{api,handle}
    }
    fn take(mut self)->Result<SendGraphs>{
        let handle=self.handle.take().ok_or_else(||anyhow!("boost compile thread unavailable"))?;
        handle.join().map_err(|_|anyhow!("boost compile thread panicked"))?
    }
}
impl Drop for BoostCompileJob{
    fn drop(&mut self){if let Some(handle)=self.handle.take(){if let Ok(Ok(graphs))=handle.join(){free_graph(&self.api,&graphs.0);free_graph(&self.api,&graphs.1);}}}
}

/// v963: compile one fixed-shape graph on the shared remote context.
fn compile_fixed_graph(io:&RgbIo,core:Handle,path:&Path,label:&str)->Result<EncoderGraph> {
    compile_fixed_graph_raw(&RawCompile{api:io.base.api,core,remote:io.base.remote_context,cache:io.compile_cache.clone()},path,label)
}
fn compile_fixed_graph_raw(raw:&RawCompile,path:&Path,label:&str)->Result<EncoderGraph> {
    let(api,core)=(raw.api,raw.core);let mut graph=EncoderGraph{model:ptr::null_mut(),compiled:ptr::null_mut(),request:ptr::null_mut()};
    let text=CString::new(path.to_string_lossy().as_bytes())?;
    let result=(||->Result<()>{
        check(unsafe{(api.model_read)(core,text.as_ptr(),ptr::null(),&mut graph.model)},"read boost graph")?;
        let mut props=vec![Property{key:c"PERFORMANCE_HINT".as_ptr(),value:c"LATENCY".as_ptr().cast()}];push_gpu_props(&mut props);
        if let Some(cache)=raw.cache.as_ref(){props.push(Property{key:c"CACHE_DIR".as_ptr(),value:cache.as_ptr().cast()});}
        let started=Instant::now();
        let mut status=unsafe{(api.compile_remote)(core,graph.model,raw.remote,props.len(),props.as_ptr(),&mut graph.compiled)};
        if status!=0&&raw.cache.is_some(){
            if !graph.compiled.is_null(){unsafe{(api.compiled_free)(graph.compiled)};graph.compiled=ptr::null_mut();}
            props.truncate(1);status=unsafe{(api.compile_remote)(core,graph.model,raw.remote,props.len(),props.as_ptr(),&mut graph.compiled)};
        }
        check(status,"compile boost graph")?;
        log::info!("neoarc-rife-boost-compile: part={} compile_ms={:.3} cache={} thread={}",label,started.elapsed().as_secs_f64()*1000.0,raw.cache.is_some(),std::thread::current().name().unwrap_or("main"));Ok(())
    })();
    if let Err(error)=result{
        unsafe{if !graph.compiled.is_null(){(api.compiled_free)(graph.compiled);}if !graph.model.is_null(){(api.model_free)(graph.model);}}
        return Err(error);
    }
    Ok(graph)
}

/// v963: NeoArc RIFE boost. The trunk (motion at t=0.5) runs once per source
/// pair into a resident OpenCL state buffer; the head rebuilds each phase from
/// it straight into the regular output tensor. No host round trip.
struct BoostGraphs {
    api:Api,
    free_buffer:unsafe extern "system" fn(Handle)->i32,
    trunk:EncoderGraph,
    head:EncoderGraph,
    state:Option<(Handle,Handle)>,
    state_dims:Vec<i64>,
    state_kind:i32,
}
impl BoostGraphs {
    fn port(io:&RgbIo,compiled:Handle,index:usize)->Result<(Vec<i64>,i32)> {
        let mut port=ptr::null_mut();check(unsafe{(io.extra.output_port)(compiled,index,&mut port)},"get boost output port")?;
        let details=(||->Result<(Vec<i64>,i32)>{
            let mut shape=Shape{rank:0,dims:ptr::null_mut()};check(unsafe{(io.extra.port_shape)(port,&mut shape)},"read boost output shape")?;
            let dims=if shape.rank==4&&!shape.dims.is_null(){unsafe{std::slice::from_raw_parts(shape.dims,4)}.to_vec()}else{Vec::new()};unsafe{let _=(io.base.api.static_shape_free)(&mut shape);}
            let mut kind=0;check(unsafe{(io.extra.port_type)(port,&mut kind)},"read boost output precision")?;Ok((dims,kind))
        })();unsafe{(io.extra.free_port)(port)};details
    }
    fn create(io:&RgbIo,core:Handle,trunk:&Path,head:&Path,w:usize,h:usize)->Result<Self> {
        let trunk=compile_fixed_graph(io,core,trunk,"trunk")?;
        let head=match compile_fixed_graph(io,core,head,"head"){Ok(g)=>g,Err(e)=>{unsafe{(io.base.api.compiled_free)(trunk.compiled);(io.base.api.model_free)(trunk.model);}return Err(e)}};
        Self::create_with(io,trunk,head,w,h)
    }
    /// v993: build from graphs compiled elsewhere (the parallel compile job).
    fn create_with(io:&RgbIo,trunk:EncoderGraph,head:EncoderGraph,w:usize,h:usize)->Result<Self> {
        let mut result=Self{api:io.base.api,free_buffer:io.base.cl.buffer_free,trunk,head,state:None,state_dims:Vec::new(),state_kind:0};
        let(state_dims,state_kind)=Self::port(io,result.trunk.compiled,0)?;
        ensure!(matches!(state_dims.as_slice(), [1, 5, sh, sw]
            if (h as i64..h as i64 + 64).contains(sh) && (w as i64..w as i64 + 64).contains(sw))
            && matches!(state_kind, 3 | 4), "boost state {state_dims:?} type {state_kind} unsupported");
        let(out_dims,out_kind)=Self::port(io,result.head.compiled,0)?;
        ensure!(out_dims==[1,3,h as i64,w as i64]&&out_kind==io.output_type,"boost head output {out_dims:?} type {out_kind} differs from the model ({})",io.output_type);
        result.state_dims=state_dims;result.state_kind=state_kind;result.bind(io)?;
        Ok(result)
    }
    fn park(&mut self){
        for graph in [&mut self.trunk,&mut self.head]{if !graph.request.is_null(){unsafe{(self.api.request_free)(graph.request)};graph.request=ptr::null_mut();}}
        if let Some((memory,tensor))=self.state.take(){unsafe{(self.api.tensor_free)(tensor);let _=(self.free_buffer)(memory);}}
    }
    fn bind(&mut self,io:&RgbIo)->Result<()> {
        self.park();
        let count=self.state_dims.iter().try_fold(if self.state_kind==3{2usize}else{4},|n,d|usize::try_from(*d).ok().and_then(|v|n.checked_mul(v))).ok_or_else(||anyhow!("boost state size overflow"))?;
        let memory=io.buffer(count)?;let mut dims=self.state_dims.clone();
        let tensor=match io.tensor(memory,self.state_kind,&mut dims){Ok(t)=>t,Err(e)=>{unsafe{let _=(self.free_buffer)(memory);}return Err(e)}};
        self.state=Some((memory,tensor));
        check(unsafe{(self.api.request_create)(self.trunk.compiled,&mut self.trunk.request)},"create boost trunk request")?;
        check(unsafe{(self.api.request_create)(self.head.compiled,&mut self.head.request)},"create boost head request")?;
        check(unsafe{(self.api.set_output)(self.trunk.request,0,tensor)},"bind boost state output")?;
        check(unsafe{(self.api.set_output)(self.head.request,0,io.base.output_tensor)},"bind boost RGB output")?;
        check(unsafe{(io.extra.set_input_index)(self.head.request,1,tensor)},"bind boost state input")?;
        Ok(())
    }
}
impl Drop for BoostGraphs {
    fn drop(&mut self){self.park();for graph in [&mut self.trunk,&mut self.head]{unsafe{if !graph.compiled.is_null(){(self.api.compiled_free)(graph.compiled);}if !graph.model.is_null(){(self.api.model_free)(graph.model);}}}}
}

impl RgbIo {
    pub(super) fn set_boost_source(&mut self,source:Option<(std::path::PathBuf,std::path::PathBuf)>){
        if source!=self.boost_source{unsafe{let _=(self.extra.finish)(self.base.queue);}self.boost.take();self.boost_valid=false;}
        self.boost_source=source;
    }
    pub(super) fn set_boost(&mut self,on:bool){if on!=self.boost_enabled{self.boost_valid=false;}self.boost_enabled=on;}
    pub(super) fn set_boost_force(&mut self,force:Option<bool>){self.boost_force=force;self.boost_valid=false;}
    pub(super) fn has_boost(&self)->bool{self.boost.is_some()}
    /// v983: the host proved the head graph is the plain Lite final stage.
    pub(super) fn set_native_head_candidate(&mut self,on:bool){
        self.native_head_candidate=on&&std::env::var("CHIDESCALER_NEOARC_NATIVE_HEAD").ok().as_deref().map(str::trim)!=Some("off");
        if !self.native_head_candidate{self.native_head=false;}
    }
    /// Kernel for the current boost state/output precision when usable.
    fn native_head_kernel(&self)->Option<(Handle,Handle)> {
        let boost=self.boost.as_ref()?;let(memory,_)=boost.state.as_ref()?;let(w,h,_)=self.shape?;
        if boost.state_dims!=[1,5,h as i64,w as i64]||self.output_size!=(w,h)||w<2||h<2{return None;}
        let index=usize::from(boost.state_kind==4)*2+usize::from(self.output_type==4);
        let kernel=self.boost_heads[index];(!kernel.is_null()).then_some((kernel,*memory))
    }
    pub(super) fn native_head_available(&self)->bool{self.native_head_candidate&&self.native_head_kernel().is_some()}
    pub(super) fn set_native_head(&mut self,on:bool){self.native_head=on&&self.native_head_available();}
    pub(super) fn native_head_active(&self)->bool{self.native_head}
    pub(super) fn boost_source(&self)->Option<&(std::path::PathBuf,std::path::PathBuf)>{self.boost_source.as_ref()}
    /// v985: exact check of the native head after a boosted run of `phase`.
    /// Reads the packed endpoints, the trunk state and the RGB output back and
    /// recomputes the Lite final stage on the CPU in f64. Returns (max, mean)
    /// absolute difference of the clamped RGB. The ONNX head itself runs in
    /// the GPU plugin's FP16 precision (grid coordinates lose up to ~0.3 px at
    /// 1280 wide), so it is a poor bit-level reference for the FP32 kernel.
    pub(super) fn native_head_cpu_check(&self,phase:f32)->Result<(f32,f64)> {
        let(_,state)=self.native_head_kernel().ok_or_else(||anyhow!("native head unavailable"))?;
        let(w,h,_)=self.shape.ok_or_else(||anyhow!("shape missing"))?;let n=w*h;
        let kind=self.boost.as_ref().map(|b|b.state_kind).unwrap_or(3);
        let read16=|memory:Handle,count:usize|->Result<Vec<f32>>{
            let mut v=vec![half::f16::ZERO;count];
            cl_check(unsafe{(self.base.cl.read)(self.base.queue,memory,1,0,count*2,v.as_mut_ptr().cast(),0,ptr::null(),ptr::null_mut())},"read native head check")?;
            Ok(v.iter().map(|x|x.to_f32()).collect())
        };
        let read32=|memory:Handle,count:usize|->Result<Vec<f32>>{
            let mut v=vec![0f32;count];
            cl_check(unsafe{(self.base.cl.read)(self.base.queue,memory,1,0,count*4,v.as_mut_ptr().cast(),0,ptr::null(),ptr::null_mut())},"read native head check")?;
            Ok(v)
        };
        let input=read16(self.input[0],6*n)?;
        let state=if kind==3{read16(state,5*n)?}else{read32(state,5*n)?};
        let output=if self.output_type==3{read16(self.base.output_mem,3*n)?}else{read32(self.base.output_mem,3*n)?};
        let t=f64::from(half::f16::from_f32(phase.clamp(0.0,1.0)).to_f32());
        let(wf,hf)=((w-1)as f64,(h-1)as f64);
        let sample=|plane:&[f32],gx:f64,gy:f64|->f64{
            let fx=((gx+1.0)*0.5*wf).clamp(0.0,wf);let fy=((gy+1.0)*0.5*hf).clamp(0.0,hf);
            let(x0,y0)=(fx.floor()as usize,fy.floor()as usize);let(x1,y1)=((x0+1).min(w-1),(y0+1).min(h-1));
            let(ax,ay)=(fx-x0 as f64,fy-y0 as f64);let v=|x:usize,y:usize|f64::from(plane[y*w+x]);
            (v(x0,y0)*(1.0-ax)+v(x1,y0)*ax)*(1.0-ay)+(v(x0,y1)*(1.0-ax)+v(x1,y1)*ax)*ay
        };
        let(r,a,b)=(1.0-t,2.0*t,2.0*(1.0-t));let(mut max,mut sum)=(0f32,0f64);
        for p in 0..n{
            let(x,y)=(p%w,p/w);let(bx,by)=(x as f64/wf*2.0-1.0,y as f64/hf*2.0-1.0);
            let s=|c:usize|f64::from(state[c*n+p]);
            let(g0x,g0y,g1x,g1y)=(bx+s(0)*a,by+s(1)*a,bx+s(2)*b,by+s(3)*b);
            let m=s(4);let(w0,w1)=(m*r,(1.0-m)*t);let k=w0/(w0+w1);
            for c in 0..3{
                let v=sample(&input[c*n..(c+1)*n],g0x,g0y)*k+sample(&input[(3+c)*n..(4+c)*n],g1x,g1y)*(1.0-k);
                let d=(v.clamp(0.0,1.0)-f64::from(output[c*n+p]).clamp(0.0,1.0)).abs();
                max=max.max(d as f32);sum+=d;
            }
        }
        Ok((max,sum/(3*n)as f64))
    }
    pub(super) fn disable_boost(&mut self){unsafe{let _=(self.extra.finish)(self.base.queue);}self.boost.take();self.boost_valid=false;}
    pub(super) fn disable_encoder_cache(&mut self){unsafe{let _=(self.extra.finish)(self.base.queue);}self.encoder.take();self.encoder_valid=false;}
    pub(super) fn has_encoder_cache(&self)->bool{self.encoder.is_some()}
    pub(super) fn new(api: Api, core: Handle, device: &CStr, ov: &Library,cache:&Path,allow_disk_cache:bool)->Result<Self> {
        Self::new_context(api,core,device,ov,None,cache,allow_disk_cache)
    }
    pub(super) fn new_gl(api:Api,core:Handle,device:&CStr,ov:&Library,rc:isize,dc:isize,cache:&Path,allow_disk_cache:bool)->Result<Self>{
        Self::new_context(api,core,device,ov,Some((rc,dc)),cache,allow_disk_cache)
    }
    fn new_context(api:Api,core:Handle,device:&CStr,ov:&Library,gl:Option<(isize,isize)>,cache:&Path,allow_disk_cache:bool)->Result<Self>{
        let base=RemoteArc::new_sharing(api,core,device,gl)?;
        macro_rules! cl {($n:literal)=>{unsafe{*base._cl_library.get(concat!($n,"\0").as_bytes())?}}}
        macro_rules! ov {($n:literal)=>{unsafe{*ov.get(concat!($n,"\0").as_bytes())?}}}
        let extra=Extra {program:cl!("clCreateProgramWithSource"),build:cl!("clBuildProgram"),kernel:cl!("clCreateKernel"),
            argument:cl!("clSetKernelArg"),enqueue:cl!("clEnqueueNDRangeKernel"),finish:cl!("clFinish"),barrier:cl!("clEnqueueBarrierWithWaitList"),
            free_program:cl!("clReleaseProgram"),free_kernel:cl!("clReleaseKernel"),
            outputs_size:ov!("ov_compiled_model_outputs_size"),output_port:ov!("ov_compiled_model_output_by_index"),port_shape:ov!("ov_const_port_get_shape"),
            port_type:ov!("ov_port_get_element_type"),free_port:ov!("ov_output_const_port_free"),
            set_input_index:ov!("ov_infer_request_set_input_tensor_by_index"),
            import_gl:unsafe{base._cl_library.get(b"clCreateFromGLTexture\0").ok().map(|s:libloading::Symbol<unsafe extern "system" fn(Handle,u64,u32,i32,u32,*mut i32)->Handle>|*s)},
            acquire_gl:unsafe{base._cl_library.get(b"clEnqueueAcquireGLObjects\0").ok().map(|s:libloading::Symbol<unsafe extern "system" fn(Handle,u32,*const Handle,u32,*const Handle,*mut Handle)->i32>|*s)},
            release_gl:unsafe{base._cl_library.get(b"clEnqueueReleaseGLObjects\0").ok().map(|s:libloading::Symbol<unsafe extern "system" fn(Handle,u32,*const Handle,u32,*const Handle,*mut Handle)->i32>|*s)},
            import_gl_buffer:unsafe{base._cl_library.get(b"clCreateFromGLBuffer\0").ok().map(|s:libloading::Symbol<unsafe extern "system" fn(Handle,u64,u32,*mut i32)->Handle>|*s)},
            copy_buffer:unsafe{base._cl_library.get(b"clEnqueueCopyBuffer\0").ok().map(|s:libloading::Symbol<unsafe extern "system" fn(Handle,Handle,Handle,usize,usize,usize,u32,*const Handle,*mut Handle)->i32>|*s)},
            flush:unsafe{base._cl_library.get(b"clFlush\0").ok().map(|s:libloading::Symbol<unsafe extern "system" fn(Handle)->i32>|*s)}};
        let mut result=Self{native:None,compile_cache:if allow_disk_cache{super::compilation_cache(cache)}else{None},parked_shape:None,pair_images:Vec::new(),image_cache:std::collections::VecDeque::new(),image_cache_enabled:std::env::var("CHIDESCALER_NEOARC_GL_IMPORT_CACHE").ok().as_deref()!=Some("off"),image_cache_hits:0,image_cache_misses:0,base,extra,program:ptr::null_mut(),pack:ptr::null_mut(),compact16:ptr::null_mut(),compact32:ptr::null_mut(),
            rife_pack:ptr::null_mut(),gl_pack:ptr::null_mut(),gl_interp_pack:ptr::null_mut(),gl_interp_vector:ptr::null_mut(),gl_image_vector:ptr::null_mut(),gl_interp_time:ptr::null_mut(),
            input_audit:std::env::var("CHIDESCALER_NEOARC_INPUT_AUDIT").ok().as_deref()==Some("on"),
            vector_pack:std::env::var("CHIDESCALER_NEOARC_GL_PACK").ok().as_deref()==Some("vector"),
            pair_cache:std::env::var("CHIDESCALER_NEOARC_GL_PAIR_CACHE").ok().as_deref()!=Some("off"),pair_packed:false,gl_compact16:ptr::null_mut(),gl_compact32:ptr::null_mut(),gl_mode:gl.is_some(),queued_pack:std::env::var("CHIDESCALER_NEOARC_QUEUE_SYNC").ok().as_deref()!=Some("finish"),
            source:ptr::null_mut(),packed_rgb:ptr::null_mut(),input:[ptr::null_mut();2],tensors:[ptr::null_mut();2],
            aux_outputs:Vec::new(),encoder:None,encoder_cache:cache.to_path_buf(),encoder_valid:false,boost:None,boost_source:None,boost_enabled:false,boost_valid:false,boost_force:None,
            scene_thumb:ptr::null_mut(),cut_copy16:ptr::null_mut(),cut_copy32:ptr::null_mut(),thumb_mem:ptr::null_mut(),
            scene_gate:!matches!(std::env::var("CHIDESCALER_NEOARC_BOOST_SCENE").ok().as_deref().map(str::trim),Some("0"|"off")),
            pair_cut:None,scene_stats:SceneStats::default(),
            boost_heads:[ptr::null_mut();4],native_head_candidate:false,native_head:false,
            gl_interp_buffer:ptr::null_mut(),buffer_imports:Vec::new(),pair_buffers:Vec::new(),
            gl_interp_staged:ptr::null_mut(),endpoint:ptr::null_mut(),endpoint_bytes:0,staged:false,phase_count:2,shape:None,output_size:(0,0),output_type:0,current:0,valid:false};
        let source=include_str!("../../backends/neoarc/rgb_io.cl");
        let text=source.as_ptr().cast();let length=source.len();let mut status=0;
        result.program=unsafe{(extra.program)(result.base.context,1,&text,&length,&mut status)};
        cl_check(status,"create RGB program")?;
        cl_check(unsafe{(extra.build)(result.program,0,ptr::null(),ptr::null(),ptr::null_mut(),ptr::null_mut())},"build RGB program")?;
        for (name,slot) in [(c"neoarc_rgb_input",&mut result.pack),(c"neoarc_rgb_f16_output",&mut result.compact16),(c"neoarc_rgb_f32_output",&mut result.compact32),(c"neoarc_rife_rgb_input",&mut result.rife_pack)] {
            *slot=unsafe{(extra.kernel)(result.program,name.as_ptr(),&mut status)};cl_check(status,"create RGB kernel")?;
        }
        if result.gl_mode {
            ensure!(extra.import_gl.is_some()&&extra.acquire_gl.is_some()&&extra.release_gl.is_some(),"CL/GL sharing entry points missing");
            for(name,slot)in[(c"neoarc_gl_input",&mut result.gl_pack),(c"neoarc_gl_interp_input",&mut result.gl_interp_pack),(c"neoarc_gl_interp_input_vector",&mut result.gl_interp_vector),(c"neoarc_gl_input_vector",&mut result.gl_image_vector),(c"neoarc_gl_interp_time",&mut result.gl_interp_time),(c"neoarc_gl_f16_output",&mut result.gl_compact16),(c"neoarc_gl_f32_output",&mut result.gl_compact32)]{
                *slot=unsafe{(extra.kernel)(result.program,name.as_ptr(),&mut status)};cl_check(status,"create GL IO kernel")?;
            }
            // v983: the scene gate is optional. A missing kernel only turns it off.
            for(name,slot)in[(c"neoarc_scene_thumb",&mut result.scene_thumb),(c"neoarc_cut_copy_f16",&mut result.cut_copy16),(c"neoarc_cut_copy_f32",&mut result.cut_copy32)]{
                *slot=unsafe{(extra.kernel)(result.program,name.as_ptr(),&mut status)};
                if status!=0||slot.is_null(){*slot=ptr::null_mut();result.scene_gate=false;}
            }
            result.gl_interp_buffer=unsafe{(extra.kernel)(result.program,c"neoarc_gl_interp_input_buffer".as_ptr(),&mut status)};
            if status!=0{result.gl_interp_buffer=ptr::null_mut();}
            result.gl_interp_staged=unsafe{(extra.kernel)(result.program,c"neoarc_gl_interp_input_staged".as_ptr(),&mut status)};
            if status!=0{result.gl_interp_staged=ptr::null_mut();}
            let[h0,h1,h2,h3]=&mut result.boost_heads;
            for(name,slot)in[(c"neoarc_boost_head_s16_o16",h0),(c"neoarc_boost_head_s16_o32",h1),(c"neoarc_boost_head_s32_o16",h2),(c"neoarc_boost_head_s32_o32",h3)]{
                *slot=unsafe{(extra.kernel)(result.program,name.as_ptr(),&mut status)};
                if status!=0{*slot=ptr::null_mut();}
            }
            if result.scene_gate{log::info!("neoarc-rife-boost-scene-policy: gate=on thumb={}x{} mad>{} md>{} corr<{} cut=copy-endpoint",SCENE_THUMB_W,SCENE_THUMB_H,SCENE_MAD,SCENE_MEAN_DIFF,SCENE_CORR);}
        }
        Ok(result)
    }
    pub(super) fn park(&mut self) {
        unsafe{let _=(self.extra.finish)(self.base.queue);}
        let encoder=self.encoder.take().map(|mut e|{e.park();e});
        let boost=self.boost.take().map(|mut b|{b.park();b});
        let shape=self.shape;let native=self.native.take();
        let model=std::mem::replace(&mut self.base.model,ptr::null_mut());
        let compiled=std::mem::replace(&mut self.base.compiled,ptr::null_mut());
        self.retire();self.base.model=model;self.base.compiled=compiled;self.parked_shape=shape;self.encoder=encoder;self.boost=boost;self.native=native;
    }
    fn retire(&mut self) {
        self.begin_interp_pair();
        unsafe {let _=(self.extra.finish)(self.base.queue);}
        self.clear_image_cache();
        self.encoder.take();self.encoder_valid=false;
        self.boost.take();self.boost_valid=false;
        self.native.take();
        self.base.retire_shape();
        for(memory,tensor)in self.aux_outputs.drain(..){unsafe{(self.base.api.tensor_free)(tensor);let _=(self.base.cl.buffer_free)(memory);}}
        for tensor in &mut self.tensors {if !tensor.is_null(){unsafe{(self.base.api.tensor_free)(*tensor)};*tensor=ptr::null_mut();}}
        for memory in &mut self.input {if !memory.is_null(){unsafe{let _=(self.base.cl.buffer_free)(*memory);};*memory=ptr::null_mut();}}
        for memory in [&mut self.source,&mut self.packed_rgb,&mut self.thumb_mem,&mut self.endpoint] {if !memory.is_null(){unsafe{let _=(self.base.cl.buffer_free)(*memory);};*memory=ptr::null_mut();}}
        self.endpoint_bytes=0;
        self.shape=None;self.valid=false;self.pair_packed=false;
    }
    fn buffer(&self,bytes:usize)->Result<Handle> {self.buffer_flags(bytes,1)}
    fn buffer_flags(&self,bytes:usize,flags:u64)->Result<Handle> {
        let mut status=0;
        let buffer=unsafe{(self.base.cl.buffer_create)(self.base.context,flags,bytes,ptr::null_mut(),&mut status)};
        cl_check(status,"allocate RGB IO buffer")?;ensure!(!buffer.is_null(),"null RGB buffer");Ok(buffer)
    }
    fn tensor(&self,memory:Handle,kind:i32,dims:&mut [i64])->Result<Handle> {
        let props=[Property{key:c"SHARED_MEM_TYPE".as_ptr(),value:c"OCL_BUFFER".as_ptr().cast()},Property{key:c"MEM_HANDLE".as_ptr(),value:memory}];
        let mut tensor=ptr::null_mut();
        check(unsafe{(self.base.api.remote_tensor_create)(self.base.remote_context,kind,Shape{rank:dims.len()as i64,dims:dims.as_mut_ptr()},props.len(),props.as_ptr(),&mut tensor)},"wrap RGB IO tensor")?;
        Ok(tensor)
    }
    pub(super) fn prepare_native(&mut self,plan:&super::neoarc_graph::CompactPlan,library:&Path,w:usize,h:usize)->Result<()> {
        if self.shape==Some((w,h,3))&&self.native.is_some(){return Ok(());}
        let resume=self.parked_shape.take()==Some((w,h,3))&&self.native.is_some();
        if !resume{self.retire();}
        ensure!(self.gl_mode,"native Compact requires GL sharing");
        let n=w.checked_mul(h).filter(|n|*n>0&&*n<=i32::MAX as usize/4).ok_or_else(||anyhow!("native input extent overflow"))?;
        let(ow,oh)=(w.checked_mul(plan.scale).ok_or_else(||anyhow!("native width overflow"))?,h.checked_mul(plan.scale).ok_or_else(||anyhow!("native height overflow"))?);
        let on=ow.checked_mul(oh).filter(|n|*n<=i32::MAX as usize/4).ok_or_else(||anyhow!("native output extent overflow"))?;
        if !resume{self.native=Some(super::neoarc_compact::NativeCompact::create(plan,library,self.base.context,self.base.queue,w,h)?);}
        self.input[0]=self.buffer(n*3*2)?;self.base.output_mem=self.buffer(on*3*2)?;
        self.output_size=(ow,oh);self.output_type=3;self.current=0;self.valid=false;self.shape=Some((w,h,3));Ok(())
    }
    pub(super) fn device_identity(&self)->Option<usize>{
        type Info=unsafe extern "system" fn(Handle,u32,usize,*mut c_void,*mut usize)->i32;
        let info=unsafe{self.base._cl_library.get::<Info>(b"clGetCommandQueueInfo\0").ok()?};let mut device:Handle=ptr::null_mut();
        let status=unsafe{info(self.base.queue,0x1091,std::mem::size_of::<Handle>(),(&mut device as *mut Handle).cast(),ptr::null_mut())};
        (status==0&&!device.is_null()).then_some(device as usize)
    }
    /// v987: OpenCL driver version of this queue's device (validation-marker key).
    pub(super) fn driver_version(&self)->Option<String>{
        type Info=unsafe extern "system" fn(Handle,u32,usize,*mut c_void,*mut usize)->i32;
        let device=self.device_identity()? as Handle;
        let info=unsafe{self.base._cl_library.get::<Info>(b"clGetDeviceInfo\0").ok()?};
        let mut size=0usize;
        if unsafe{info(device,0x102D,0,ptr::null_mut(),&mut size)}!=0||size==0||size>4096{return None;}
        let mut bytes=vec![0u8;size];
        if unsafe{info(device,0x102D,size,bytes.as_mut_ptr().cast(),ptr::null_mut())}!=0{return None;}
        Some(String::from_utf8_lossy(&bytes).trim_end_matches('\0').to_string())
    }
    pub(super) fn is_native(&self)->bool{self.native.is_some()}
    pub(super) fn can_resume_native(&self,w:usize,h:usize)->bool{self.native.is_some()&&self.parked_shape==Some((w,h,3))}
    pub(super) fn prepare(&mut self,core:Handle,path:&Path,channels:usize,w:usize,h:usize,f32:bool)->Result<()> {
        if self.shape==Some((w,h,channels)){return Ok(());}
        let resume=self.parked_shape.take()==Some((w,h,channels))&&!self.base.compiled.is_null();
        if !resume{self.retire();}let api=self.base.api;
        ensure!(channels==7||channels==14||(channels>=3&&channels%3==0),"invalid RGB history channels");
        let n=w.checked_mul(h).filter(|n|*n>0&&*n<=i32::MAX as usize/4).ok_or_else(||anyhow!("RGB extent overflow"))?;
        let count=n.checked_mul(channels).filter(|n|*n<=i32::MAX as usize).ok_or_else(||anyhow!("RGB tensor too large"))?;
        // v993: compile the boost trunk/head on a helper thread while the main
        // remote model compiles (cached imports took ~2 s + ~0.9 s serially).
        let boost_job=if !resume&&self.boost.is_none()&&self.gl_mode&&channels==7&&!f32
            &&std::env::var("CHIDESCALER_NEOARC_RIFE_BOOST").ok().as_deref()!=Some("off")
            &&std::env::var("CHIDESCALER_NEOARC_PARALLEL_COMPILE").ok().as_deref()!=Some("off"){
            self.boost_source.clone().map(|(trunk,head)|BoostCompileJob::spawn(
                RawCompile{api:self.base.api,core,remote:self.base.remote_context,cache:self.compile_cache.clone()},trunk,head))
        }else{None};
        if !resume {
        let path=CString::new(path.to_string_lossy().as_bytes())?;
        check(unsafe{(api.model_read)(core,path.as_ptr(),ptr::null(),&mut self.base.model)},"read RGB model")?;
        let mut dims=[1,channels as i64,h as i64,w as i64];
        let mut shape=PartialShape{rank:Dimension{min:0,max:0},dims:ptr::null_mut()};
        check(unsafe{(api.shape_create)(4,dims.as_ptr(),&mut shape)},"shape RGB input")?;
        let status=unsafe{(api.model_reshape)(self.base.model,shape)};unsafe{(api.shape_free)(&mut shape)};api.check_detail(status,"reshape RGB model")?;
        let mut props=vec![Property{key:c"PERFORMANCE_HINT".as_ptr(),value:c"LATENCY".as_ptr().cast()}];push_gpu_props(&mut props);
        if f32{props.push(Property{key:c"INFERENCE_PRECISION_HINT".as_ptr(),value:c"f32".as_ptr().cast()});}
        if std::env::var("CHIDESCALER_NEOARC_PROFILE").ok().as_deref()==Some("on"){props.push(Property{key:c"PERF_COUNT".as_ptr(),value:c"YES".as_ptr().cast()});}
        if let Some(cache)=self.compile_cache.as_ref(){props.push(Property{key:c"CACHE_DIR".as_ptr(),value:cache.as_ptr().cast()});}
        let compile_started=Instant::now();
        let mut status=unsafe{(api.compile_remote)(core,self.base.model,self.base.remote_context,props.len(),props.as_ptr(),&mut self.base.compiled)};
        if status!=0&&self.compile_cache.is_some(){
            if !self.base.compiled.is_null(){unsafe{(api.compiled_free)(self.base.compiled)};self.base.compiled=ptr::null_mut();}
            props.retain(|p|unsafe{CStr::from_ptr(p.key)}!=c"CACHE_DIR");
            log::warn!("neoarc-compile-cache-retry: route=remote action=uncached");
            status=unsafe{(api.compile_remote)(core,self.base.model,self.base.remote_context,props.len(),props.as_ptr(),&mut self.base.compiled)};
        }
        check(status,"compile RGB remote model")?;
        log::info!("neoarc-compile: route=remote shape={}x{} channels={} cache={} compile_ms={:.3}",w,h,channels,self.compile_cache.is_some(),compile_started.elapsed().as_secs_f64()*1000.0);
        }else{log::info!("neoarc-compile: route=remote shape={}x{} channels={} in_memory_graph=true working_buffers=recreated",w,h,channels);}
        let mut dims=[1,channels as i64,h as i64,w as i64];
        let mut port=ptr::null_mut();check(unsafe{(self.extra.output_port)(self.base.compiled,0,&mut port)},"get RGB output port")?;
        let output=(||->Result<(usize,usize,i32)>{
            let mut s=Shape{rank:0,dims:ptr::null_mut()};
            check(unsafe{(self.extra.port_shape)(port,&mut s)},"get RGB output shape")?;
            let d=if s.rank==4&&!s.dims.is_null(){unsafe{std::slice::from_raw_parts(s.dims,4)}.to_vec()}else{Vec::new()};
            unsafe{let _=(api.static_shape_free)(&mut s);}
            ensure!(d.len()==4&&d[0]==1&&d[1]==3&&d[2]>0&&d[3]>0,"RGB output must be NCHW3");
            let mut kind=0;check(unsafe{(self.extra.port_type)(port,&mut kind)},"get RGB output type")?;
            ensure!(matches!(kind,3|4),"unsupported RGB output precision {kind}");
            Ok((usize::try_from(d[3])?,usize::try_from(d[2])?,kind))
        })();unsafe{(self.extra.free_port)(port)};let(ow,oh,kind)=output?;
        let on=ow.checked_mul(oh).filter(|n|*n<=i32::MAX as usize/4).ok_or_else(||anyhow!("RGB output too large"))?;
        if !self.gl_mode{self.source=self.buffer(n*if channels==7{8}else{4})?;self.packed_rgb=self.buffer(on*4)?;}
        for index in 0..if channels==3||channels==7{1}else{2} {
            self.input[index]=self.buffer(count*2)?;self.tensors[index]=self.tensor(self.input[index],3,&mut dims)?;
        }
        self.base.output_mem=self.buffer(on*3*if kind==3{2}else{4})?;
        let mut od=[1,3,oh as i64,ow as i64];self.base.output_tensor=self.tensor(self.base.output_mem,kind,&mut od)?;
        check(unsafe{(api.request_create)(self.base.compiled,&mut self.base.request)},"create RGB request")?;
        check(unsafe{(api.set_output)(self.base.request,0,self.base.output_tensor)},"bind RGB output")?;
        if matches!(channels,7|14)&&std::env::var("CHIDESCALER_NEOARC_AUX_OUTPUT").ok().as_deref()!=Some("host") {
            let mut outputs=0;check(unsafe{(self.extra.outputs_size)(self.base.compiled,&mut outputs)},"get remote output count")?;
            ensure!(outputs<=16,"too many remote side outputs");
            for index in 1..outputs {
                let mut port=ptr::null_mut();check(unsafe{(self.extra.output_port)(self.base.compiled,index,&mut port)},"get remote side output")?;
                let details=(||->Result<(Vec<i64>,i32,usize)>{
                    let mut shape=Shape{rank:0,dims:ptr::null_mut()};check(unsafe{(self.extra.port_shape)(port,&mut shape)},"get side output shape")?;
                    let dims=if (1..=4).contains(&shape.rank)&&!shape.dims.is_null(){unsafe{std::slice::from_raw_parts(shape.dims,shape.rank as usize)}.to_vec()}else{Vec::new()};
                    unsafe{let _=(api.static_shape_free)(&mut shape);}ensure!(!dims.is_empty(),"unsupported side output shape");
                    let count=dims.iter().try_fold(1usize,|total,v|usize::try_from(*v).ok().and_then(|v|total.checked_mul(v))).ok_or_else(||anyhow!("side output overflow"))?;
                    let mut kind=0;check(unsafe{(self.extra.port_type)(port,&mut kind)},"get side output type")?;ensure!(matches!(kind,3|4),"unsupported side output type");
                    let bytes=count.checked_mul(if kind==3{2}else{4}).filter(|v|*v>0&&*v<=1024*1024).ok_or_else(||anyhow!("side output too large"))?;Ok((dims,kind,bytes))
                })();unsafe{(self.extra.free_port)(port)};let(mut dims,kind,bytes)=details?;
                let memory=self.buffer(bytes)?;let tensor=match self.tensor(memory,kind,&mut dims){Ok(t)=>t,Err(e)=>{unsafe{let _=(self.base.cl.buffer_free)(memory);}return Err(e)}};
                self.aux_outputs.push((memory,tensor));check(unsafe{(api.set_output)(self.base.request,index,tensor)},"bind GPU side output")?;
                log::info!("neoarc-side-output-gpu: index={} bytes={} host_readback=false",index,bytes);
            }
        }
        self.output_size=(ow,oh);self.output_type=kind;self.current=0;self.valid=false;self.shape=Some((w,h,channels));
        self.phase_count=2;self.encoder_valid=false;
        if std::env::var("CHIDESCALER_NEOARC_ENCODER_CACHE").ok().as_deref()==Some("off"){self.encoder.take();}
        if self.gl_mode&&channels==7&&!f32&&std::env::var("CHIDESCALER_NEOARC_ENCODER_CACHE").ok().as_deref()!=Some("off") {
            let encoder=self.encoder.take();
            let result=if let Some(mut encoder)=encoder{encoder.bind(self).map(|()|Some(encoder))}
                else if resume{Ok(None)}else{EncoderGraphs::create(self,core,path,w,h)};
            match result{Ok(encoder)=>self.encoder=encoder,Err(error)=>log::warn!("neoarc-encoder-cache-rejected: shape={}x{} reason={error:#} action=original-gpu-graph",w,h)}
        }
        self.boost_valid=false;
        // The canonical request owns and validates its auxiliary outputs.
        // The separately compiled trunk/head only bind motion state and RGB.
        let boost_allowed=self.gl_mode&&channels==7&&!f32
            &&std::env::var("CHIDESCALER_NEOARC_RIFE_BOOST").ok().as_deref()!=Some("off");
        if !boost_allowed{self.boost.take();}
        else if let Some(mut boost)=self.boost.take(){
            match boost.bind(self){Ok(())=>self.boost=Some(boost),Err(error)=>log::warn!("neoarc-rife-boost-fallback: shape={}x{} reason={error:#} stage=rebind action=original-gpu-graph",w,h)}
        }else if let Some((trunk,head))=self.boost_source.clone(){
            let started=Instant::now();
            let created=match boost_job{
                Some(job)=>job.take().and_then(|graphs|{let SendGraphs(t,hd)=graphs;BoostGraphs::create_with(self,t,hd,w,h)}),
                None=>BoostGraphs::create(self,core,&trunk,&head,w,h),
            };
            match created{
                Ok(boost)=>{log::info!("neoarc-rife-boost-ready: shape={}x{} state=gpu-resident build_ms={:.0}",w,h,started.elapsed().as_secs_f64()*1000.0);self.boost=Some(boost);}
                Err(error)=>log::warn!("neoarc-rife-boost-fallback: shape={}x{} reason={error:#} stage=build action=original-gpu-graph",w,h),
            }
        }
        log::info!("neoarc-rgb-io-ready: input={}x{} channels={} output={}x{} precision={} history=gpu-ping-pong host_fp16_bytes=0",w,h,channels,ow,oh,kind);
        Ok(())
    }
    fn arg<T>(&self,kernel:Handle,index:u32,value:&T)->Result<()> {
        cl_check(unsafe{(self.extra.argument)(kernel,index,std::mem::size_of::<T>(),(value as *const T).cast())},"set RGB kernel argument")
    }
    fn dispatch(&self,kernel:Handle,count:usize)->Result<()> {
        cl_check(unsafe{(self.extra.enqueue)(self.base.queue,kernel,1,ptr::null(),&count,ptr::null(),0,ptr::null(),ptr::null_mut())},"enqueue RGB kernel")
    }
    fn submit_pack(&self)->Result<()> {
        if self.queued_pack {
            // Shared OCL_QUEUE: order preprocessing before inference without a host wait.
            cl_check(unsafe{(self.extra.barrier)(self.base.queue,0,ptr::null(),ptr::null_mut())},"queue input packing barrier")
        } else {cl_check(unsafe{(self.extra.finish)(self.base.queue)},"complete input packing")}
    }
    pub(super) fn matches_shape(&self,w:usize,h:usize,channels:usize)->bool{self.shape==Some((w,h,channels))}
    pub(super) fn output_size(&self)->(usize,usize){self.output_size}
    fn clear_image_cache(&mut self){
        for image in self.image_cache.drain(..){unsafe{let _=(self.base.cl.buffer_free)(image.memory);}}
        self.pair_buffers.clear();
        for(_,_,memory)in self.buffer_imports.drain(..){unsafe{let _=(self.base.cl.buffer_free)(memory);}}
    }
    /// v985: GL pixel-buffer endpoints are usable on this context.
    pub(super) fn supports_buffer_input(&self)->bool{
        self.gl_mode&&self.extra.import_gl_buffer.is_some()&&!self.gl_interp_buffer.is_null()
            &&matches!(self.shape,Some((_,_,7)))
    }
    /// v985: like prepare_gl_pair, but the two RIFE endpoints arrive as GL
    /// buffers holding RGBA FP16 pixels (`w*h*8` bytes each). CL imports of the
    /// persistent buffers are cached by GL name and freed with the image cache.
    pub(super) fn prepare_gl_pair_buffers(&mut self,buffers:&[u32],outputs:&[(u32,u64)])->Result<()> {
        ensure!(self.supports_buffer_input()&&buffers.len()==2,"GL buffer interpolation input unsupported");
        let(w,h,_)=self.shape.ok_or_else(||anyhow!("GL interpolation not prepared"))?;let bytes=w*h*8;
        // Outputs first: its import-cache fallback clears every cached import.
        self.prepare_gl_pair(&[],outputs)?;
        let mut handles=Vec::new();
        for &name in buffers{
            ensure!(name!=0,"invalid interpolation GL buffer");
            if let Some(&(_,_,memory))=self.buffer_imports.iter().find(|(n,b,_)|*n==name&&*b==bytes){handles.push(memory);continue;}
            self.buffer_imports.retain(|(n,_,memory)|if *n==name{unsafe{let _=(self.base.cl.buffer_free)(*memory);}false}else{true});
            let mut status=0;let memory=unsafe{self.extra.import_gl_buffer.unwrap()(self.base.context,4,name,&mut status)};
            cl_check(status,"import interpolation GL buffer")?;ensure!(!memory.is_null(),"null interpolation GL buffer");
            self.buffer_imports.push((name,bytes,memory));handles.push(memory);
        }
        self.pair_buffers=handles;Ok(())
    }
    /// v985b: call on the GL owner thread once the GL copy into the pair's
    /// pixel buffers has completed (its fence signalled). Acquires the two GL
    /// buffers, copies them into CL-owned memory, releases them and flushes,
    /// all without waiting. The inference worker then never acquires a GL
    /// buffer (v985 hung there: Intel's implicit GL sync from a non-owner
    /// thread could wait on the render thread indefinitely).
    pub(super) fn stage_pair_buffers(&mut self)->Result<bool> {
        if self.pair_buffers.len()!=2||self.staged{return Ok(self.staged);}
        let(copy,flush)=match(self.extra.copy_buffer,self.extra.flush){(Some(c),Some(f))=>(c,f),_=>return Ok(false)};
        if self.gl_interp_staged.is_null(){return Ok(false);}
        let(w,h,_)=self.shape.ok_or_else(||anyhow!("GL interpolation not prepared"))?;let bytes=w*h*8;
        if self.endpoint.is_null()||self.endpoint_bytes!=2*bytes{
            if !self.endpoint.is_null(){unsafe{let _=(self.base.cl.buffer_free)(self.endpoint);}self.endpoint=ptr::null_mut();}
            self.endpoint=self.buffer(2*bytes)?;self.endpoint_bytes=2*bytes;
        }
        let(acquire,release)=(self.extra.acquire_gl.unwrap(),self.extra.release_gl.unwrap());
        let objects=self.pair_buffers.clone();
        cl_check(unsafe{acquire(self.base.queue,2,objects.as_ptr(),0,ptr::null(),ptr::null_mut())},"acquire staged GL buffers")?;
        let copied=(||->Result<()>{
            for(index,source)in objects.iter().enumerate(){
                cl_check(unsafe{copy(self.base.queue,*source,self.endpoint,0,index*bytes,bytes,0,ptr::null(),ptr::null_mut())},"copy staged GL buffer")?;
            }
            Ok(())
        })();
        let released=cl_check(unsafe{release(self.base.queue,2,objects.as_ptr(),0,ptr::null(),ptr::null_mut())},"release staged GL buffers");
        if copied.is_err()||released.is_err(){unsafe{let _=(self.extra.finish)(self.base.queue);}copied?;released?;}
        cl_check(unsafe{flush(self.base.queue)},"flush staged GL buffers")?;
        self.staged=true;Ok(true)
    }
    pub(super) fn image_cache_stats(&self)->(bool,usize,u64,u64,usize){
        (self.image_cache_enabled,self.image_cache.len(),self.image_cache_hits,self.image_cache_misses,self.image_cache.iter().map(|i|i.bytes).sum())
    }
    fn shared_image(&mut self,name:u32,allocation:u64,flags:u64,bytes:usize,protected:&[Handle])->Result<(Handle,bool)>{
        let cache=self.image_cache_enabled&&allocation!=0&&bytes>0&&bytes<=64*1024*1024;
        if cache{
            if let Some(index)=self.image_cache.iter().position(|i|i.allocation==allocation&&i.name==name&&i.flags==flags){
                self.image_cache_hits+=1;
                let image=self.image_cache.remove(index).unwrap();let memory=image.memory;self.image_cache.push_back(image);return Ok((memory,false));
            }
        }
        self.image_cache_misses+=1;
        let mut status=0;let memory=unsafe{self.extra.import_gl.unwrap()(self.base.context,flags,0x0de1,0,name,&mut status)};
        cl_check(status,"import SISR GL image")?;ensure!(!memory.is_null(),"null SISR GL image");
        if cache{
            while self.image_cache.len()>=8||self.image_cache.iter().map(|i|i.bytes).sum::<usize>()+bytes>64*1024*1024{
                let Some(index)=self.image_cache.iter().position(|i|!protected.contains(&i.memory))else{return Ok((memory,true))};
                let old=self.image_cache.remove(index).unwrap();unsafe{let _=(self.base.cl.buffer_free)(old.memory);}
            }
            self.image_cache.push_back(SharedImage{allocation,name,flags,memory,bytes});return Ok((memory,false));
        }
        Ok((memory,true))
    }
    pub(super) fn run_gl(&mut self,source:u32,destination:u32,reset:bool,source_id:u64,destination_id:u64)->Result<[f64;3]>{
        let result=self.run_gl_once(source,destination,reset,source_id,destination_id);
        if result.is_err()&&self.image_cache_enabled{
            unsafe{let _=(self.extra.finish)(self.base.queue);}self.clear_image_cache();self.image_cache_enabled=false;self.valid=false;
            log::warn!("neoarc-gl-import-cache-fallback: reason={:#} action=fresh-gpu-import history=reset",result.as_ref().err().unwrap());
            return self.run_gl_once(source,destination,true,source_id,destination_id);
        }
        result
    }
    fn run_gl_once(&mut self,source:u32,destination:u32,reset:bool,source_id:u64,destination_id:u64)->Result<[f64;3]>{
        ensure!(self.gl_mode&&source!=destination,"invalid CL/GL IO objects");
        let(w,h,channels)=self.shape.ok_or_else(||anyhow!("GL IO not prepared"))?;
        let acquire=self.extra.acquire_gl.unwrap();let release=self.extra.release_gl.unwrap();
        let import_started=Instant::now();
        // The identity denotes storage, not image contents. Every frame still
        // acquires, packs and infers fresh pixels. Source may not be evicted
        // while reserving this call's destination image.
        let(a,free_a)=self.shared_image(source,source_id,4,w*h*8,&[])?;
        let(b,free_b)=match self.shared_image(destination,destination_id,2,self.output_size.0*self.output_size.1*4,&[a]){Ok(v)=>v,Err(e)=>{if free_a{unsafe{let _=(self.base.cl.buffer_free)(a);}}return Err(e)}};
        let objects=[a,b];let import_ms=import_started.elapsed().as_secs_f64()*1000.0;
        let mut acquired=false;
        let mut result=(||->Result<[f64;3]>{
            let started=Instant::now();cl_check(unsafe{acquire(self.base.queue,2,objects.as_ptr(),0,ptr::null(),ptr::null_mut())},"acquire GL images")?;acquired=true;
            let frames=channels/3;let next=if frames==1{0}else{self.current^1};let old=if frames==1{self.input[0]}else{self.input[self.current]};
            let kernel=if self.vector_pack{self.gl_image_vector}else{self.gl_pack};
            self.arg(kernel,0,&objects[0])?;self.arg(kernel,1,&old)?;self.arg(kernel,2,&self.input[next])?;
            self.arg(kernel,3,&(w as i32))?;self.arg(kernel,4,&(h as i32))?;self.arg(kernel,5,&(frames as i32))?;
            self.arg(kernel,6,&i32::from(self.valid&&!reset))?;self.dispatch(kernel,if self.vector_pack{w*h}else{w*h*channels})?;
            self.submit_pack()?;self.audit_input(next,reset)?;
            let pack_ms=import_ms+started.elapsed().as_secs_f64()*1000.0;
            let started=Instant::now();
            if let Some(native)=self.native.as_mut(){
                native.run(self.input[next],self.base.output_mem)?;
                // Statistics include GPU completion, never just native enqueue time.
                cl_check(unsafe{(self.extra.finish)(self.base.queue)},"complete native Compact inference")?;
            }else{
                check(unsafe{(self.base.api.set_input)(self.base.request,self.tensors[next])},"bind GL input")?;
                check(unsafe{(self.base.api.infer)(self.base.request)},"infer GL remote model")?;
            }
            let run_ms=started.elapsed().as_secs_f64()*1000.0;let started=Instant::now();
            let(ow,oh)=self.output_size;let kernel=if self.output_type==3{self.gl_compact16}else{self.gl_compact32};
            self.arg(kernel,0,&self.base.output_mem)?;self.arg(kernel,1,&objects[1])?;self.arg(kernel,2,&(ow as i32))?;self.arg(kernel,3,&(oh as i32))?;
            self.dispatch(kernel,ow*oh)?;
            cl_check(unsafe{release(self.base.queue,2,objects.as_ptr(),0,ptr::null(),ptr::null_mut())},"release GL images")?;acquired=false;
            cl_check(unsafe{(self.extra.finish)(self.base.queue)},"complete GL output")?;
            let out_ms=started.elapsed().as_secs_f64()*1000.0;self.current=next;self.valid=true;
            Ok([pack_ms,run_ms,out_ms])
        })();
        if acquired{unsafe{let _=release(self.base.queue,2,objects.as_ptr(),0,ptr::null(),ptr::null_mut());let _=(self.extra.finish)(self.base.queue);}}
        let release_started=Instant::now();
        for(memory,owned)in[(a,free_a),(b,free_b)]{if owned{unsafe{let _=(self.base.cl.buffer_free)(memory);}}}
        if let Ok(times)=result.as_mut(){times[2]+=release_started.elapsed().as_secs_f64()*1000.0;}
        if result.is_err(){self.clear_image_cache();}
        if result.is_err(){self.valid=false;}
        result
    }

    pub(super) fn disable_interp_pair_cache(&mut self){self.pair_cache=false;self.pair_packed=false;}
    pub(super) fn diagnostic_infer_profile(&mut self,library:&Library,profile:bool)->Result<serde_json::Value> {
        #[repr(C)]struct Row{status:i32,real_time:i64,cpu_time:i64,name:*const c_char,execution:*const c_char,kind:*const c_char}
        #[repr(C)]struct Rows{data:*mut Row,size:usize}
        let mut samples=Vec::new();
        for iteration in 0..32 {
            let started=Instant::now();check(unsafe{(self.base.api.infer)(self.base.request)},"diagnostic GPU-only infer")?;
            if iteration>=8{samples.push(started.elapsed().as_secs_f64()*1000.0);}
        }
        let mut operators=Vec::new();
        if profile {
            let get:unsafe extern "C" fn(Handle,*mut Rows)->i32=unsafe{*library.get(b"ov_infer_request_get_profiling_info\0")?};
            let free:unsafe extern "C" fn(*mut Rows)=unsafe{*library.get(b"ov_profiling_info_list_free\0")?};
            let mut rows=Rows{data:ptr::null_mut(),size:0};check(unsafe{get(self.base.request,&mut rows)},"read GPU operator profile")?;
            let result=(||->Result<()>{
                ensure!(rows.size<=100000&&(rows.size==0||!rows.data.is_null()),"invalid GPU profiling array");
                if rows.size>0 {for row in unsafe{std::slice::from_raw_parts(rows.data,rows.size)} {
                    let text=|p:*const c_char|if p.is_null(){String::new()}else{unsafe{CStr::from_ptr(p)}.to_string_lossy().into_owned()};
                    operators.push(serde_json::json!({"node":text(row.name),"execution":text(row.execution),"op":text(row.kind),"status":row.status,"real_us":row.real_time,"cpu_us":row.cpu_time}));
                }}Ok(())
            })();unsafe{free(&mut rows)};result?;
            operators.sort_by_key(|v|std::cmp::Reverse(v["real_us"].as_i64().unwrap_or(0)));
        }
        let mut sorted=samples.clone();sorted.sort_by(f64::total_cmp);
        Ok(serde_json::json!({"profiling_enabled":profile,"scope":"resident OCL input/output tensors; infer API only; no pack, upload, readback, GL or present inside samples","p50_ms":sorted[sorted.len()/2],"p95_ms":sorted[(sorted.len()*95/100).min(sorted.len()-1)],"samples_ms":samples,"operators":operators,"operator_times_overlap_possible":true}))
    }

    pub(super) fn begin_interp_pair(&mut self){
        self.pair_packed=false;self.encoder_valid=false;self.boost_valid=false;self.pair_cut=None;self.pair_buffers.clear();self.staged=false;
        // Borrowed registrations remain in the bounded LRU, never pixel data.
        for(_,object,owned)in self.pair_images.drain(..){if owned{unsafe{let _=(self.base.cl.buffer_free)(object);}}}
    }
    pub(super) fn prepare_gl_pair(&mut self,inputs:&[(u32,u64)],outputs:&[(u32,u64)])->Result<()> {
        let result=self.prepare_gl_pair_once(inputs,outputs);
        if result.is_err()&&self.image_cache_enabled{
            unsafe{let _=(self.extra.finish)(self.base.queue);}self.begin_interp_pair();self.clear_image_cache();self.image_cache_enabled=false;
            log::warn!("neoarc-interp-import-cache-fallback: reason={:#} action=fresh-gpu-import",result.as_ref().err().unwrap());
            return self.prepare_gl_pair_once(inputs,outputs);
        }
        result
    }
    fn prepare_gl_pair_once(&mut self,inputs:&[(u32,u64)],outputs:&[(u32,u64)])->Result<()> {
        self.begin_interp_pair();self.phase_count=outputs.len();
        let(w,h,_)=self.shape.ok_or_else(||anyhow!("GL interpolation not prepared"))?;
        let bytes_out=self.output_size.0*self.output_size.1*4;
        let result=(||->Result<()>{
            for((id,allocation),flags,bytes)in inputs.iter().map(|v|(*v,4,w*h*8)).chain(outputs.iter().map(|v|(*v,2,bytes_out))){
                ensure!(id!=0,"invalid interpolation GL name");
                if self.pair_images.iter().any(|(old,_,_)|*old==id){continue;}
                // All handles used by this immutable pair are pinned against
                // eviction until begin_interp_pair, including earlier outputs.
                let protected:Vec<_>=self.pair_images.iter().map(|(_,h,_)|*h).collect();
                let(object,owned)=self.shared_image(id,allocation,flags,bytes,&protected)?;
                self.pair_images.push((id,object,owned));
            }
            Ok(())
        })();
        if result.is_err(){self.begin_interp_pair();}result
    }
    pub(super) fn run_gl_interp(&mut self,sources:&[u32],destination:u32,phase:f32)->Result<[f64;3]> {
        let(w,h,channels)=self.shape.ok_or_else(||anyhow!("GL interpolation not prepared"))?;
        ensure!(self.gl_mode&&matches!((sources.len(),channels),(2,7)|(4,14)),"GL interpolation layout unsupported");
        ensure!(!sources.contains(&destination),"GL interpolation input/output alias");
        let import=self.extra.import_gl.unwrap();let acquire=self.extra.acquire_gl.unwrap();let release=self.extra.release_gl.unwrap();
        let reuse=self.pair_cache&&self.pair_packed;
        let mut objects=Vec::new();let mut ownership=Vec::new();let mut acquired=false;let prepared=!self.pair_images.is_empty();
        // v985: endpoints already copied into GL pixel buffers for this pair.
        // v985b: staged = endpoints already in CL-owned memory (no GL input objects).
        let staged=prepared&&self.staged&&sources.len()==2;
        let buffered=prepared&&!staged&&self.pair_buffers.len()==2&&sources.len()==2;
        let result=(||->Result<[f64;3]>{
            if buffered&&!reuse{objects.extend_from_slice(&self.pair_buffers);}
            for(id,flags)in sources.iter().copied().filter(|_|!reuse&&!buffered&&!staged).map(|id|(id,4)).chain(std::iter::once((destination,2))) {
                if prepared {
                    objects.push(self.pair_images.iter().find(|(old,_,_)|*old==id).ok_or_else(||anyhow!("interpolation GL image not prepared on owner thread"))?.1);
                }else{
                    let mut status=0;let object=unsafe{import(self.base.context,flags,0x0de1,0,id,&mut status)};
                    cl_check(status,"import interpolation GL texture")?;ensure!(!object.is_null(),"null interpolation GL image");objects.push(object);
                }
            }
            for object in &objects{if !ownership.contains(object){ownership.push(*object);}}
            let started=Instant::now();cl_check(unsafe{acquire(self.base.queue,ownership.len()as u32,ownership.as_ptr(),0,ptr::null(),ptr::null_mut())},"acquire interpolation textures")?;acquired=true;
            let acquire_ms=started.elapsed().as_secs_f64()*1000.0;
            if !reuse&&log::log_enabled!(log::Level::Trace){
                // v985 diagnostics: split the first-phase input cost (acquire call vs GPU drain).
                let drain=Instant::now();let _=unsafe{(self.extra.finish)(self.base.queue)};
                log::trace!("neoarc-gl-acquire-split: objects={} acquire_call_ms={:.3} acquire_drain_ms={:.3}",ownership.len(),acquire_ms,drain.elapsed().as_secs_f64()*1000.0);
            }
            if reuse {
                self.arg(self.gl_interp_time,0,&self.input[0])?;self.arg(self.gl_interp_time,1,&((w*h)as i32))?;
                self.arg(self.gl_interp_time,2,&((sources.len()*3)as i32))?;self.arg(self.gl_interp_time,3,&phase)?;
                self.dispatch(self.gl_interp_time,w*h)?;
            }else if staged{
                let kernel=self.gl_interp_staged;
                self.arg(kernel,0,&self.endpoint)?;self.arg(kernel,1,&self.input[0])?;
                self.arg(kernel,2,&(w as i32))?;self.arg(kernel,3,&(h as i32))?;self.arg(kernel,4,&phase)?;
                self.dispatch(kernel,w*h)?;
            }else if buffered{
                let kernel=self.gl_interp_buffer;
                self.arg(kernel,0,&objects[0])?;self.arg(kernel,1,&objects[1])?;self.arg(kernel,2,&self.input[0])?;
                self.arg(kernel,3,&(w as i32))?;self.arg(kernel,4,&(h as i32))?;self.arg(kernel,5,&phase)?;
                let enqueue=Instant::now();self.dispatch(kernel,w*h)?;
                if log::log_enabled!(log::Level::Trace){
                    let enqueue_ms=enqueue.elapsed().as_secs_f64()*1000.0;let drain=Instant::now();let _=unsafe{(self.extra.finish)(self.base.queue)};
                    log::trace!("neoarc-gl-pack-split: input=buffer enqueue_ms={:.3} kernel_drain_ms={:.3}",enqueue_ms,drain.elapsed().as_secs_f64()*1000.0);
                }
            }else{
                let kernel=if self.vector_pack{self.gl_interp_vector}else{self.gl_interp_pack};
                for index in 0..4 {self.arg(kernel,index as u32,&objects[index.min(sources.len()-1)])?;}
                self.arg(kernel,4,&self.input[0])?;self.arg(kernel,5,&(w as i32))?;
                self.arg(kernel,6,&(h as i32))?;self.arg(kernel,7,&(channels as i32))?;
                self.arg(kernel,8,&(sources.len()as i32))?;self.arg(kernel,9,&phase)?;
                let enqueue=Instant::now();
                self.dispatch(kernel,if self.vector_pack{w*h}else{w*h*channels})?;
                if log::log_enabled!(log::Level::Trace){
                    let enqueue_ms=enqueue.elapsed().as_secs_f64()*1000.0;let drain=Instant::now();let _=unsafe{(self.extra.finish)(self.base.queue)};
                    log::trace!("neoarc-gl-pack-split: enqueue_ms={:.3} kernel_drain_ms={:.3}",enqueue_ms,drain.elapsed().as_secs_f64()*1000.0);
                }
            }
            self.submit_pack()?;
            let pack=started.elapsed().as_secs_f64()*1000.0;
            let started=Instant::now();
            // v963: boosted pair -> trunk once (state), then this phase's head.
            let mut boosted=false;
            let boost_on=self.boost.is_some()&&self.boost_force.unwrap_or(self.boost_enabled);
            // v983: the shared t=0.5 motion makes a scene cut switch one phase
            // early and leaves a ghost of the old shot. Judge the pair once on
            // the packed endpoints (never during forced validation runs) and
            // present a hard cut for such pairs instead of interpolating.
            // Only prepared pairs reset pair_cut (begin_interp_pair), so an
            // unprepared ad-hoc call never inherits the previous pair's verdict.
            if !prepared{self.pair_cut=None;}
            if boost_on&&prepared&&self.boost_force.is_none()&&self.scene_gate&&self.pair_cut.is_none()&&sources.len()==2 {
                let judge_started=Instant::now();
                match self.judge_scene_cut(w,h){
                    Ok((cut,mad,md,corr))=>{
                        let judge_ms=judge_started.elapsed().as_secs_f64()*1000.0;
                        let stats=&mut self.scene_stats;stats.pairs+=1;stats.cuts+=u64::from(cut);stats.judge_ms+=judge_ms;
                        if cut||stats.pairs<=2||stats.pairs%1800==0{
                            log::info!("neoarc-rife-boost-scene: pairs={} cuts={} cut={} mad={:.4} md={:.4} corr={:.4} judge_ms={:.3} avg_judge_ms={:.3} action={}",
                                stats.pairs,stats.cuts,cut,mad,md,corr,judge_ms,stats.judge_ms/stats.pairs as f64,if cut{"copy-endpoint"}else{"boost"});
                        }
                        self.pair_cut=Some(cut);
                    }
                    Err(error)=>{self.scene_gate=false;self.pair_cut=Some(false);
                        log::warn!("neoarc-rife-boost-scene-disabled: reason={error:#} action=boost-without-scene-gate");}
                }
            }
            if boost_on&&self.pair_cut==Some(true){
                let(kernel,frame)=(if self.output_type==3{self.cut_copy16}else{self.cut_copy32},i32::from(phase>=0.5));
                self.arg(kernel,0,&self.input[0])?;self.arg(kernel,1,&self.base.output_mem)?;
                self.arg(kernel,2,&((w*h)as i32))?;self.arg(kernel,3,&frame)?;self.dispatch(kernel,3*w*h)?;
                boosted=true;
            }else if boost_on{
                let(trunk,head)={let b=self.boost.as_ref().unwrap();(b.trunk.request,b.head.request)};
                let(api,set_index,input,trunk_done)=(self.base.api,self.extra.set_input_index,self.tensors[0],self.boost_valid);
                // v983: Lite heads run as one OpenCL kernel on the same in-order
                // queue (the trunk infer has completed when it returns).
                let native=if self.native_head{self.native_head_kernel()}else{None};
                let result=(||->Result<()>{
                    if !trunk_done{
                        check(unsafe{set_index(trunk,0,input)},"bind boost trunk input")?;
                        check(unsafe{(api.infer)(trunk)},"infer boost trunk")?;
                    }
                    if let Some((kernel,state))=native{
                        let t=half::f16::from_f32(phase.clamp(0.0,1.0)).to_f32();
                        self.arg(kernel,0,&self.input[0])?;self.arg(kernel,1,&state)?;self.arg(kernel,2,&self.base.output_mem)?;
                        self.arg(kernel,3,&(w as i32))?;self.arg(kernel,4,&(h as i32))?;self.arg(kernel,5,&t)?;
                        return self.dispatch(kernel,w*h);
                    }
                    check(unsafe{set_index(head,0,input)},"bind boost head input")?;
                    check(unsafe{(api.infer)(head)},"infer boost head")
                })();
                match result{
                    Ok(())=>{self.boost_valid=true;boosted=true;}
                    Err(error)=>{unsafe{let _=(self.extra.finish)(self.base.queue);}self.boost.take();self.boost_valid=false;
                        log::warn!("neoarc-rife-boost-fallback: reason={error:#} stage=run action=original-gpu-graph");}
                }
            }
            if !boosted{
            let feature_request=self.encoder.as_ref().filter(|_|self.phase_count>1).map(|e|e.graphs[usize::from(self.encoder_valid)].request);
            let request=feature_request.unwrap_or(self.base.request);
            check(unsafe{(self.extra.set_input_index)(request,0,self.tensors[0])},"bind GL interpolation input")?;
            let status=unsafe{(self.base.api.infer)(request)};
            if status!=0&&feature_request.is_some(){
                unsafe{let _=(self.extra.finish)(self.base.queue);}self.encoder.take();self.encoder_valid=false;
                log::warn!("neoarc-encoder-cache-fallback: action=original-gpu-graph status={status}");
                check(unsafe{(self.base.api.set_input)(self.base.request,self.tensors[0])},"bind original interpolation input")?;
                check(unsafe{(self.base.api.infer)(self.base.request)},"infer original GL interpolation")?;
            }else{check(status,"infer GL interpolation")?;if feature_request.is_some(){self.encoder_valid=true;}}
            }
            let run=started.elapsed().as_secs_f64()*1000.0;let started=Instant::now();
            let(ow,oh)=self.output_size;let kernel=if self.output_type==3{self.gl_compact16}else{self.gl_compact32};
            self.arg(kernel,0,&self.base.output_mem)?;self.arg(kernel,1,objects.last().unwrap())?;
            self.arg(kernel,2,&(ow as i32))?;self.arg(kernel,3,&(oh as i32))?;self.dispatch(kernel,ow*oh)?;
            cl_check(unsafe{release(self.base.queue,ownership.len()as u32,ownership.as_ptr(),0,ptr::null(),ptr::null_mut())},"release interpolation textures")?;acquired=false;
            cl_check(unsafe{(self.extra.finish)(self.base.queue)},"complete shared interpolation output")?;
            Ok([pack,run,started.elapsed().as_secs_f64()*1000.0])
        })();
        if acquired {unsafe{let _=release(self.base.queue,ownership.len()as u32,ownership.as_ptr(),0,ptr::null(),ptr::null_mut());}}
        // Drain even after a failed release before any GL image can be reused.
        if result.is_err(){unsafe{let _=(self.extra.finish)(self.base.queue);}}
        self.pair_packed=result.is_ok();
        if !prepared{for object in objects{unsafe{let _=(self.base.cl.buffer_free)(object);}}}
        result
    }

    /// v983: thumbnail the two packed endpoints on the GPU (27 KB readback) and
    /// apply the NeoAMD scene-cut test. Only full-geometry RIFE outputs can take
    /// the copy lane, so any other layout is reported as "not a cut".
    fn judge_scene_cut(&mut self,w:usize,h:usize)->Result<(bool,f32,f32,f32)> {
        ensure!(!self.scene_thumb.is_null()&&!self.cut_copy16.is_null()&&!self.cut_copy32.is_null(),"scene gate kernels missing");
        if self.output_size!=(w,h){return Ok((false,0.0,0.0,f32::NAN));}
        let count=6*SCENE_THUMB_PLANE;
        if self.thumb_mem.is_null(){self.thumb_mem=self.buffer(count*2)?;}
        self.arg(self.scene_thumb,0,&self.input[0])?;self.arg(self.scene_thumb,1,&self.thumb_mem)?;
        self.arg(self.scene_thumb,2,&(w as i32))?;self.arg(self.scene_thumb,3,&(h as i32))?;
        self.dispatch(self.scene_thumb,count)?;
        let mut values=vec![half::f16::ZERO;count];
        cl_check(unsafe{(self.base.cl.read)(self.base.queue,self.thumb_mem,1,0,count*2,values.as_mut_ptr().cast(),0,ptr::null(),ptr::null_mut())},"read scene thumbnail")?;
        let thumb:Vec<f32>=values.iter().map(|v|v.to_f32()).collect();
        let(mad,md,corr)=scene_cut_metrics(&thumb);
        Ok((scene_cut_decision(mad,md,corr),mad,md,corr))
    }

    pub(super) fn read_rife_output(&self)->Result<Vec<half::f16>> {
        ensure!(self.output_type==3,"RIFE remote output must be FP16");
        let count=self.output_size.0*self.output_size.1*3;let mut values=vec![half::f16::ZERO;count];
        cl_check(unsafe{(self.base.cl.read)(self.base.queue,self.base.output_mem,1,0,count*2,values.as_mut_ptr().cast(),0,ptr::null(),ptr::null_mut())},"read RIFE validation output")?;Ok(values)
    }
    pub(super) fn run_rife(&mut self,w:usize,h:usize,frames:&[&[u8]],stride:usize,phase:f32)->Result<(Vec<u8>,[f64;3])> {
        let result=self.run_rife_once(w,h,frames,stride,phase);
        // Nonblocking DMA borrows frame storage until inference completes. On any
        // early error, drain the queue before those caller-owned bytes can die.
        if result.is_err(){unsafe{let _=(self.extra.finish)(self.base.queue);}}
        result
    }
    fn run_rife_once(&mut self,w:usize,h:usize,frames:&[&[u8]],stride:usize,phase:f32)->Result<(Vec<u8>,[f64;3])> {
        ensure!(self.shape==Some((w,h,7))&&frames.len()==2&&matches!(stride,3|4),"RIFE RGB shape/stride mismatch");
        let n=w*h;ensure!(frames.iter().all(|v|v.len()>=n*stride),"RIFE RGB input too short");
        let started=Instant::now();
        for(index,frame)in frames.iter().enumerate(){cl_check(unsafe{(self.base.cl.write)(self.base.queue,self.source,0,index*n*stride,n*stride,frame.as_ptr().cast(),0,ptr::null(),ptr::null_mut())},"upload RIFE RGB endpoint")?;}
        self.arg(self.rife_pack,0,&self.source)?;self.arg(self.rife_pack,1,&self.input[0])?;self.arg(self.rife_pack,2,&(n as i32))?;self.arg(self.rife_pack,3,&(stride as i32))?;self.arg(self.rife_pack,4,&phase)?;
        self.dispatch(self.rife_pack,7*n)?;self.submit_pack()?;let pack_ms=started.elapsed().as_secs_f64()*1000.0;
        check(unsafe{(self.base.api.set_input)(self.base.request,self.tensors[0])},"bind RIFE RGB tensor")?;
        let started=Instant::now();check(unsafe{(self.base.api.infer)(self.base.request)},"infer RIFE RGB remote model")?;let run_ms=started.elapsed().as_secs_f64()*1000.0;
        let started=Instant::now();let(ow,oh)=self.output_size;ensure!((ow,oh)==(w,h),"RIFE RGB output geometry mismatch");
        let kernel=if self.output_type==3{self.compact16}else{self.compact32};self.arg(kernel,0,&self.base.output_mem)?;self.arg(kernel,1,&self.packed_rgb)?;self.arg(kernel,2,&(n as i32))?;self.arg(kernel,3,&3i32)?;self.dispatch(kernel,n)?;
        let len=n*3;let mut rgb:Vec<u8>=Vec::with_capacity(len);
        cl_check(unsafe{(self.base.cl.read)(self.base.queue,self.packed_rgb,1,0,len,rgb.as_mut_ptr().cast(),0,ptr::null(),ptr::null_mut())},"read RIFE RGB output")?;
        unsafe{rgb.set_len(len);}
        Ok((rgb,[pack_ms,run_ms,started.elapsed().as_secs_f64()*1000.0]))
    }

    // Developer-only full input fingerprint; disabled without allocating or reading GPU memory.
    fn audit_input(&self,next:usize,reset:bool)->Result<()> {
        if !self.input_audit{return Ok(());}
        use sha2::{Digest,Sha256};
        let(w,h,c)=self.shape.ok_or_else(||anyhow!("audit shape missing"))?;
        let mut bytes=vec![0u8;w*h*c*2];
        cl_check(unsafe{(self.base.cl.read)(self.base.queue,self.input[next],1,0,bytes.len(),bytes.as_mut_ptr().cast(),0,ptr::null(),ptr::null_mut())},"audit packed input")?;
        log::info!("neoarc-input-audit: gl={} shape={}x{}x{} reset={} history_valid={} sha256={:x}",self.gl_mode,w,h,c,reset,self.valid,Sha256::digest(&bytes));Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(&mut self,core:Handle,path:&Path,channels:usize,w:usize,h:usize,
        pixels:&[u8],stride:usize,out_stride:usize,reset:bool,f32:bool)->Result<(usize,usize,Vec<u8>,[f64;3])> {
        ensure!(matches!(stride,3|4)&&matches!(out_stride,3|4),"invalid RGB stride");
        self.prepare(core,path,channels,w,h,f32)?;
        let n=w*h;ensure!(pixels.len()>=n*stride,"short RGB input");
        let started=Instant::now();let frames=channels/3;
        let next=if frames==1{0}else{self.current^1};
        let old=if frames==1{self.input[0]}else{self.input[self.current]};
        cl_check(unsafe{(self.base.cl.write)(self.base.queue,self.source,1,0,n*stride,pixels.as_ptr().cast(),0,ptr::null(),ptr::null_mut())},"upload RGB pixels")?;
        self.arg(self.pack,0,&self.source)?;self.arg(self.pack,1,&old)?;self.arg(self.pack,2,&self.input[next])?;
        self.arg(self.pack,3,&(n as i32))?;self.arg(self.pack,4,&(stride as i32))?;self.arg(self.pack,5,&(frames as i32))?;
        self.arg(self.pack,6,&i32::from(self.valid&&!reset))?;self.dispatch(self.pack,n*channels)?;
        self.submit_pack()?;self.audit_input(next,reset)?;
        let pack_ms=started.elapsed().as_secs_f64()*1000.0;
        check(unsafe{(self.base.api.set_input)(self.base.request,self.tensors[next])},"bind RGB input")?;
        let started=Instant::now();check(unsafe{(self.base.api.infer)(self.base.request)},"infer RGB remote model")?;
        let run_ms=started.elapsed().as_secs_f64()*1000.0;let started=Instant::now();
        let(ow,oh)=self.output_size;let on=ow*oh;let kernel=if self.output_type==3{self.compact16}else{self.compact32};
        self.arg(kernel,0,&self.base.output_mem)?;self.arg(kernel,1,&self.packed_rgb)?;self.arg(kernel,2,&(on as i32))?;
        self.arg(kernel,3,&(out_stride as i32))?;self.dispatch(kernel,on)?;
        let mut rgb=vec![0u8;on*out_stride];
        cl_check(unsafe{(self.base.cl.read)(self.base.queue,self.packed_rgb,1,0,rgb.len(),rgb.as_mut_ptr().cast(),0,ptr::null(),ptr::null_mut())},"read compact RGB pixels")?;
        let out_ms=started.elapsed().as_secs_f64()*1000.0;self.current=next;self.valid=true;
        Ok((ow,oh,rgb,[pack_ms,run_ms,out_ms]))
    }
}
impl Drop for RgbIo {
    fn drop(&mut self) {
        self.retire();unsafe{
            for k in [self.pack,self.compact16,self.compact32,self.rife_pack,self.gl_pack,self.gl_interp_pack,self.gl_interp_vector,self.gl_image_vector,self.gl_interp_time,self.gl_compact16,self.gl_compact32,self.scene_thumb,self.cut_copy16,self.cut_copy32,self.boost_heads[0],self.boost_heads[1],self.boost_heads[2],self.boost_heads[3],self.gl_interp_buffer,self.gl_interp_staged]{if !k.is_null(){let _=(self.extra.free_kernel)(k);}}
            if !self.program.is_null(){let _=(self.extra.free_program)(self.program);}
        }
    }
}

#[cfg(test)]
mod scene_gate_tests {
    use super::*;
    fn thumb(f0:impl Fn(usize,usize,usize)->f32,f1:impl Fn(usize,usize,usize)->f32)->Vec<f32>{
        let mut v=Vec::with_capacity(6*SCENE_THUMB_PLANE);
        for f in 0..2{for c in 0..3{for y in 0..SCENE_THUMB_H{for x in 0..SCENE_THUMB_W{
            let value=if f==0{f0(c,y,x)}else{f1(c,y,x)};v.push(half::f16::from_f32(value).to_f32());
        }}}}
        v
    }
    #[test]
    fn neoarc_scene_same_or_moving_picture_is_not_a_cut(){
        let a=|c:usize,y:usize,x:usize|((x*7+y*3+c*11)%64)as f32/64.0;
        let(mad,md,corr)=scene_cut_metrics(&thumb(a,a));
        assert_eq!(mad,0.0);assert_eq!(md,0.0);assert!(corr>0.999);assert!(!scene_cut_decision(mad,md,corr));
        // A one-pixel pan of a smooth gradient stays strongly correlated.
        let b=|c:usize,y:usize,x:usize|((x+y) as f32/100.0+c as f32*0.1).min(1.0);
        let shifted=move |c:usize,y:usize,x:usize|b(c,y,x+1);
        let(mad,md,corr)=scene_cut_metrics(&thumb(b,shifted));
        assert!(!scene_cut_decision(mad,md,corr),"pan judged as cut: mad={mad} md={md} corr={corr}");
    }
    #[test]
    fn neoarc_scene_different_shot_is_a_cut(){
        // Dark left-to-right gradient vs bright inverted gradient (new shot).
        let a=|_c:usize,_y:usize,x:usize|0.1+0.3*x as f32/63.0;
        let b=|_c:usize,y:usize,x:usize|0.9-0.3*x as f32/63.0-0.1*y as f32/35.0;
        let(mad,md,corr)=scene_cut_metrics(&thumb(a,b));
        assert!(mad>SCENE_MAD&&md>SCENE_MEAN_DIFF&&corr<SCENE_CORR,"mad={mad} md={md} corr={corr}");
        assert!(scene_cut_decision(mad,md,corr));
    }
    #[test]
    fn neoarc_scene_flat_frames_never_cut(){
        // Fade from black to white: large difference but zero variance -> NaN corr.
        let(mad,md,corr)=scene_cut_metrics(&thumb(|_,_,_|0.0,|_,_,_|1.0));
        assert!(mad>0.9&&md>0.9&&corr.is_nan());assert!(!scene_cut_decision(mad,md,corr));
    }
}
