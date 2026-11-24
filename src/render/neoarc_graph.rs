//! Preserve the original ONNX result and weights, but materialize FP16 Conv
//! results before downstream fusion. Only scalar side outputs reach the host.
use anyhow::{Result, anyhow, ensure};
use sha2::{Digest, Sha256};
use std::{collections::{HashSet, HashMap}, path::{Path, PathBuf}};
#[path="neoarc_graph/dw_branch.rs"]
mod dw_branch;
pub(super) use dw_branch::prepare_depthwise_branches;

struct Field<'a> { number: u64, value: u64, bytes: &'a [u8], raw: &'a [u8] }
fn varint(data: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let byte = *data.get(*pos).ok_or_else(|| anyhow!("truncated ONNX varint"))?;
        *pos += 1;
        ensure!(shift != 63 || byte <= 1, "ONNX varint overflow");
        value |= u64::from(byte & 127) << shift;
        if byte & 128 == 0 { return Ok(value); }
    }
    Err(anyhow!("invalid ONNX varint"))
}
fn fields(data: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut result = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let start = pos;
        let key = varint(data, &mut pos)?;
        ensure!(key >> 3 != 0, "invalid ONNX field");
        let mut value = 0;
        let bytes = match key & 7 {
            0 => { value = varint(data, &mut pos)?; &[][..] }
            wire @ (1 | 2 | 5) => {
                let count = match wire { 1 => 8, 5 => 4, _ => usize::try_from(varint(data, &mut pos)?)? };
                let end = pos.checked_add(count).ok_or_else(|| anyhow!("ONNX length overflow"))?;
                let bytes = data.get(pos..end).ok_or_else(|| anyhow!("truncated ONNX field"))?;
                pos = end; bytes
            }
            _ => return Err(anyhow!("unsupported ONNX wire type")),
        };
        result.push(Field { number: key >> 3, value, bytes, raw: &data[start..pos] });
    }
    Ok(result)
}
fn put_varint(output: &mut Vec<u8>, mut value: u64) {
    while value >= 128 { output.push(value as u8 | 128); value >>= 7; }
    output.push(value as u8);
}
fn integer(output: &mut Vec<u8>, field: u64, value: u64) {
    put_varint(output, field << 3); put_varint(output, value);
}
fn message(output: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    put_varint(output, field << 3 | 2); put_varint(output, bytes.len() as u64); output.extend_from_slice(bytes);
}

// Conservative rollout: topology discovery is generic, but only the exact
// stabilized graphs covered by the GPU channel-provenance/phase tests qualify.
pub(super) struct EncoderPlan {
    source: Vec<u8>,
    removed: HashSet<usize>,
    values: HashSet<Vec<u8>>,
    pub features: Vec<Vec<u8>>,
    pub output_count: usize,
    directory: PathBuf,
}
fn tensor_info(name: &[u8], dims: Option<&[i64]>) -> Vec<u8> {
    let mut shape=Vec::new();
    if let Some(dims)=dims { for size in dims {let mut d=Vec::new();integer(&mut d,1,*size as u64);message(&mut shape,1,&d);} }
    else {for _ in 0..4 {message(&mut shape,1,&[]);}}
    let mut tensor=Vec::new();integer(&mut tensor,1,10);message(&mut tensor,2,&shape);
    let mut ty=Vec::new();message(&mut ty,1,&tensor);
    let mut info=Vec::new();message(&mut info,1,name);message(&mut info,2,&ty);info
}
pub(super) fn write_derived(path:&Path,bytes:&[u8])->Result<()> {
    // Always verify the small derived file's contents; an interrupted or edited
    // cache is never trusted as an executable graph merely because it exists.
    if std::fs::read(path).ok().as_deref()==Some(bytes){return Ok(());}
    std::fs::create_dir_all(path.parent().ok_or_else(||anyhow!("cache parent missing"))?)?;
    static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
    let tmp=path.with_extension(format!("{}-{}.tmp",std::process::id(),NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)));
    std::fs::write(&tmp,bytes)?;
    if let Err(error)=std::fs::rename(&tmp,path){let _=std::fs::remove_file(&tmp);if std::fs::read(path).ok().as_deref()!=Some(bytes){return Err(error.into());}}
    Ok(())
}
impl EncoderPlan {
    pub(super) fn discover(cache:&Path,path:&Path,w:usize,h:usize)->Result<Option<Self>> {
        let source=std::fs::read(path)?;let digest=format!("{:x}",Sha256::digest(&source));
        if !matches!(digest.as_str(),
            "47174cb16fd0c5fde66502941fde4af7b7b931ee86b6c07d895b2628c8126419"|
            "993ffe1695fdd2dc43adadf380832cc8ae67c12f4d93cc108f8d877e22937e2f"|
            "67f349ef5ceb423b093caedad2a006cff865dd6a75576db3a653fe2fa76ef195"|
            "15fa728a952938a510f8f2d34276b90fc484b3d7256850c5c7c399bc46c11860"){return Ok(None);}
        let model=fields(&source)?;let graph=fields(model.iter().find(|f|f.number==7).ok_or_else(||anyhow!("graph missing"))?.bytes)?;
        let nodes:Vec<_>=graph.iter().filter(|f|f.number==1).map(|f|fields(f.bytes)).collect::<Result<_>>()?;
        let mut weights=HashMap::new();
        for tensor in graph.iter().filter(|f|f.number==5){let t=fields(tensor.bytes)?;let mut dims=Vec::new();for d in t.iter().filter(|f|f.number==1){if d.bytes.is_empty(){dims.push(d.value);}else{let mut pos=0;while pos<d.bytes.len(){dims.push(varint(d.bytes,&mut pos)?);}}}if let Some(name)=t.iter().find(|f|f.number==8){weights.insert(name.bytes,dims);}}
        let mut producer=HashMap::new();for(i,node)in nodes.iter().enumerate(){for out in node.iter().filter(|f|f.number==2){producer.insert(out.bytes,i);}}
        let mut removed=HashSet::new();let mut features=Vec::new();
        for(i,node)in nodes.iter().enumerate(){
            let inputs:Vec<_>=node.iter().filter(|f|f.number==1).collect();
            if !node.iter().any(|f|f.number==4&&f.bytes==b"ConvTranspose")||inputs.len()<2||weights.get(inputs[1].bytes).is_none_or(|v|v.as_slice()!=[16,4,4,4]){continue;}
            let mut index=i;let mut chain=vec![i];
            for expected in [b"LeakyRelu".as_slice(),b"Conv",b"LeakyRelu",b"Conv",b"LeakyRelu",b"Conv"]{
                let value=nodes[index].iter().find(|f|f.number==1).ok_or_else(||anyhow!("encoder input missing"))?.bytes;
                index=*producer.get(value).ok_or_else(||anyhow!("encoder producer missing"))?;
                ensure!(nodes[index].iter().any(|f|f.number==4&&f.bytes==expected),"encoder topology changed");chain.push(index);
            }
            let first_inputs:Vec<_>=nodes[index].iter().filter(|f|f.number==1).collect();
            ensure!(first_inputs.len()>=2&&weights.get(first_inputs[1].bytes).is_some_and(|d|d.len()==4&&d[1]==3),"encoder is not RGB");
            removed.extend(chain);features.push(node.iter().find(|f|f.number==2).ok_or_else(||anyhow!("encoder output missing"))?.bytes.to_vec());
        }
        ensure!((1..=2).contains(&features.len()),"encoder branch count unsupported");
        let values=nodes.iter().enumerate().filter(|(i,_)|removed.contains(i)).flat_map(|(_,n)|n.iter().filter(|f|f.number==2).map(|f|f.bytes.to_vec())).collect();
        Ok(Some(Self{output_count:graph.iter().filter(|f|f.number==12).count(),source,removed,values,features,directory:cache.join("encoder-pair-v1").join(digest).join(format!("{w}x{h}"))}))
    }
    pub(super) fn full(&self)->Result<PathBuf>{
        let model=fields(&self.source)?;let mut bytes=Vec::new();
        for f in model{if f.number==7{let mut graph=f.bytes.to_vec();for name in &self.features{message(&mut graph,12,&tensor_info(name,None));}message(&mut bytes,7,&graph);}else{bytes.extend_from_slice(f.raw);}}
        let path=self.directory.join("first.onnx");write_derived(&path,&bytes)?;Ok(path)
    }
    pub(super) fn tail(&self,w:usize,h:usize,shapes:&[Vec<i64>])->Result<PathBuf>{
        ensure!(shapes.len()==self.features.len(),"feature shape count differs");let model=fields(&self.source)?;let mut bytes=Vec::new();
        for f in model{if f.number!=7{bytes.extend_from_slice(f.raw);continue;}let graph=fields(f.bytes)?;let mut updated=Vec::new();let mut index=0;
            for g in graph{match g.number{
                1=>{let current=index;index+=1;if self.removed.contains(&current){continue;}let n=fields(g.bytes)?;
                    if n.iter().any(|f|f.number==4&&f.bytes==b"ReduceMean")&&n.iter().find(|f|f.number==1).is_some_and(|f|self.values.contains(f.bytes)){
                        let output=n.iter().find(|f|f.number==2).ok_or_else(||anyhow!("side output missing"))?.bytes;
                        let mut zero=Vec::new();for _ in 0..4{integer(&mut zero,1,1);}integer(&mut zero,2,10);message(&mut zero,9,&[0,0]);
                        let mut attr=Vec::new();message(&mut attr,1,b"value");message(&mut attr,5,&zero);integer(&mut attr,20,4);
                        let mut node=Vec::new();message(&mut node,2,output);message(&mut node,4,b"Constant");message(&mut node,5,&attr);message(&mut updated,1,&node);
                    }else{updated.extend_from_slice(g.raw);}
                }
                11=>{let info=fields(g.bytes)?;let name=info.iter().find(|f|f.number==1).ok_or_else(||anyhow!("input name missing"))?.bytes;message(&mut updated,11,&tensor_info(name,Some(&[1,7,h as i64,w as i64])));}
                _=>updated.extend_from_slice(g.raw),
            }}
            for(name,shape)in self.features.iter().zip(shapes){message(&mut updated,11,&tensor_info(name,Some(shape)));}message(&mut bytes,7,&updated);
        }
        let path=self.directory.join("later.onnx");write_derived(&path,&bytes)?;Ok(path)
    }
}

pub(super) fn cache_path(cache: &Path, original: &Path) -> Result<PathBuf> {
    // v993: memoize the content hash per file identity (path, length, mtime);
    // every stage rebuild hashed the same ~10 MB models several times.
    type Key = (PathBuf, u64, std::time::SystemTime);
    static MEMO: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<Key, String>>> = std::sync::OnceLock::new();
    let key = std::fs::metadata(original).ok().and_then(|m| Some((original.to_path_buf(), m.len(), m.modified().ok()?)));
    if let Some(key) = key.as_ref() {
        if let Some(hash) = MEMO.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner()).get(key) {
            return Ok(cache.join(format!("{hash}.onnx")));
        }
    }
    let hash = format!("{:x}", Sha256::digest(std::fs::read(original)?));
    if let Some(key) = key {
        MEMO.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner()).insert(key, hash.clone());
    }
    Ok(cache.join(format!("{hash}.onnx")))
}

/// This runtime's GPU compiled-model import changed numerical results in
/// reduction graphs, in both cache modes. Keep their original FP16 execution
/// but compile from the source model; do not identify affected models by name.
pub(super) fn requires_uncached_gpu_model(original:&Path)->Result<bool>{
    fn graph_has_reduction(graph:&[u8],depth:usize)->Result<bool>{
        ensure!(depth<64,"nested model graph exceeds cache inspection limit");
        for field in fields(graph)?.into_iter().filter(|f|f.number==1){
            let node=fields(field.bytes)?;
            if node.iter().any(|f|f.number==4&&f.bytes.starts_with(b"Reduce")){return Ok(true);}
            for attribute in node.iter().filter(|f|f.number==5){
                for value in fields(attribute.bytes)?.into_iter().filter(|f|matches!(f.number,6|11)){
                    if graph_has_reduction(value.bytes,depth+1)?{return Ok(true);}
                }
            }
        }
        Ok(false)
    }
    let source=std::fs::read(original)?;let model=fields(&source)?;
    if model.iter().any(|f|f.number==25){return Ok(true);}
    let graph=model.iter().find(|f|f.number==7).ok_or_else(||anyhow!("model graph missing for cache inspection"))?;
    graph_has_reduction(graph.bytes,0)
}

fn patched(original: &[u8]) -> Result<(Vec<u8>, usize)> {
    let model = fields(original)?;
    let graph = model.iter().find(|f| f.number == 7).ok_or_else(|| anyhow!("ONNX graph missing"))?;
    let mut opset = None;
    for entry in model.iter().filter(|f| f.number == 8) {
        let entry = fields(entry.bytes)?;
        if entry.iter().find(|f| f.number == 1).is_none_or(|f| f.bytes.is_empty()) {
            opset = entry.iter().find(|f| f.number == 2).map(|f| f.value);
        }
    }
    let opset = opset.ok_or_else(|| anyhow!("ONNX standard opset missing"))?;
    ensure!((11..=21).contains(&opset), "unsupported stabilization opset {opset}");
    let graph_fields = fields(graph.bytes)?;
    ensure!(!graph.bytes.windows(b"__neoarc_stable_fp16_".len()).any(|s| s == b"__neoarc_stable_fp16_"), "graph already stabilized");
    let mut fp16_weights = HashSet::new();
    for tensor in graph_fields.iter().filter(|f| f.number == 5) {
        let tensor = fields(tensor.bytes)?;
        ensure!(!tensor.iter().any(|f| f.number == 13 || (f.number == 14 && f.value != 0)), "external ONNX weights require original location");
        let mut rank = 0;
        for dimension in tensor.iter().filter(|f| f.number == 1) {
            if dimension.bytes.is_empty() { rank += 1; }
            else { let mut pos = 0; while pos < dimension.bytes.len() { varint(dimension.bytes, &mut pos)?; rank += 1; } }
        }
        if rank == 4 && tensor.iter().any(|f| f.number == 2 && f.value == 10) {
            if let Some(name) = tensor.iter().find(|f| f.number == 8) { fp16_weights.insert(name.bytes); }
        }
    }
    let mut activations = HashMap::new();
    for node in graph_fields.iter().filter(|f| f.number == 1) {
        let node = fields(node.bytes)?;
        if node.iter().any(|f| f.number == 4 && matches!(f.bytes, b"PRelu" | b"LeakyRelu")) {
            if let (Some(input), Some(output)) = (node.iter().find(|f| f.number == 1), node.iter().find(|f| f.number == 2)) {
                activations.insert(input.bytes, output.bytes);
            }
        }
    }
    let mut added = Vec::new();
    let axes_name = b"__neoarc_stable_fp16_axes";
    if opset >= 18 {
        let mut tensor = Vec::new();
        integer(&mut tensor, 1, 4); integer(&mut tensor, 2, 7);
        message(&mut tensor, 8, axes_name); message(&mut tensor, 7, &[0, 1, 2, 3]);
        message(&mut added, 5, &tensor);
    }
    let mut count = 0;
    let mut summary = Vec::new();
    for node in graph_fields.iter().filter(|f| f.number == 1) {
        let node = fields(node.bytes)?;
        if !node.iter().any(|f| f.number == 4 && f.bytes == b"Conv") { continue; }
        let inputs: Vec<_> = node.iter().filter(|f| f.number == 1).collect();
        if inputs.len() < 2 || !fp16_weights.contains(inputs[1].bytes) { continue; }
        ensure!(node.iter().find(|f| f.number == 7).is_none_or(|f| f.bytes.is_empty()), "custom Conv domain");
        let source = node.iter().find(|f| f.number == 2).ok_or_else(|| anyhow!("Conv output missing"))?.bytes;
        let source = activations.get(source).copied().unwrap_or(source);
        let name = format!("__neoarc_stable_fp16_{count}");
        let mut reduction = Vec::new();
        message(&mut reduction, 1, source);
        if opset >= 18 { message(&mut reduction, 1, axes_name); }
        message(&mut reduction, 2, name.as_bytes()); message(&mut reduction, 4, b"ReduceMean");
        if opset < 18 {
            let mut attr = Vec::new();
            message(&mut attr, 1, b"axes"); message(&mut attr, 8, &[0, 1, 2, 3]); integer(&mut attr, 20, 7);
            message(&mut reduction, 5, &attr);
        }
        let mut keep = Vec::new();
        message(&mut keep, 1, b"keepdims"); integer(&mut keep, 3, 1); integer(&mut keep, 20, 2);
        message(&mut reduction, 5, &keep); message(&mut added, 1, &reduction);
        message(&mut summary, 1, name.as_bytes());
        count += 1;
    }
    ensure!(count > 0, "no FP16 convolutions eligible for stabilization");
    let name = b"__neoarc_stable_fp16_summary";
    message(&mut summary, 2, name); message(&mut summary, 4, b"Concat");
    let mut axis = Vec::new(); message(&mut axis, 1, b"axis"); integer(&mut axis, 3, 1); integer(&mut axis, 20, 2);
    message(&mut summary, 5, &axis); message(&mut added, 1, &summary);
    let mut shape = Vec::new();
    for size in [1, count, 1, 1] { let mut dim = Vec::new(); integer(&mut dim, 1, size); message(&mut shape, 1, &dim); }
    let mut tensor = Vec::new(); integer(&mut tensor, 1, 10); message(&mut tensor, 2, &shape);
    let mut ty = Vec::new(); message(&mut ty, 1, &tensor);
    let mut output = Vec::new(); message(&mut output, 1, name); message(&mut output, 2, &ty);
    message(&mut added, 12, &output);
    let mut new_graph = graph.bytes.to_vec(); new_graph.extend_from_slice(&added);
    let mut result = Vec::new();
    for field in model {
        if field.number == 7 { message(&mut result, 7, &new_graph); }
        else { result.extend_from_slice(field.raw); }
    }
    Ok((result, usize::try_from(count)?))
}

pub(super) fn prepare(cache: &Path, original: &Path) -> Result<PathBuf> {
    let destination = cache_path(cache, original)?;
    if destination.is_file() { return Ok(destination); }
    let (bytes, count) = patched(&std::fs::read(original)?)?;
    std::fs::create_dir_all(cache)?;
    static NEXT_WRITE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = NEXT_WRITE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = destination.with_extension(format!("{}.{sequence}.tmp", std::process::id()));
    std::fs::write(&temporary, &bytes)?;
    if let Err(error) = std::fs::rename(&temporary, &destination) {
        let _ = std::fs::remove_file(&temporary);
        if destination.is_file() { return Ok(destination); }
        return Err(error.into());
    }
    log::info!("neoarc-stable-fp16-graph: model={} boundaries={} extra_output_bytes={} cache={}", original.display(), count, count*2, destination.display());
    Ok(destination)
}

#[path="neoarc_graph/tanh_gate.rs"]
mod tanh_gate;
pub(super) use tanh_gate::prepare_tanh_gates;

#[path="neoarc_graph/channel_select.rs"]
mod channel_select;
pub(super) use channel_select::ChannelSelectPlan;

#[path="neoarc_graph/compact.rs"]
mod compact;
pub(super) use compact::{CompactPlan,discover_compact};

#[path="neoarc_graph/flow_channels.rs"]
mod flow_channels;
pub(super) use flow_channels::prepare_flow_channels;

#[path="neoarc_graph/concat_prefix.rs"]
mod concat_prefix;
pub(super) use concat_prefix::prepare_concat_prefix;
