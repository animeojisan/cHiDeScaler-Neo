//! Infer real port types with OpenVINO; never guess flow-channel precision.
use super::*;
use super::dw_branch::{Tensor,values,is_op,tensor,node};
struct Target { index:usize,source:usize,begin:i64,end:i64 }
pub(in crate::render) struct ChannelSelectPlan {
    source:Vec<u8>,targets:Vec<Target>,digest:String,
    pub(in crate::render) analysis:PathBuf,
    pub(in crate::render) names:Vec<Vec<u8>>,
}
impl ChannelSelectPlan {
    pub(in crate::render) fn qualified_480p(&self)->bool{
        // Topology matching is generic; rollout is limited to the exact model
        // selected for controlled experiments; not qualified for default use.
        matches!(self.digest.as_str(),"9d6c7f6f3d63ebe5b4db0d715f9760c949d7dc1fe8142fce48ccfedcca62a477"|"90deed0d4b672c918e112a760107455a9a1689db22b5b45311b23953ab3dd35f")
    }

    pub(in crate::render) fn discover(cache:&Path,original:&Path)->Result<Option<Self>>{
        let source=std::fs::read(original)?;let model=fields(&source)?;if model.iter().any(|f|f.number==25){return Ok(None);}
        let graph=fields(model.iter().find(|f|f.number==7).ok_or_else(||anyhow!("graph missing"))?.bytes)?;
        let nodes:Vec<_>=graph.iter().filter(|f|f.number==1).map(|f|fields(f.bytes)).collect::<Result<_>>()?;
        let prefix=b"__neoarc_channel_gather_v1_";
        if nodes.iter().flat_map(|n|n.iter()).any(|f|matches!(f.number,1|2)&&f.bytes.starts_with(prefix)){return Ok(None);}
        let mut constants=HashMap::new();
        for t in graph.iter().filter(|f|f.number==5){let fs=fields(t.bytes)?;if fs.iter().any(|f|f.number==13){return Ok(None);}if let Some(name)=fs.iter().find(|f|f.number==8){if name.bytes.starts_with(prefix){return Ok(None);}constants.insert(name.bytes,Tensor::parse(t.bytes)?);}}
        for n in &nodes{if is_op(n,b"Constant"){for a in n.iter().filter(|f|f.number==5){let fs=fields(a.bytes)?;if fs.iter().any(|f|f.number==1&&f.bytes==b"value"){if let(Some(name),Some(t))=(values(n,2).first(),fs.iter().find(|f|f.number==5)){constants.insert(*name,Tensor::parse(t.bytes)?);}}}}}
        let ints=|name:&[u8]|->Option<Vec<i64>>{let t=constants.get(name)?;if t.kind!=7||t.raw.len()%8!=0||t.raw.len()>256{return None;}Some(t.raw.chunks_exact(8).map(|b|i64::from_le_bytes(b.try_into().unwrap())).collect())};
        let mut targets=Vec::new();let mut names:Vec<Vec<u8>>=Vec::new();
        for(index,n)in nodes.iter().enumerate(){
            if !is_op(n,b"Slice"){continue;}let inputs=values(n,1);if inputs.len()<4||values(n,2).len()!=1{continue;}
            let(Some(begin),Some(end),Some(axes))=(ints(inputs[1]),ints(inputs[2]),ints(inputs[3]))else{continue;};
            if begin.len()!=1||end.len()!=1||!(axes==[1]||axes==[-3]){continue;}
            if inputs.len()>4&&!inputs[4].is_empty()&&ints(inputs[4]).as_deref()!=Some(&[1]){continue;}
            // A singleton interval can be normalized only after the actual
            // source-channel dimension is known. Include signed bounds here.
            let source_index=if let Some(i)=names.iter().position(|v|v.as_slice()==inputs[0]){i}else{names.push(inputs[0].to_vec());names.len()-1};
            targets.push(Target{index,source:source_index,begin:begin[0],end:end[0]});
        }
        if targets.is_empty(){return Ok(None);}
        // This metadata-only model is read/reshaped, never compiled or run.
        // Untyped diagnostic outputs let the frontend infer FP16 vs FP32.
        let mut body=Vec::new();for f in graph{if f.number!=12{body.extend_from_slice(f.raw);}}
        for name in &names{let mut value=Vec::new();message(&mut value,1,name);message(&mut body,12,&value);}
        let mut bytes=Vec::new();for f in model{if f.number==7{message(&mut bytes,7,&body);}else{bytes.extend_from_slice(f.raw);}}
        let digest=format!("{:x}",Sha256::digest(&source));let analysis=cache.join("channel-select-analysis-v1").join(format!("{digest}.onnx"));write_derived(&analysis,&bytes)?;
        Ok(Some(Self{source,targets,digest,analysis,names}))
    }
    pub(in crate::render) fn materialize(&self,cache:&Path,w:usize,h:usize,metadata:&[Option<(u64,usize)>])->Result<Option<PathBuf>>{
        ensure!(metadata.len()==self.names.len(),"channel metadata count mismatch");
        let model=fields(&self.source)?;let graph=fields(model.iter().find(|f|f.number==7).unwrap().bytes)?;
        let nodes:Vec<_>=graph.iter().filter(|f|f.number==1).map(|f|fields(f.bytes)).collect::<Result<_>>()?;
        let mut changed=HashMap::new();let mut tensors=Vec::new();
        for target in &self.targets{
            let Some((kind,c))=metadata[target.source]else{continue;};if !matches!(kind,1|10)||!matches!(c,2|4){continue;}
            let norm=|v:i64|if v<0{v.saturating_add(c as i64).clamp(0,c as i64)}else{v.min(c as i64)};
            let(begin,end)=(norm(target.begin),norm(target.end));if end-begin!=1{continue;}
            let base=format!("__neoarc_channel_gather_v1_{}",changed.len());let weight=format!("{base}_weight");
            // Gather copies the selected channel without floating-point
            // multiplication, conversion or a denormal-sensitive dot product.
            tensors.push(tensor(weight.as_bytes(),7,&[1],&(begin as i64).to_le_bytes()));
            let n=&nodes[target.index];let gather=node(b"Gather",base.as_bytes(),&[values(n,1)[0],weight.as_bytes()],values(n,2)[0],&[super::dw_branch::attribute(b"axis",&[1],true)]);changed.insert(target.index,gather);
        }
        if changed.is_empty(){return Ok(None);}
        let mut body=Vec::new();let mut index=0;for f in graph{if f.number==1{if let Some(n)=changed.get(&index){message(&mut body,1,n);}else{body.extend_from_slice(f.raw);}index+=1;}else{body.extend_from_slice(f.raw);}}
        for t in tensors{message(&mut body,5,&t);}let mut bytes=Vec::new();for f in model{if f.number==7{message(&mut bytes,7,&body);}else{bytes.extend_from_slice(f.raw);}}
        let path=cache.join("channel-gather-v1").join(&self.digest).join(format!("{w}x{h}.onnx"));write_derived(&path,&bytes)?;
        log::info!("neoarc-channel-gather-graph: copies={} shape={}x{} path={} precision=source-port-type",changed.len(),w,h,path.display());Ok(Some(path))
    }
}
