//! Exact spatial/channel mapping; numerical rollout is validated by the GPU caller.
use super::*;

pub(super) fn repeated(fs:&[Field<'_>],number:u64)->Result<Vec<u64>>{
    let mut values=Vec::new();
    for f in fs.iter().filter(|f|f.number==number){
        if f.bytes.is_empty(){values.push(f.value);}else{let mut p=0;while p<f.bytes.len(){values.push(varint(f.bytes,&mut p)?);}}
    }
    Ok(values)
}
pub(super) fn attr<'a>(node:&[Field<'a>],name:&[u8])->Result<Option<Vec<Field<'a>>>>{
    for a in node.iter().filter(|f|f.number==5){let fs=fields(a.bytes)?;if fs.iter().any(|f|f.number==1&&f.bytes==name){return Ok(Some(fs));}}
    Ok(None)
}
pub(super) fn scalar_attr(node:&[Field<'_>],name:&[u8],default:u64)->Result<u64>{
    Ok(attr(node,name)?.and_then(|fs|fs.iter().find(|f|f.number==3).map(|f|f.value)).unwrap_or(default))
}
pub(super) fn list_attr(node:&[Field<'_>],name:&[u8],default:&[u64])->Result<Vec<u64>>{
    match attr(node,name)?{Some(fs)=>repeated(&fs,8),None=>Ok(default.to_vec())}
}
pub(super) struct Tensor { pub(super) kind:u64,pub(super) dims:Vec<u64>,pub(super) raw:Vec<u8> }
impl Tensor {
    pub(super) fn parse(bytes:&[u8])->Result<Self>{
        let fs=fields(bytes)?;let kind=fs.iter().find(|f|f.number==2).ok_or_else(||anyhow!("tensor type missing"))?.value;
        let dims=repeated(&fs,1)?;
        let raw=if let Some(f)=fs.iter().find(|f|f.number==9){f.bytes.to_vec()}
        else if kind==10{repeated(&fs,5)?.into_iter().flat_map(|x|(x as u16).to_le_bytes()).collect()}
        else if kind==7{repeated(&fs,7)?.into_iter().flat_map(u64::to_le_bytes).collect()}
        else{Vec::new()};
        Ok(Self{kind,dims,raw})
    }
    fn sizes(&self)->Result<Vec<u64>>{
        ensure!(self.kind==7&&self.dims.len()==1&&self.dims[0]<=256&&self.raw.len()==self.dims[0]as usize*8,"split lengths must be constant int64");
        Ok(self.raw.chunks_exact(8).map(|v|u64::from_le_bytes(v.try_into().unwrap())).collect())
    }
    fn half(&self,dims:&[u64])->Result<&[u8]>{
        ensure!(self.kind==10&&self.dims==dims&&self.raw.len()==dims.iter().product::<u64>()as usize*2,"expected inline FP16 weights");Ok(&self.raw)
    }
}
pub(super) fn tensor(name:&[u8],kind:u64,dims:&[u64],raw:&[u8])->Vec<u8>{
    let mut t=Vec::new();for d in dims{integer(&mut t,1,*d);}integer(&mut t,2,kind);message(&mut t,8,name);message(&mut t,9,raw);t
}
pub(super) fn attribute(name:&[u8],values:&[u64],scalar:bool)->Vec<u8>{
    let mut a=Vec::new();message(&mut a,1,name);integer(&mut a,20,if scalar{2}else{7});
    for v in values{integer(&mut a,if scalar{3}else{8},*v);}a
}
pub(super) fn node(op:&[u8],name:&[u8],inputs:&[&[u8]],output:&[u8],attrs:&[Vec<u8>])->Vec<u8>{
    let mut n=Vec::new();for i in inputs{message(&mut n,1,i);}message(&mut n,2,output);message(&mut n,3,name);message(&mut n,4,op);for a in attrs{message(&mut n,5,a);}n
}
pub(super) fn values<'a>(node:&[Field<'a>],number:u64)->Vec<&'a [u8]>{node.iter().filter(|f|f.number==number).map(|f|f.bytes).collect()}
pub(super) fn is_op(node:&[Field<'_>],op:&[u8])->bool{
    node.iter().any(|f|f.number==4&&f.bytes==op)&&node.iter().filter(|f|f.number==7).all(|f|f.bytes.is_empty()||f.bytes==b"ai.onnx")
}

pub(in crate::render) fn prepare_depthwise_branches(cache:&Path,original:&Path)->Result<Option<PathBuf>>{
    let source=std::fs::read(original)?;let model=fields(&source)?;
    // Local functions, external weights and subgraphs are deliberately outside
    // this rollout. No guessed dimensions or filename-based matching.
    if model.iter().any(|f|f.number==25){return Ok(None);}
    let graph=fields(model.iter().find(|f|f.number==7).ok_or_else(||anyhow!("graph missing"))?.bytes)?;
    let nodes:Vec<_>=graph.iter().filter(|f|f.number==1).map(|f|fields(f.bytes)).collect::<Result<_>>()?;
    if !nodes.iter().any(|n|is_op(n,b"Concat")&&values(n,1).len()==5)||!nodes.iter().any(|n|is_op(n,b"Split")){return Ok(None);}
    let prefix=b"__neoarc_dw_branch_v2_";
    if nodes.iter().flat_map(|n|n.iter()).any(|f|matches!(f.number,1|2)&&f.bytes.starts_with(prefix)){return Ok(None);}
    if graph.iter().filter(|f|f.number==5).any(|f|fields(f.bytes).is_ok_and(|fs|fs.iter().any(|f|f.number==8&&f.bytes.starts_with(prefix)))){return Ok(None);}
    let mut constants=HashMap::new();
    for f in graph.iter().filter(|f|f.number==5){let fs=fields(f.bytes)?;if let Some(name)=fs.iter().find(|f|f.number==8){constants.insert(name.bytes,Tensor::parse(f.bytes)?);}}
    for n in &nodes{if is_op(n,b"Constant"){if let(Some(out),Some(a))=(values(n,2).first(),attr(n,b"value")?){if let Some(t)=a.iter().find(|f|f.number==5){constants.insert(*out,Tensor::parse(t.bytes)?);}}}}
    let mut producers=HashMap::new();for(i,n)in nodes.iter().enumerate(){for out in values(n,2){producers.insert(out,i);}}
    let mut replaced=HashMap::new();let mut inserted:HashMap<usize,Vec<u8>>=HashMap::new();let mut renamed:HashMap<Vec<u8>,Vec<u8>>=HashMap::new();let mut added=Vec::new();let mut count=0;
    for(index,n)in nodes.iter().enumerate(){
        if !is_op(n,b"Concat")||values(n,1).len()!=5||scalar_attr(n,b"axis",0)?!=1{continue;}
        let candidate=(||->Result<(Vec<u8>,Vec<Vec<u8>>,usize,Vec<u8>,Vec<u8>,Vec<u8>)>{
            let inputs=values(n,1);let output=values(n,2);ensure!(output.len()==1,"concat output count");
            let outer=&nodes[*producers.get(inputs[0]).ok_or_else(||anyhow!("outer split missing"))?];
            ensure!(is_op(outer,b"Split")&&scalar_attr(outer,b"axis",0)?==1,"channel split required");
            let oi=values(outer,1);let oo=values(outer,2);ensure!(oi.len()==2&&oo.len()==3&&oo[1]==inputs[0],"outer split layout");
            let parts=constants.get(oi[1]).ok_or_else(||anyhow!("dynamic split"))?.sizes()?;
            ensure!(parts.len()==3&&parts.iter().all(|x|*x>0&&*x<=256)&&parts[0]==parts[1]+parts[2]&&parts[0]<=256,"unsupported outer channels");
            let inner=&nodes[*producers.get(inputs[1]).ok_or_else(||anyhow!("inner split missing"))?];
            ensure!(is_op(inner,b"Split")&&scalar_attr(inner,b"axis",0)?==1,"inner channel split");
            let ii=values(inner,1);let io=values(inner,2);ensure!(ii.len()==2&&io.len()==4&&ii[0]==oo[2]&&io[0]==inputs[1],"inner split layout");
            let lengths=constants.get(ii[1]).ok_or_else(||anyhow!("dynamic inner split"))?.sizes()?;
            ensure!(lengths.len()==4&&lengths.iter().all(|x|*x>0&&*x<=256)&&lengths.iter().sum::<u64>()==parts[2],"inner split lengths");
            let mut kernels=Vec::new();let mut max_h=1;let mut max_w=1;
            for k in 0..3{
                let conv=&nodes[*producers.get(inputs[k+2]).ok_or_else(||anyhow!("depthwise producer missing"))?];let ci=values(conv,1);
                ensure!(is_op(conv,b"Conv")&&ci.len()==3&&ci[0]==io[k+1],"depthwise branch layout");
                let channels=lengths[k+1];ensure!(scalar_attr(conv,b"group",1)?==channels,"depthwise group mismatch");
                let w=constants.get(ci[1]).ok_or_else(||anyhow!("dynamic weight"))?;let b=constants.get(ci[2]).ok_or_else(||anyhow!("dynamic bias"))?;
                ensure!(w.dims.len()==4&&w.dims[0]==channels&&w.dims[1]==1,"depthwise weights shape");
                let(h,ww)=(w.dims[2],w.dims[3]);ensure!(h>0&&ww>0&&h<=15&&ww<=15&&h%2==1&&ww%2==1,"unsupported kernel extent");
                ensure!(list_attr(conv,b"strides",&[1,1])?==[1,1]&&list_attr(conv,b"dilations",&[1,1])?==[1,1]&&list_attr(conv,b"pads",&[0,0,0,0])?==[h/2,ww/2,h/2,ww/2],"non-unit or asymmetric branch geometry");
                if let Some(ap)=attr(conv,b"auto_pad")?{ensure!(ap.iter().filter(|f|f.number==4).all(|f|f.bytes.is_empty()||f.bytes==b"NOTSET"),"auto padding unsupported");}
                let weight=w.half(&[channels,1,h,ww])?;let bias=b.half(&[channels])?;
                max_h=max_h.max(h);max_w=max_w.max(ww);kernels.push((channels,h,ww,weight,bias));
            }
            let channels=parts[0]as usize;let kh=max_h as usize;let kw=max_w as usize;
            let mut weights=vec![0u8;channels*kh*kw*2];let mut biases=vec![0u8;channels*2];let mut ch=(parts[1]+lengths[0])as usize;
            for c in 0..ch{let i=(c*kh*kw+(kh/2)*kw+kw/2)*2;weights[i..i+2].copy_from_slice(&0x3c00u16.to_le_bytes());}
            for(n,h,w,original,bias)in kernels{for c in 0..n as usize{for y in 0..h as usize{for x in 0..w as usize{let src=((c*h as usize+y)*w as usize+x)*2;let dst=(((ch+c)*kh+(kh-h as usize)/2+y)*kw+(kw-w as usize)/2+x)*2;weights[dst..dst+2].copy_from_slice(&original[src..src+2]);}}}biases[ch*2..(ch+n as usize)*2].copy_from_slice(bias);ch+=n as usize;}
            ensure!(ch==channels,"merged channel count");
            let dense_index=*producers.get(oi[0]).ok_or_else(||anyhow!("dense projection missing"))?;
            let dense=&nodes[dense_index];let di=values(dense,1);let dout=values(dense,2);
            ensure!(is_op(dense,b"Conv")&&di.len()==3&&dout.len()==1&&scalar_attr(dense,b"group",1)?==1,"dense projection required");
            let dw=constants.get(di[1]).ok_or_else(||anyhow!("constant dense weight required"))?;
            let db=constants.get(di[2]).ok_or_else(||anyhow!("constant dense bias required"))?;
            let total_channels=parts.iter().sum::<u64>();
            ensure!(dw.dims.len()==4&&dw.dims[0]==total_channels&&dw.dims[1]>0&&dw.dims[1]<=4096&&dw.dims[2]>0&&dw.dims[2]<=15&&dw.dims[3]>0&&dw.dims[3]<=15,"dense weight layout");
            let dense_weights=dw.half(&dw.dims)?;let dense_bias=db.half(&[total_channels])?;
            let row_bytes=dense_weights.len()/total_channels as usize;
            let split_at=parts[0]as usize;
            let base=format!("__neoarc_dw_branch_v2_{count}");let name=|suffix:&str|format!("{base}_{suffix}").into_bytes();
            let wn=name("weight");let bn=name("bias");let tail=name("tail");let gate=name("gate");
            let mut tensors=vec![tensor(&wn,10,&[channels as u64,1,max_h,max_w],&weights),tensor(&bn,10,&[channels as u64],&biases)];
            let mut projections=Vec::new();
            let attributes:Vec<_>=dense.iter().filter(|f|f.number==5).map(|f|f.bytes.to_vec()).collect();
            for(suffix,begin,end,out)in [("gate",0,split_at,&gate),("tail",split_at,total_channels as usize,&tail)]{
                let wname=name(&format!("{suffix}_weight"));let bname=name(&format!("{suffix}_bias"));
                let mut dims=dw.dims.clone();dims[0]=(end-begin)as u64;
                tensors.push(tensor(&wname,10,&dims,&dense_weights[begin*row_bytes..end*row_bytes]));
                tensors.push(tensor(&bname,10,&[(end-begin)as u64],&dense_bias[begin*2..end*2]));
                let projection=node(b"Conv",&name(&format!("{suffix}_conv")),&[di[0],&wname,&bname],out,&attributes);
                message(&mut projections,1,&projection);
            }
            let conv=node(b"Conv",&name("conv"),&[&tail,&wn,&bn],output[0],&[attribute(b"group",&[channels as u64],true),attribute(b"pads",&[max_h/2,max_w/2,max_h/2,max_w/2],false),attribute(b"kernel_shape",&[max_h,max_w],false)]);
            let mut bytes=Vec::new();message(&mut bytes,1,&conv);Ok((bytes,tensors,dense_index,projections,oo[0].to_vec(),gate))
        })();
        if let Ok((bytes,tensors,dense_index,projections,old_gate,new_gate))=candidate{
            replaced.insert(index,bytes);added.extend(tensors);inserted.entry(dense_index).or_default().extend(projections);renamed.insert(old_gate,new_gate);count+=1;
        }
    }
    if count==0{return Ok(None);}
    let mut body=Vec::new();let mut index=0;
    for f in graph{
        if f.number==1{
            if let Some(new)=replaced.get(&index){body.extend_from_slice(new);}
            else{
                let mut changed=Vec::new();for part in fields(f.bytes)?{
                    if part.number==1{if let Some(new)=renamed.get(part.bytes){message(&mut changed,1,new);continue;}}
                    changed.extend_from_slice(part.raw);
                }
                message(&mut body,1,&changed);
            }
            if let Some(extra)=inserted.get(&index){body.extend_from_slice(extra);}
            index+=1;
        }else{body.extend_from_slice(f.raw);}
    }
    for t in added{message(&mut body,5,&t);}let mut result=Vec::new();for f in model{if f.number==7{message(&mut result,7,&body);}else{result.extend_from_slice(f.raw);}}
    let path=cache.join("dw-branch-v2").join(format!("{:x}.onnx",Sha256::digest(&source)));write_derived(&path,&result)?;
    log::info!("neoarc-depthwise-graph: model={} branches={} path={} precision=fp16 validation=pending",original.display(),count,path.display());Ok(Some(path))
}
