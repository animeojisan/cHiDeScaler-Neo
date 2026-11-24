//! Structural Compact CNN contract; model filenames never select execution.
use super::*;
use super::dw_branch::{Tensor,values,is_op,scalar_attr,list_attr,attr};
pub(in crate::render) struct CompactLayer {pub ic:usize,pub oc:usize,pub weights:Vec<u16>,pub bias:Vec<u16>,pub slope:Vec<f32>}
pub(in crate::render) struct CompactPlan {pub layers:Vec<CompactLayer>,pub scale:usize,pub digest:String}
fn text_attr<'a>(node:&[Field<'a>],name:&[u8])->Result<Option<&'a[u8]>>{Ok(attr(node,name)?.and_then(|a|a.into_iter().find(|f|f.number==4).map(|f|f.bytes)))}
fn half_values(t:&Tensor,count:usize)->Result<Vec<u16>>{ensure!(t.kind==10&&t.raw.len()==count*2,"Compact parameter must be inline FP16");Ok(t.raw.chunks_exact(2).map(|v|u16::from_le_bytes([v[0],v[1]])).collect())}
pub(in crate::render) fn discover_compact(original:&Path)->Result<Option<CompactPlan>>{
 let source=std::fs::read(original)?;let model=fields(&source)?;if model.iter().any(|f|f.number==25){return Ok(None);}
 let graph=fields(model.iter().find(|f|f.number==7).ok_or_else(||anyhow!("graph missing"))?.bytes)?;
 let nodes:Vec<_>=graph.iter().filter(|f|f.number==1).map(|f|fields(f.bytes)).collect::<Result<_>>()?;
 let convs:Vec<_>=nodes.iter().filter(|n|is_op(n,b"Conv")).collect();if !matches!(convs.len(),10|18){return Ok(None);}
 let build=(||->Result<CompactPlan>{
  let inputs:Vec<_>=graph.iter().filter(|f|f.number==11).collect();let outputs:Vec<_>=graph.iter().filter(|f|f.number==12).collect();ensure!(inputs.len()==1&&outputs.len()==1,"single RGB IO required");
  let io=|bytes:&[u8]|->Result<(Vec<u8>,u64)>{let v=fields(bytes)?;let name=v.iter().find(|f|f.number==1).ok_or_else(||anyhow!("IO name"))?.bytes.to_vec();let ty=fields(v.iter().find(|f|f.number==2).ok_or_else(||anyhow!("IO type"))?.bytes)?;let tensor=fields(ty.iter().find(|f|f.number==1).ok_or_else(||anyhow!("tensor IO"))?.bytes)?;Ok((name,tensor.iter().find(|f|f.number==1).ok_or_else(||anyhow!("tensor type"))?.value))};
  let(input,itype)=io(inputs[0].bytes)?;let(output,otype)=io(outputs[0].bytes)?;ensure!(itype==10&&otype==10,"FP16 RGB contract required");
  let mut tensors:HashMap<&[u8],Tensor>=HashMap::new();
  let parse=|bytes:&[u8]|->Result<Tensor>{let fs=fields(bytes)?;ensure!(!fs.iter().any(|f|f.number==13||f.number==14&&f.value!=0),"external parameter");let mut t=Tensor::parse(bytes)?;if t.kind==1&&t.raw.is_empty(){for f in fs.iter().filter(|f|f.number==4){t.raw.extend_from_slice(f.bytes);}}Ok(t)};
  for f in graph.iter().filter(|f|f.number==5){let fs=fields(f.bytes)?;if let Some(name)=fs.iter().find(|f|f.number==8){tensors.insert(name.bytes,parse(f.bytes)?);}}
  for n in &nodes{if is_op(n,b"Constant"){for f in n.iter().filter(|f|f.number==5){let fs=fields(f.bytes)?;if fs.iter().any(|f|f.number==1&&f.bytes==b"value"){if let Some(t)=fs.iter().find(|f|f.number==5){tensors.insert(values(n,2)[0],parse(t.bytes)?);}}}}}
  let mut producers=HashMap::new();for n in &nodes{for o in values(n,2){producers.insert(o,n);}}
  let strip=|name:&[u8]|->Result<Vec<u8>>{let mut value=name.to_vec();for _ in 0..16{let Some(n)=producers.get(value.as_slice())else{return Ok(value);};if is_op(n,b"Identity"){value=values(n,1)[0].to_vec();continue;}if is_op(n,b"Cast"){ensure!(matches!(scalar_attr(n,b"to",0)?,1|10),"non-float cast");value=values(n,1)[0].to_vec();continue;}return Ok(value);}Err(anyhow!("cast chain too deep"))};
  let mut current=input.clone();let mut layers=Vec::new();
  for(index,n)in convs.iter().enumerate(){let args=values(n,1);let out=values(n,2);ensure!(args.len()==3&&out.len()==1&&strip(args[0])?==current,"Compact CNN must be a straight chain");
   ensure!(scalar_attr(n,b"group",1)?==1&&list_attr(n,b"strides",&[1,1])?==[1,1]&&list_attr(n,b"dilations",&[1,1])?==[1,1]&&list_attr(n,b"pads",&[])?==[1,1,1,1],"unsupported convolution geometry");ensure!(text_attr(n,b"auto_pad")?.is_none_or(|v|v==b"NOTSET"),"auto padding");
   let w=tensors.get(args[1]).ok_or_else(||anyhow!("weight not constant"))?;ensure!(w.dims.len()==4&&w.dims[2..]==[3,3],"3x3 required");let(oc,ic)=(w.dims[0]as usize,w.dims[1]as usize);let last=index+1==convs.len();ensure!(ic==if index==0{3}else{64},"unsupported input channels");ensure!(oc==if last{oc}else{64},"unsupported features");if last{ensure!(matches!(oc,3|12),"unsupported output scale");}
   let b=tensors.get(args[2]).ok_or_else(||anyhow!("bias not constant"))?;ensure!(b.dims==[oc as u64],"bias shape");
   let slope=if !last{let act=nodes.iter().find(|a|is_op(a,b"PRelu")&&values(a,1).len()==2&&strip(values(a,1)[0]).is_ok_and(|v|v==out[0])).ok_or_else(||anyhow!("PRelu missing"))?;let a=values(act,1);let t=tensors.get(a[1]).ok_or_else(||anyhow!("slope not constant"))?;ensure!(t.dims.iter().product::<u64>()==64,"channel slope shape");current=values(act,2)[0].to_vec();half_values(t,64)?.into_iter().map(|v|half::f16::from_bits(v).to_f32()).collect()}else{current=out[0].to_vec();Vec::new()};
   layers.push(CompactLayer{ic,oc,weights:half_values(w,oc*ic*9)?,bias:half_values(b,oc)?,slope});
  }
  let scale=if layers.last().unwrap().oc==3{1}else{2};let mut root=strip(&output)?;
  // GL output is already clamped to [0,1] during half -> RGBA8 conversion.
  // An explicit final Clip with these exact bounds can share that conversion.
  if let Some(n)=producers.get(root.as_slice()){if is_op(n,b"Clip"){
   let args=values(n,1);ensure!(args.len()==3,"explicit RGB Clip bounds required");
   for (name,expected) in [(args[1],0f32),(args[2],1f32)]{let name=strip(name)?;let t=tensors.get(name.as_slice()).ok_or_else(||anyhow!("constant Clip bound required"))?;let value=if t.kind==10{half::f16::from_bits(half_values(t,1)?[0]).to_f32()}else{ensure!(t.kind==1&&t.raw.len()==4,"scalar float Clip bound required");f32::from_le_bytes(t.raw.as_slice().try_into().unwrap())};ensure!(value==expected,"only final unit RGB Clip can fuse into GL output");}
   root=strip(args[0])?;
  }}
  let add=producers.get(root.as_slice()).ok_or_else(||anyhow!("residual missing"))?;ensure!(is_op(add,b"Add")&&values(add,1).len()==2,"RGB residual Add required");
  let mut depth=false;let mut resized=false;
  for name in values(add,1){let value=strip(name)?;let n=producers.get(value.as_slice()).ok_or_else(||anyhow!("residual branch missing"))?;
   if is_op(n,b"DepthToSpace"){ensure!(!depth&&scalar_attr(n,b"blocksize",0)?==scale as u64&&strip(values(n,1)[0])?==current,"subpixel path");ensure!(scale==1||text_attr(n,b"mode")?==Some(b"CRD"),"subpixel order");depth=true;}
   else if is_op(n,b"Resize"){let args=values(n,1);ensure!(!resized&&args.len()>=3&&strip(args[0])?==input&&args.get(3).is_none_or(|v|v.is_empty()),"RGB resize path");ensure!(text_attr(n,b"mode")?==Some(b"nearest")&&text_attr(n,b"coordinate_transformation_mode")?==Some(b"asymmetric")&&text_attr(n,b"nearest_mode")?.is_none_or(|v|v==b"floor"),"nearest integer residual required");let scale_name=strip(args[2])?;let t=tensors.get(scale_name.as_slice()).ok_or_else(||anyhow!("static scale required"))?;let ss:Vec<f32>=if t.kind==10{half_values(t,4)?.into_iter().map(|v|half::f16::from_bits(v).to_f32()).collect()}else{ensure!(t.kind==1&&t.raw.len()==16,"float scale vector");t.raw.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect()};ensure!(ss==[1.,1.,scale as f32,scale as f32],"scale mismatch");resized=true;}
   else{return Err(anyhow!("unexpected residual operation"));}
  }
  ensure!(depth&&resized,"complete residual contract required");Ok(CompactPlan{layers,scale,digest:format!("{:x}",Sha256::digest(&source))})
 })();match build{Ok(p)=>Ok(Some(p)),Err(e)=>{log::debug!("neoarc-compact-not-selected: model={} reason={e:#}",original.display());Ok(None)}}
}
