//! Algebraically identical gate; FP16 differences are checked at runtime.
use super::*;
use super::dw_branch::{values,is_op,scalar_attr,list_attr,repeated,tensor,node};

pub(in crate::render) fn prepare_tanh_gates(cache:&Path,original:&Path)->Result<Option<PathBuf>>{
    let source=std::fs::read(original)?;let model=fields(&source)?;
    if model.iter().any(|f|f.number==25){return Ok(None);}
    let graph=fields(model.iter().find(|f|f.number==7).ok_or_else(||anyhow!("graph missing"))?.bytes)?;
    let nodes:Vec<_>=graph.iter().filter(|f|f.number==1).map(|f|fields(f.bytes)).collect::<Result<_>>()?;
    let prefix=b"__neoarc_tanh_gate_v1_";
    if nodes.iter().flat_map(|n|n.iter()).any(|f|matches!(f.number,1|2)&&f.bytes.starts_with(prefix)){return Ok(None);}
    let mut weights=HashMap::new();
    for t in graph.iter().filter(|f|f.number==5){
        let fs=fields(t.bytes)?;if fs.iter().any(|f|f.number==13){return Ok(None);}
        if let Some(name)=fs.iter().find(|f|f.number==8){if name.bytes.starts_with(prefix){return Ok(None);}weights.insert(name.bytes,fs);}
    }
    let mut producer=HashMap::new();let mut users:HashMap<&[u8],Vec<usize>>=HashMap::new();
    for(i,n)in nodes.iter().enumerate(){for out in values(n,2){producer.insert(out,i);}for input in values(n,1){users.entry(input).or_default().push(i);}}
    let two=b"__neoarc_tanh_gate_v1_two";let one=b"__neoarc_tanh_gate_v1_one";
    let mut replacements=HashMap::new();let mut count=0;
    for(i,n)in nodes.iter().enumerate(){
        if !is_op(n,b"Tanh"){continue;}
        let qualified=(||->Result<()>{
            let input=values(n,1);let output=values(n,2);ensure!(input.len()==1&&output.len()==1,"gate arity");
            let conv=&nodes[*producer.get(input[0]).ok_or_else(||anyhow!("projection missing"))?];let ci=values(conv,1);
            ensure!(is_op(conv,b"Conv")&&ci.len()>=2&&scalar_attr(conv,b"group",1)?==1,"dense gate projection required");
            let t=weights.get(ci[1]).ok_or_else(||anyhow!("constant projection required"))?;let dims=repeated(t,1)?;
            ensure!(t.iter().any(|f|f.number==2&&f.value==10)&&dims.len()==4&&dims[0]>=16&&dims[0]<=256&&dims[0]==dims[1]&&dims[2..]==[1,1],"FP16 square pointwise projection required");
            ensure!(t.iter().any(|f|f.number==9&&f.bytes.len()==(dims[0]*dims[1]*2)as usize),"inline weights required");
            ensure!(list_attr(conv,b"strides",&[1,1])?==[1,1]&&list_attr(conv,b"dilations",&[1,1])?==[1,1]&&list_attr(conv,b"pads",&[0,0,0,0])?==[0,0,0,0],"pointwise geometry");
            let consumers=users.get(output[0]).ok_or_else(||anyhow!("unused gate"))?;ensure!(consumers.len()==1,"shared gate output");
            let mul=&nodes[consumers[0]];ensure!(is_op(mul,b"Mul"),"gate product required");
            let other=values(mul,1).into_iter().find(|v|*v!=output[0]).ok_or_else(||anyhow!("residual missing"))?;
            let add=&nodes[*producer.get(other).ok_or_else(||anyhow!("residual add missing"))?];
            ensure!(is_op(add,b"Add")&&values(add,1).contains(&ci[0]),"residual/projection provenance mismatch");Ok(())
        })();
        if qualified.is_err(){continue;}
        let input=values(n,1)[0];let output=values(n,2)[0];let base=format!("__neoarc_tanh_gate_v1_{count}");
        let twice=format!("{base}_twice");let sigmoid=format!("{base}_sigmoid");let scaled=format!("{base}_scaled");let mut bytes=Vec::new();
        // tanh(x) = 2*sigmoid(2*x)-1. Keep all constants FP16 and let the
        // GPU compiler fuse supported sigmoid/linear post-operations.
        for(op,inputs,out)in [(b"Mul".as_slice(),vec![input,two.as_slice()],twice.as_bytes()),(b"Sigmoid",vec![twice.as_bytes()],sigmoid.as_bytes()),(b"Mul",vec![sigmoid.as_bytes(),two.as_slice()],scaled.as_bytes()),(b"Sub",vec![scaled.as_bytes(),one.as_slice()],output)]{
            let nn=node(op,out,&inputs,out,&[]);message(&mut bytes,1,&nn);
        }
        replacements.insert(i,bytes);count+=1;
    }
    if count==0{return Ok(None);}
    let mut body=Vec::new();let mut index=0;
    for f in graph{if f.number==1{if let Some(new)=replacements.get(&index){body.extend_from_slice(new);}else{body.extend_from_slice(f.raw);}index+=1;}else{body.extend_from_slice(f.raw);}}
    for t in [tensor(two,10,&[],&0x4000u16.to_le_bytes()),tensor(one,10,&[],&0x3c00u16.to_le_bytes())]{message(&mut body,5,&t);}
    let mut bytes=Vec::new();for f in model{if f.number==7{message(&mut bytes,7,&body);}else{bytes.extend_from_slice(f.raw);}}
    let path=cache.join("tanh-gate-v1").join(format!("{:x}.onnx",Sha256::digest(&source)));write_derived(&path,&bytes)?;
    log::info!("neoarc-tanh-gate-graph: model={} gates={} path={} validation=pending",original.display(),count,path.display());Ok(Some(path))
}
