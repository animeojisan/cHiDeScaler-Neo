//! Reuse a preceding dense block's identical channel prefix, without changing
//! any values, arithmetic, learned weights, or the original model file.
use super::*;
use super::dw_branch::{is_op,values,scalar_attr};

fn rewrite(source:&[u8])->Result<Option<(Vec<u8>,usize)>> {
    let model=fields(source)?;
    let graph=fields(model.iter().find(|f|f.number==7).ok_or_else(||anyhow!("graph missing"))?.bytes)?;
    // Only standard-domain, explicitly channel-axis Concat nodes qualify.
    // Keep original input tuples as keys even after rewriting a prefix node.
    let mut prefixes:HashMap<Vec<Vec<u8>>,Vec<u8>>=HashMap::new();
    let mut body=Vec::new();let mut changed=0;
    for field in graph {
        if field.number!=1 {body.extend_from_slice(field.raw);continue;}
        let node=fields(field.bytes)?;
        if !is_op(&node,b"Concat") || scalar_attr(&node,b"axis",u64::MAX)?!=1
            || node.iter().any(|f|f.number==7&&!f.bytes.is_empty()) {
            message(&mut body,1,field.bytes);continue;
        }
        let inputs:Vec<Vec<u8>>=values(&node,1).into_iter().map(|v|v.to_vec()).collect();
        let outputs=values(&node,2);
        if inputs.len()<2||inputs.iter().any(Vec::is_empty)||outputs.len()!=1 {
            message(&mut body,1,field.bytes);continue;
        }
        if inputs.len()>2 {
            if let Some(prefix)=prefixes.get(&inputs[..inputs.len()-1]) {
                let mut replacement=Vec::new();message(&mut replacement,1,prefix);
                message(&mut replacement,1,inputs.last().unwrap());
                for f in &node {if f.number!=1 {replacement.extend_from_slice(f.raw);}}
                message(&mut body,1,&replacement);changed+=1;
            } else {message(&mut body,1,field.bytes);}
        } else {message(&mut body,1,field.bytes);}
        prefixes.insert(inputs,outputs[0].to_vec());
    }
    if changed==0{return Ok(None);}
    let mut result=Vec::new();for f in model {if f.number==7 {message(&mut result,7,&body);}else{result.extend_from_slice(f.raw);}}
    Ok(Some((result,changed)))
}
pub(in crate::render) fn prepare_concat_prefix(cache:&Path,original:&Path)->Result<Option<PathBuf>> {
    let source=std::fs::read(original)?;
    let Some((result,count))=rewrite(&source)? else{return Ok(None);};
    let path=cache.join("concat-prefix-v1").join(format!("{:x}.onnx",Sha256::digest(&source)));
    write_derived(&path,&result)?;
    log::info!("neoarc-concat-prefix-graph: model={} reused_prefixes={} validation=pending",original.display(),count);
    Ok(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn concat(inputs:&[&[u8]],output:&[u8],axis:u64)->Vec<u8>{
        let mut n=Vec::new();for i in inputs{message(&mut n,1,i);}message(&mut n,2,output);message(&mut n,4,b"Concat");
        let mut a=Vec::new();message(&mut a,1,b"axis");integer(&mut a,3,axis);integer(&mut a,20,2);message(&mut n,5,&a);n
    }
    #[test] fn prefix_requires_matching_order_and_axis(){
        let mut g=Vec::new();for n in [concat(&[b"x",b"a"],b"xa",1),concat(&[b"x",b"a",b"b"],b"xab",1),concat(&[b"x",b"a",b"b",b"c"],b"xabc",1),concat(&[b"a",b"x",b"b"],b"wrong_order",1),concat(&[b"x",b"a",b"b"],b"wrong_axis",2)]{message(&mut g,1,&n);}
        let mut m=Vec::new();message(&mut m,7,&g);let (out,n)=rewrite(&m).unwrap().unwrap();assert_eq!(n,2);
        let outer=fields(&out).unwrap();let nodes=fields(outer[0].bytes).unwrap();
        assert_eq!(values(&fields(nodes[1].bytes).unwrap(),1),vec![b"xa".as_slice(),b"b".as_slice()]);
        assert_eq!(values(&fields(nodes[2].bytes).unwrap(),1),vec![b"xab".as_slice(),b"c".as_slice()]);
        assert_eq!(values(&fields(nodes[3].bytes).unwrap(),1).len(),3);
        assert_eq!(values(&fields(nodes[4].bytes).unwrap(),1).len(),3);
    }
}

