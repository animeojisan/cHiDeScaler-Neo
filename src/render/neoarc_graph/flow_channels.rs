//! v974: TemporalFix splits two-channel flow tensors into single channels,
//! applies the same per-element arithmetic to each, and concatenates them
//! again. On OpenVINO GPU those single-channel slices of blocked (fsv16)
//! tensors run on reference kernels. Keep the arithmetic, drop the split:
//!   A: Concat_k(Mul(Slice_k(X), c))           -> Mul(X, c)
//!   B: Concat_k(Add(E_k, Mul(Slice_k(X), s_k))) -> Add(Concat_k(E_k), Mul(X, Concat_k(s_k)))
//! Every output element is produced by the same operations on the same
//! operands, so the result is unchanged (B570 640x480: identical bytes).
use super::*;
use super::dw_branch::{values, is_op, scalar_attr, repeated, node};

struct Const { kind: u64, raw: Vec<u8> }

fn tensor_const(t: &[u8]) -> Result<Option<Const>> {
    let fs = fields(t)?;
    if fs.iter().any(|f| f.number == 13 || f.number == 14) { return Ok(None); }
    let kind = fs.iter().find(|f| f.number == 2).map(|f| f.value).unwrap_or(0);
    if let Some(r) = fs.iter().find(|f| f.number == 9) {
        return Ok(Some(Const { kind, raw: r.bytes.to_vec() }));
    }
    if kind == 7 {
        let v = repeated(&fs, 7)?;
        if !v.is_empty() { return Ok(Some(Const { kind, raw: v.iter().flat_map(|x| x.to_le_bytes()).collect() })); }
    }
    Ok(None)
}

fn int_scalar(c: Option<&Const>) -> Option<i64> {
    let c = c?;
    (c.kind == 7 && c.raw.len() == 8).then(|| i64::from_le_bytes(c.raw[..8].try_into().unwrap()))
}

pub(in crate::render) fn prepare_flow_channels(cache: &Path, original: &Path) -> Result<Option<PathBuf>> {
    let source = std::fs::read(original)?;
    let model = fields(&source)?;
    let graph = fields(model.iter().find(|f| f.number == 7).ok_or_else(|| anyhow!("graph missing"))?.bytes)?;
    let nodes: Vec<_> = graph.iter().filter(|f| f.number == 1).map(|f| fields(f.bytes)).collect::<Result<_>>()?;
    let prefix = b"__neoarc_flow_channels_v1_";
    if nodes.iter().flat_map(|n| n.iter()).any(|f| matches!(f.number, 1 | 2) && f.bytes.starts_with(prefix)) { return Ok(None); }
    let mut consts: HashMap<&[u8], Const> = HashMap::new();
    for t in graph.iter().filter(|f| f.number == 5) {
        let fs = fields(t.bytes)?;
        if let (Some(name), Some(c)) = (fs.iter().find(|f| f.number == 8), tensor_const(t.bytes)?) { consts.insert(name.bytes, c); }
    }
    for n in &nodes {
        if !is_op(n, b"Constant") { continue; }
        let Some(out) = values(n, 2).first().copied() else { continue };
        for a in n.iter().filter(|f| f.number == 5) {
            let af = fields(a.bytes)?;
            if af.iter().any(|f| f.number == 1 && f.bytes == b"value") {
                if let Some(t) = af.iter().find(|f| f.number == 5) {
                    if let Some(c) = tensor_const(t.bytes)? { consts.insert(out, c); }
                }
            }
        }
    }
    let mut producer = HashMap::new();
    let mut users: HashMap<&[u8], usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for out in values(n, 2) { producer.insert(out, i); }
        for input in values(n, 1) { *users.entry(input).or_default() += 1; }
    }
    let graph_outputs: HashSet<&[u8]> = graph.iter().filter(|f| f.number == 12)
        .filter_map(|f| fields(f.bytes).ok()?.into_iter().find(|g| g.number == 1).map(|g| g.bytes)).collect();
    let single_use = |name: &[u8]| users.get(name).copied() == Some(1) && !graph_outputs.contains(name);
    // Slice of one channel k along axis 1 -> (source, k).
    let channel_slice = |name: &[u8]| -> Option<(&[u8], i64)> {
        let n = &nodes[*producer.get(name)?];
        if !is_op(n, b"Slice") { return None; }
        let ins = values(n, 1);
        if ins.len() < 4 { return None; }
        let (start, end, axis) = (int_scalar(consts.get(ins[1])), int_scalar(consts.get(ins[2])), int_scalar(consts.get(ins[3])));
        if axis != Some(1) || end != start.map(|s| s + 1) { return None; }
        if ins.len() > 4 && !ins[4].is_empty() && int_scalar(consts.get(ins[4])) != Some(1) { return None; }
        Some((ins[0], start?))
    };
    let mut replace: HashMap<usize, Vec<Vec<u8>>> = HashMap::new();
    let mut drop: HashSet<usize> = HashSet::new();
    let mut count = 0usize;
    for (i, n) in nodes.iter().enumerate() {
        if !is_op(n, b"Concat") || scalar_attr(n, b"axis", u64::MAX)? != 1 { continue; }
        let parts = values(n, 1);
        let Some(&out) = values(n, 2).first() else { continue };
        if parts.len() < 2 { continue; }
        let producers: Option<Vec<usize>> = parts.iter().map(|p| producer.get(p).copied()).collect();
        let Some(producers) = producers else { continue };
        if !parts.iter().all(|p| single_use(p)) { continue; }
        // A
        if producers.iter().all(|&p| is_op(&nodes[p], b"Mul") && values(&nodes[p], 1).len() == 2) {
            let slices: Option<Vec<_>> = producers.iter().map(|&p| channel_slice(values(&nodes[p], 1)[0])).collect();
            let scalars: Vec<_> = producers.iter().map(|&p| consts.get(values(&nodes[p], 1)[1])).collect();
            if let Some(slices) = slices {
                let same_source = slices.iter().all(|s| s.0 == slices[0].0);
                let in_order = slices.iter().enumerate().all(|(k, s)| s.1 == k as i64);
                let same_scalar = scalars.iter().all(|c| c.is_some_and(|c| c.raw.len() <= 4 && Some((c.kind, &c.raw)) == scalars[0].map(|f| (f.kind, &f.raw))));
                if same_source && in_order && same_scalar {
                    let c = values(&nodes[producers[0]], 1)[1];
                    let name = format!("__neoarc_flow_channels_v1_{count}");
                    replace.insert(i, vec![node(b"Mul", name.as_bytes(), &[slices[0].0, c], out, &[])]);
                    drop.extend(producers.iter().copied());
                    count += 1;
                    continue;
                }
            }
        }
        // B
        if producers.iter().all(|&p| is_op(&nodes[p], b"Add") && values(&nodes[p], 1).len() == 2) {
            let mut ok = true;
            let (mut grids, mut scales, mut sources, mut muls) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (k, &p) in producers.iter().enumerate() {
                let add = values(&nodes[p], 1);
                let Some(&m) = producer.get(add[1]) else { ok = false; break };
                let mul = values(&nodes[m], 1);
                if !is_op(&nodes[m], b"Mul") || mul.len() != 2 || !single_use(add[1]) { ok = false; break; }
                match channel_slice(mul[0]) {
                    Some((x, idx)) if idx == k as i64 => { sources.push(x); }
                    _ => { ok = false; break; }
                }
                grids.push(add[0]); scales.push(mul[1]); muls.push(m);
            }
            if ok && sources.iter().all(|s| *s == sources[0]) {
                let base = format!("__neoarc_flow_channels_v1_{count}");
                let (grid, scale, product) = (format!("{base}_grid"), format!("{base}_scale"), format!("{base}_product"));
                let axis = super::dw_branch::attribute(b"axis", &[1], true);
                replace.insert(i, vec![
                    node(b"Concat", grid.as_bytes(), &grids, grid.as_bytes(), &[axis.clone()]),
                    node(b"Concat", scale.as_bytes(), &scales, scale.as_bytes(), &[axis]),
                    node(b"Mul", product.as_bytes(), &[sources[0], scale.as_bytes()], product.as_bytes(), &[]),
                    node(b"Add", base.as_bytes(), &[grid.as_bytes(), product.as_bytes()], out, &[]),
                ]);
                drop.extend(producers.iter().copied());
                drop.extend(muls);
                count += 1;
            }
        }
    }
    if count == 0 { return Ok(None); }
    let mut new_graph = Vec::new();
    let mut index = 0usize;
    for f in &graph {
        if f.number != 1 { new_graph.extend_from_slice(f.raw); continue; }
        let current = index; index += 1;
        if drop.contains(&current) { continue; }
        match replace.get(&current) {
            Some(list) => for body in list { message(&mut new_graph, 1, body); },
            None => new_graph.extend_from_slice(f.raw),
        }
    }
    let mut result = Vec::new();
    for f in &model {
        if f.number == 7 { message(&mut result, 7, &new_graph); } else { result.extend_from_slice(f.raw); }
    }
    let path = cache.join("flow-channels-v1").join(format!("{:x}.onnx", Sha256::digest(&source)));
    write_derived(&path, &result)?;
    log::info!("neoarc-flow-channels-graph: model={} rewrites={count} cache={}", original.display(), path.display());
    Ok(Some(path))
}
