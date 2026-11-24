use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow, ensure};

use super::onnx_wire_varint;

const PREFIX: &str = "neo_rife_boost/";
const FLOAT: u64 = 1;
const HALF: u64 = 10;

fn varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 128 {
        out.push(value as u8 | 128);
        value >>= 7;
    }
    out.push(value as u8);
}

fn integer(out: &mut Vec<u8>, id: u64, value: u64) {
    varint(out, id << 3);
    varint(out, value);
}

fn message(out: &mut Vec<u8>, id: u64, value: &[u8]) {
    varint(out, id << 3 | 2);
    varint(out, value.len() as u64);
    out.extend_from_slice(value);
}

struct Field<'a> {
    id: u64,
    data: &'a [u8],
    raw: &'a [u8],
}

fn fields(bytes: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut offset = 0;
    let mut result = Vec::new();
    while offset < bytes.len() {
        let start = offset;
        let key = onnx_wire_varint(bytes, &mut offset)?;
        let data = match key & 7 {
            0 => {
                onnx_wire_varint(bytes, &mut offset)?;
                &[][..]
            }
            wire => {
                let length = match wire {
                    1 => 8,
                    2 => usize::try_from(onnx_wire_varint(bytes, &mut offset)?)?,
                    5 => 4,
                    _ => anyhow::bail!("unsupported ONNX wire type"),
                };
                let end = offset
                    .checked_add(length)
                    .ok_or_else(|| anyhow!("ONNX length overflow"))?;
                let data = bytes
                    .get(offset..end)
                    .ok_or_else(|| anyhow!("truncated ONNX field"))?;
                offset = end;
                data
            }
        };
        result.push(Field {
            id: key >> 3,
            data,
            raw: &bytes[start..offset],
        });
    }
    Ok(result)
}

fn name<'a>(fields: &[Field<'a>], id: u64) -> Result<&'a str> {
    let field = fields
        .iter()
        .find(|field| field.id == id)
        .ok_or_else(|| anyhow!("ONNX field {id} missing"))?;
    Ok(std::str::from_utf8(field.data)?)
}

struct Node<'a> {
    raw: &'a [u8],
    inputs: Vec<&'a str>,
    outputs: Vec<&'a str>,
    op: &'a str,
}

impl<'a> Node<'a> {
    fn parse(raw: &'a [u8]) -> Result<Self> {
        let fields = fields(raw)?;
        let names = |id| -> Result<Vec<&'a str>> {
            fields
                .iter()
                .filter(|field| field.id == id)
                .map(|field| Ok(std::str::from_utf8(field.data)?))
                .collect()
        };
        Ok(Self {
            raw,
            inputs: names(1)?,
            outputs: names(2)?,
            op: name(&fields, 4)?,
        })
    }

    fn append(&self, graph: &mut Graph, fixed_input: Option<&str>) -> Result<()> {
        if let Some(fixed) = fixed_input.filter(|_| self.inputs.contains(&"input")) {
            let mut node = Vec::new();
            for field in fields(self.raw)? {
                if field.id == 1 && field.data == b"input" {
                    message(&mut node, 1, fixed.as_bytes());
                } else {
                    node.extend_from_slice(field.raw);
                }
            }
            message(&mut graph.bytes, 1, &node);
        } else {
            message(&mut graph.bytes, 1, self.raw);
        }
        Ok(())
    }
}

#[derive(Default)]
struct Graph {
    bytes: Vec<u8>,
    next: usize,
}

impl Graph {
    fn temp(&mut self) -> String {
        self.next += 1;
        format!("{PREFIX}{}", self.next)
    }

    fn emit(
        &mut self,
        output: &str,
        op: &str,
        inputs: &[&str],
        attr: Option<(&str, &[u64], bool)>,
    ) {
        let mut node = Vec::new();
        for input in inputs {
            message(&mut node, 1, input.as_bytes());
        }
        message(&mut node, 2, output.as_bytes());
        message(&mut node, 4, op.as_bytes());
        if let Some((key, values, list)) = attr {
            let mut attribute = Vec::new();
            message(&mut attribute, 1, key.as_bytes());
            integer(&mut attribute, 20, if list { 7 } else { 2 });
            for value in values {
                integer(&mut attribute, if list { 8 } else { 3 }, *value);
            }
            message(&mut node, 5, &attribute);
        }
        message(&mut self.bytes, 1, &node);
    }

    fn node(&mut self, op: &str, inputs: &[&str], attr: Option<(&str, &[u64], bool)>) -> String {
        let output = self.temp();
        self.emit(&output, op, inputs, attr);
        output
    }

    fn tensor(&mut self, shape: &[u64], dtype: u64, raw: &[u8]) -> String {
        let output = self.temp();
        let mut tensor = Vec::new();
        for dim in shape {
            integer(&mut tensor, 1, *dim);
        }
        integer(&mut tensor, 2, dtype);
        message(&mut tensor, 8, output.as_bytes());
        message(&mut tensor, 9, raw);
        message(&mut self.bytes, 5, &tensor);
        output
    }

    fn ints(&mut self, values: &[i64]) -> String {
        let raw: Vec<_> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        self.tensor(&[values.len() as u64], 7, &raw)
    }

    fn scalar(&mut self, value: f32) -> String {
        self.tensor(&[], FLOAT, &value.to_le_bytes())
    }

    fn slice(&mut self, input: &str, axis: i64, begin: i64, end: i64) -> String {
        let starts = self.ints(&[begin]);
        let ends = self.ints(&[end]);
        let axes = self.ints(&[axis]);
        self.node("Slice", &[input, &starts, &ends, &axes], None)
    }

    fn cast(&mut self, input: &str, dtype: u64) -> String {
        self.node("Cast", &[input], Some(("to", &[dtype], false)))
    }

    fn transpose(&mut self, input: &str, perm: &[u64]) -> String {
        self.node("Transpose", &[input], Some(("perm", perm, true)))
    }

    fn value(&mut self, field: u64, name: &str, dtype: u64, shape: &[u64]) {
        let mut shape_proto = Vec::new();
        for dim in shape {
            let mut dimension = Vec::new();
            integer(&mut dimension, 1, *dim);
            message(&mut shape_proto, 1, &dimension);
        }
        let mut tensor = Vec::new();
        integer(&mut tensor, 1, dtype);
        message(&mut tensor, 2, &shape_proto);
        let mut ty = Vec::new();
        message(&mut ty, 1, &tensor);
        let mut value = Vec::new();
        message(&mut value, 1, name.as_bytes());
        message(&mut value, 2, &ty);
        message(&mut self.bytes, field, &value);
    }
}

struct Cut {
    flows: &'static [&'static str],
    mask: &'static str,
    nchw: bool,
    flow_type: u64,
    motion_alignment: u64,
}

fn cut(producers: &HashMap<&str, usize>, nodes: &[Node<'_>], dtype: u64) -> Result<Cut> {
    // The released RIFE exports have three final-warp layouts. In the batched
    // layouts, the first warp is frame 0 and the second is frame 1.
    let op = |name| producers.get(name).map(|&index| nodes[index].op);
    let cut = if dtype == FLOAT
        && op("diagparallel/sample_grid") == Some("Add")
        && op("diagparallel/flow_grid") == Some("Transpose")
    {
        Cut {
            flows: &["diagparallel/flow_grid"],
            mask: "diagparallel/mask_full",
            nchw: false,
            flow_type: FLOAT,
            motion_alignment: 1,
        }
    } else if dtype == HALF
        && op("/core/Add_23_output_0") == Some("Add")
        && op("/core/Add_24_output_0") == Some("Add")
        && op("/core/Concat_34_output_0") == Some("Concat")
        && op("/core/Concat_35_output_0") == Some("Concat")
    {
        Cut {
            flows: &["/core/Concat_34_output_0", "/core/Concat_35_output_0"],
            mask: "/core/Sigmoid_output_0",
            nchw: false,
            flow_type: FLOAT,
            motion_alignment: 32,
        }
    } else if dtype == HALF
        && op("/Transpose_5_output_0") == Some("Transpose")
        && op("/Add_8_output_0") == Some("Add")
        && op("/Mul_19_output_0") == Some("Mul")
    {
        Cut {
            flows: &["/Mul_19_output_0"],
            mask: "/Sigmoid_output_0",
            nchw: true,
            flow_type: HALF,
            motion_alignment: 64,
        }
    } else {
        anyhow::bail!("unrecognized RIFE final warp layout");
    };
    ensure!(producers.contains_key(cut.mask), "RIFE final mask missing");
    Ok(cut)
}

fn dependencies<'a>(
    nodes: &[Node<'a>],
    producers: &HashMap<&'a str, usize>,
    initializers: &HashMap<&'a str, &'a [u8]>,
    roots: &[&'a str],
    boundaries: &[&str],
) -> Result<(Vec<bool>, HashSet<&'a str>)> {
    let mut needed = vec![false; nodes.len()];
    let mut weights = HashSet::new();
    let mut pending = roots.to_vec();
    while let Some(name) = pending.pop() {
        if name.is_empty() || boundaries.contains(&name) {
            continue;
        }
        if let Some(&index) = producers.get(name) {
            if !std::mem::replace(&mut needed[index], true) {
                pending.extend(nodes[index].inputs.iter().copied());
            }
        } else {
            ensure!(
                initializers.contains_key(name),
                "unresolved ONNX input {name}"
            );
            weights.insert(name);
        }
    }
    Ok((needed, weights))
}

fn assemble(model: &[Field<'_>], graph: &Graph, label: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(graph.bytes.len() + 128);
    for field in model {
        if field.id == 7 {
            let mut graph_name = Vec::new();
            message(&mut graph_name, 2, label);
            varint(&mut bytes, 7 << 3 | 2);
            varint(&mut bytes, (graph.bytes.len() + graph_name.len()) as u64);
            bytes.extend_from_slice(&graph.bytes);
            bytes.extend_from_slice(&graph_name);
        } else {
            bytes.extend_from_slice(field.raw);
        }
    }
    bytes
}

/// v963: element type of the graph input named "input" (true = FP16).
/// NeoArc compiles the original ONNX directly, so the split graphs keep the
/// model's own IO precision (FP16 standard models, FP32 Lite50/Lite75).
pub(super) fn input_is_half(bytes: &[u8]) -> Result<bool> {
    let model = fields(bytes)?;
    let graph = fields(
        model
            .iter()
            .find(|field| field.id == 7)
            .ok_or_else(|| anyhow!("ONNX graph missing"))?
            .data,
    )?;
    for field in graph.iter().filter(|field| field.id == 11) {
        let info = fields(field.data)?;
        if name(&info, 1)? != "input" {
            continue;
        }
        let kind = fields(info.iter().find(|f| f.id == 2).ok_or_else(|| anyhow!("ONNX input type missing"))?.data)?;
        let tensor = fields(kind.iter().find(|f| f.id == 1).ok_or_else(|| anyhow!("ONNX input is not a tensor"))?.data)?;
        let elem = tensor.iter().find(|f| f.id == 1).ok_or_else(|| anyhow!("ONNX input element type missing"))?;
        let mut offset = 0;
        onnx_wire_varint(elem.raw, &mut offset)?;
        let value = onnx_wire_varint(elem.raw, &mut offset)?;
        return match value {
            HALF => Ok(true),
            FLOAT => Ok(false),
            other => Err(anyhow!("unsupported RIFE input element type {other}")),
        };
    }
    Err(anyhow!("ONNX graph input 'input' missing"))
}

// Adapt just the exchange tensors to the FP16 GPU bridge. Model weights,
/// internal operators, and any FP32 motion state keep their original types.
pub(super) fn half_bridge(bytes: &[u8], wrap_output: bool) -> Result<Vec<u8>> {
    const INPUT: &str = "__neo_bridge_float_input";
    const OUTPUT: &str = "__neo_bridge_float_output";
    let model = fields(bytes)?;
    let graph = model.iter().find(|f| f.id == 7).ok_or_else(|| anyhow!("missing ONNX graph"))?;
    let graph_fields = fields(graph.data)?;
    let mut rewritten = Graph { bytes: Vec::new(), next: 0 };
    rewritten.emit(INPUT, "Cast", &["input"], Some(("to", &[FLOAT], false)));
    for field in graph_fields {
        if field.id == 1 {
            let mut node = Vec::new();
            for part in fields(field.data)? {
                ensure!(part.id != 1 && part.id != 2 || (part.data != INPUT.as_bytes() && part.data != OUTPUT.as_bytes()), "bridge name collision");
                match (part.id, part.data) {
                    (1, b"input") => message(&mut node, 1, INPUT.as_bytes()),
                    (1 | 2, b"output") if wrap_output => message(&mut node, part.id, OUTPUT.as_bytes()),
                    _ => node.extend_from_slice(part.raw),
                }
            }
            message(&mut rewritten.bytes, 1, &node);
        } else if matches!(field.id, 11 | 12 | 13) {
            let info = fields(field.data)?;
            let port = name(&info, 1)?;
            if (field.id == 11 && port == "input") || (wrap_output && field.id == 12 && port == "output") {
                let mut new_info = Vec::new();
                for item in info {
                    if item.id == 2 {
                        let mut ty = Vec::new();
                        for t in fields(item.data)? {
                            if t.id == 1 {
                                let mut tensor = Vec::new();
                                for entry in fields(t.data)? {
                                    if entry.id == 1 { integer(&mut tensor, 1, HALF); }
                                    else { tensor.extend_from_slice(entry.raw); }
                                }
                                message(&mut ty, 1, &tensor);
                            } else { ty.extend_from_slice(t.raw); }
                        }
                        message(&mut new_info, 2, &ty);
                    } else { new_info.extend_from_slice(item.raw); }
                }
                message(&mut rewritten.bytes, field.id, &new_info);
            } else if !(field.id == 13 && (port == "input" || (wrap_output && port == "output"))) {
                rewritten.bytes.extend_from_slice(field.raw);
            }
        } else {
            rewritten.bytes.extend_from_slice(field.raw);
        }
    }
    if wrap_output { rewritten.emit("output", "Cast", &[OUTPUT], Some(("to", &[HALF], false))); }
    let mut result = Vec::new();
    for field in model {
        if field.id == 7 { message(&mut result, 7, &rewritten.bytes); }
        else { result.extend_from_slice(field.raw); }
    }
    Ok(result)
}

pub(super) fn split(bytes: &[u8], h: i64, w: i64, fp16: bool) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(h > 0 && w > 0, "invalid RIFE dimensions");
    let dtype = if fp16 { HALF } else { FLOAT };
    let model = fields(bytes)?;
    let graph = fields(
        model
            .iter()
            .find(|field| field.id == 7)
            .ok_or_else(|| anyhow!("ONNX graph missing"))?
            .data,
    )?;
    let mut nodes = Vec::new();
    let mut initializers = HashMap::new();
    let mut producers = HashMap::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for field in graph {
        match field.id {
            1 => {
                let node = Node::parse(field.data)?;
                for &output in &node.outputs {
                    ensure!(
                        !output.starts_with(PREFIX) && output != "state",
                        "reserved ONNX name"
                    );
                    ensure!(
                        producers.insert(output, nodes.len()).is_none(),
                        "duplicate ONNX output"
                    );
                }
                nodes.push(node);
            }
            5 => {
                let initializer = fields(field.data)?;
                let key = name(&initializer, 8)?;
                ensure!(
                    !key.starts_with(PREFIX) && key != "state",
                    "reserved ONNX name"
                );
                initializers.insert(key, field.data);
            }
            11 => inputs.push(name(&fields(field.data)?, 1)?),
            12 => outputs.push(name(&fields(field.data)?, 1)?),
            _ => {}
        }
    }
    ensure!(
        inputs == ["input"] && outputs == ["output"],
        "unsupported RIFE signature"
    );
    let cut = cut(&producers, &nodes, dtype)?;
    let mut roots = cut.flows.to_vec();
    roots.push(cut.mask);
    let (trunk_nodes, trunk_weights) =
        dependencies(&nodes, &producers, &initializers, &roots, &["input"])?;
    let mut head_boundaries = roots;
    head_boundaries.push("input");
    let (head_nodes, head_weights) = dependencies(
        &nodes,
        &producers,
        &initializers,
        &["output"],
        &head_boundaries,
    )?;
    ensure!(
        nodes
            .iter()
            .zip(&head_nodes)
            .any(|(node, used)| *used && node.op == "GridSample"),
        "RIFE head has no warp"
    );
    ensure!(
        !nodes
            .iter()
            .zip(&head_nodes)
            .any(|(node, used)| *used
                && matches!(node.op, "Conv" | "ConvTranspose" | "Gemm" | "MatMul")),
        "RIFE head still estimates motion"
    );

    let (h, w) = (h as u64, w as u64);
    let input_shape = [1, 7, h, w];
    // Final flow and mask still live at the export's internally padded extent.
    // The RGB input and cropped head output stay at the requested resolution.
    let aligned = |n: u64| n.div_ceil(cut.motion_alignment) * cut.motion_alignment;
    let state_shape = [1, 5, aligned(h), aligned(w)];
    let mut trunk = Graph::default();
    trunk.value(11, "input", dtype, &input_shape);
    trunk.value(12, "state", dtype, &state_shape);
    let rgb = trunk.slice("input", 1, 0, 6);
    let mid_raw = if fp16 {
        &0x3800u16.to_le_bytes()[..]
    } else {
        &0.5f32.to_le_bytes()[..]
    };
    let mid = trunk.tensor(&[], dtype, mid_raw);
    let phase_shape = trunk.ints(&[1, 1, h as i64, w as i64]);
    let half = trunk.node("Expand", &[&mid, &phase_shape], None);
    let fixed = trunk.node("Concat", &[&rgb, &half], Some(("axis", &[1], false)));
    for (node, used) in nodes.iter().zip(trunk_nodes) {
        if used {
            node.append(&mut trunk, Some(&fixed))?;
        }
    }
    for &weight in &trunk_weights {
        message(&mut trunk.bytes, 5, initializers[weight]);
    }
    let mut flows = cut
        .flows
        .iter()
        .map(|flow| {
            let planes = if cut.nchw {
                flow.to_string()
            } else {
                trunk.transpose(flow, &[0, 3, 1, 2])
            };
            trunk.cast(&planes, dtype)
        })
        .collect::<Vec<_>>();
    if let [batched] = flows.as_slice() {
        flows = vec![trunk.slice(batched, 0, 0, 1), trunk.slice(batched, 0, 1, 2)];
    }
    let mask = trunk.cast(cut.mask, dtype);
    trunk.emit(
        "state",
        "Concat",
        &[&flows[0], &flows[1], &mask],
        Some(("axis", &[1], false)),
    );

    let mut head = Graph::default();
    head.value(11, "input", dtype, &input_shape);
    head.value(11, "state", dtype, &state_shape);
    head.value(12, "output", dtype, &[1, 3, h, w]);
    let t_plane = head.slice("input", 1, 6, 7);
    let t_row = head.slice(&t_plane, 2, 0, 1);
    let t_pixel = head.slice(&t_row, 3, 0, 1);
    let t = head.cast(&t_pixel, FLOAT);
    let one = head.scalar(1.0);
    let two = head.scalar(2.0);
    let remaining = head.node("Sub", &[&one, &t], None);
    let scale0 = head.node("Mul", &[&two, &t], None);
    let scale1 = head.node("Mul", &[&two, &remaining], None);
    let mut flows = Vec::new();
    for (begin, scale) in [(0, scale0), (2, scale1)] {
        let flow = head.slice("state", 1, begin, begin + 2);
        let flow = head.cast(&flow, FLOAT);
        let scaled = head.node("Mul", &[&flow, &scale], None);
        flows.push(if cut.nchw {
            scaled
        } else {
            head.transpose(&scaled, &[0, 2, 3, 1])
        });
    }
    if cut.flows.len() == 1 {
        let combined = head.node(
            "Concat",
            &[&flows[0], &flows[1]],
            Some(("axis", &[0], false)),
        );
        flows = vec![combined];
    }
    for (&name, flow) in cut.flows.iter().zip(&flows) {
        head.emit(name, "Cast", &[flow], Some(("to", &[cut.flow_type], false)));
    }
    let mask = head.slice("state", 1, 4, 5);
    let mask = head.cast(&mask, FLOAT);
    let inv_mask = head.node("Sub", &[&one, &mask], None);
    let weight0 = head.node("Mul", &[&mask, &remaining], None);
    let weight1 = head.node("Mul", &[&inv_mask, &t], None);
    let sum = head.node("Add", &[&weight0, &weight1], None);
    let adjusted = head.node("Div", &[&weight0, &sum], None);
    head.emit(
        cut.mask,
        "Cast",
        &[&adjusted],
        Some(("to", &[dtype], false)),
    );
    for (node, used) in nodes.iter().zip(head_nodes) {
        if used {
            node.append(&mut head, None)?;
        }
    }
    for &weight in &head_weights {
        message(&mut head.bytes, 5, initializers[weight]);
    }

    Ok((
        assemble(&model, &trunk, b"rife_boost_trunk"),
        assemble(&model, &head, b"rife_boost_head"),
    ))
}
/// v983: true when the boost head of this model is exactly the plain Lite RIFE
/// final stage ("diagparallel" layout): flow x2t / x2(1-t), the phase-adjusted
/// mask, one batched GridSample(bilinear, align_corners=1, border) of frames
/// 0/1 and the mask blend. NeoArc can then run the head as one OpenCL kernel
/// (`neoarc_boost_head_*`) instead of ~90 small graph operators. The caller
/// still proves numerical agreement with the ONNX head before using it.
pub(super) fn native_head_compatible(bytes: &[u8], fp16: bool) -> bool {
    native_head_check(bytes, fp16).unwrap_or(false)
}
fn native_head_check(bytes: &[u8], fp16: bool) -> Result<bool> {
    let dtype = if fp16 { HALF } else { FLOAT };
    let model = fields(bytes)?;
    let graph = fields(
        model.iter().find(|field| field.id == 7).ok_or_else(|| anyhow!("ONNX graph missing"))?.data,
    )?;
    let (mut nodes, mut producers, mut initializers) = (Vec::new(), HashMap::new(), HashMap::new());
    for field in &graph {
        match field.id {
            1 => {
                let node = Node::parse(field.data)?;
                for &output in &node.outputs {
                    producers.insert(output, nodes.len());
                }
                nodes.push(node);
            }
            5 => {
                let initializer = fields(field.data)?;
                initializers.insert(name(&initializer, 8)?, field.data);
            }
            _ => {}
        }
    }
    let cut = cut(&producers, &nodes, dtype)?;
    if cut.flows != ["diagparallel/flow_grid"] || cut.mask != "diagparallel/mask_full" {
        return Ok(false);
    }
    let mut boundaries = cut.flows.to_vec();
    boundaries.push(cut.mask);
    boundaries.push("input");
    let (head_nodes, _) = dependencies(&nodes, &producers, &initializers, &["output"], &boundaries)?;
    const ALLOWED: &[&str] = &[
        "Add", "Cast", "Concat", "Constant", "Div", "Expand", "Gather", "GridSample", "Mul",
        "Range", "Reshape", "Shape", "Slice", "Split", "Sub", "Transpose", "Unsqueeze",
    ];
    let mut samples = 0;
    for (node, used) in nodes.iter().zip(&head_nodes) {
        if !*used {
            continue;
        }
        if !ALLOWED.contains(&node.op) {
            return Ok(false);
        }
        if node.op == "GridSample" {
            samples += 1;
            let (mut align, mut mode, mut padding) = (None, None, None);
            for part in fields(node.raw)?.into_iter().filter(|f| f.id == 5) {
                let attr = fields(part.data)?;
                let key = name(&attr, 1)?;
                let int = attr.iter().find(|f| f.id == 3).map(|f| -> Result<u64> {
                    let mut offset = 0;
                    onnx_wire_varint(f.raw, &mut offset)?;
                    onnx_wire_varint(f.raw, &mut offset)
                });
                let text = attr.iter().find(|f| f.id == 4).map(|f| f.data);
                match key {
                    "align_corners" => align = int.transpose()?,
                    "mode" => mode = text,
                    "padding_mode" => padding = text,
                    _ => {}
                }
            }
            if align != Some(1) || mode != Some(&b"bilinear"[..]) || padding != Some(&b"border"[..]) {
                return Ok(false);
            }
        }
    }
    // The graph output must be the two-term mask blend.
    let blend = producers.get("output").map(|&index| &nodes[index]);
    Ok(samples == 1
        && blend.is_some_and(|node| node.op == "Add"
            && node.inputs == ["diagparallel/blend0", "diagparallel/blend1"]))
}

// ---------------------------------------------------------------------------
// v971: DirectML/Arc ConvTranspose rewrite.
//
// On Arc B570 DirectML runs ConvTranspose(4x4, stride 2, pad 1) at roughly one
// tenth of the throughput of an ordinary 3x3 Conv (32->52 at 320x192: 3.24 ms,
// against 1.11 ms for the equivalent Conv). The transposed convolution equals
// a 3x3 Conv with four times the output channels followed by DepthToSpace(2,
// CRD). When the ConvTranspose already feeds a lone DepthToSpace(2, CRD) — the
// RIFE `lastconv` pattern — both pixel shuffles merge into one
// DepthToSpace(4, CRD). The weights are only rearranged (raw bytes, zeros for
// unused taps), so FP16 and FP32 models keep their exact coefficients.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Wire<'a> {
    Int(u64),
    Bytes(&'a [u8]),
}

fn wire_fields(bytes: &[u8]) -> Result<Vec<(u64, Wire<'_>)>> {
    let mut offset = 0;
    let mut result = Vec::new();
    while offset < bytes.len() {
        let key = onnx_wire_varint(bytes, &mut offset)?;
        let value = match key & 7 {
            0 => Wire::Int(onnx_wire_varint(bytes, &mut offset)?),
            1 | 2 | 5 => {
                let length = match key & 7 {
                    1 => 8,
                    5 => 4,
                    _ => usize::try_from(onnx_wire_varint(bytes, &mut offset)?)?,
                };
                let end = offset.checked_add(length).ok_or_else(|| anyhow!("ONNX length overflow"))?;
                let data = bytes.get(offset..end).ok_or_else(|| anyhow!("truncated ONNX field"))?;
                offset = end;
                Wire::Bytes(data)
            }
            _ => anyhow::bail!("unsupported ONNX wire type"),
        };
        result.push((key >> 3, value));
    }
    Ok(result)
}

/// Repeated int64 values of one field id, packed or unpacked.
fn repeated_ints(items: &[(u64, Wire<'_>)], id: u64) -> Result<Vec<i64>> {
    let mut values = Vec::new();
    for (field, value) in items {
        if *field != id {
            continue;
        }
        match value {
            Wire::Int(v) => values.push(*v as i64),
            Wire::Bytes(packed) => {
                let mut offset = 0;
                while offset < packed.len() {
                    values.push(onnx_wire_varint(packed, &mut offset)? as i64);
                }
            }
        }
    }
    Ok(values)
}

struct Attr<'a> {
    i: Option<i64>,
    s: Option<&'a [u8]>,
    ints: Vec<i64>,
}

fn node_attrs<'a>(node: &[(u64, Wire<'a>)]) -> Result<HashMap<&'a str, Attr<'a>>> {
    let mut attrs = HashMap::new();
    for (field, value) in node {
        let (5, Wire::Bytes(raw)) = (*field, *value) else { continue };
        let items = wire_fields(raw)?;
        let mut name = None;
        let mut attr = Attr { i: None, s: None, ints: repeated_ints(&items, 8)? };
        for (id, value) in &items {
            match (*id, *value) {
                (1, Wire::Bytes(n)) => name = Some(std::str::from_utf8(n)?),
                (3, Wire::Int(v)) => attr.i = Some(v as i64),
                (4, Wire::Bytes(s)) => attr.s = Some(s),
                _ => {}
            }
        }
        if let Some(name) = name {
            attrs.insert(name, attr);
        }
    }
    Ok(attrs)
}

fn strings<'a>(items: &[(u64, Wire<'a>)], id: u64) -> Result<Vec<&'a str>> {
    items
        .iter()
        .filter(|(field, _)| *field == id)
        .map(|(_, value)| match value {
            Wire::Bytes(b) => Ok(std::str::from_utf8(b)?),
            Wire::Int(_) => Err(anyhow!("ONNX string field {id} is not length-delimited")),
        })
        .collect()
}

struct Weight<'a> {
    dims: Vec<i64>,
    dtype: u64,
    raw: &'a [u8],
}

fn parse_weight<'a>(tensor: &'a [u8]) -> Result<(String, Option<Weight<'a>>)> {
    let items = wire_fields(tensor)?;
    let name = strings(&items, 8)?.first().copied().unwrap_or_default().to_string();
    let dtype = items.iter().find_map(|(id, v)| match (*id, *v) { (2, Wire::Int(t)) => Some(t), _ => None });
    let raw = items.iter().find_map(|(id, v)| match (*id, *v) { (9, Wire::Bytes(r)) => Some(r), _ => None });
    let external = items.iter().any(|(id, _)| *id == 13 || *id == 14);
    let weight = match (dtype, raw) {
        (Some(dtype @ (FLOAT | HALF)), Some(raw)) if !external => {
            Some(Weight { dims: repeated_ints(&items, 1)?, dtype, raw })
        }
        _ => None,
    };
    Ok((name, weight))
}

/// Channel dimension of the graph input named "input", when it is fixed.
pub(super) fn input_channels(bytes: &[u8]) -> Result<Option<i64>> {
    let model = fields(bytes)?;
    let graph = model.iter().find(|f| f.id == 7).ok_or_else(|| anyhow!("ONNX graph missing"))?;
    for field in fields(graph.data)?.iter().filter(|f| f.id == 11) {
        let info = wire_fields(field.data)?;
        if strings(&info, 1)?.first().copied() != Some("input") {
            continue;
        }
        let ty = info.iter().find_map(|(id, v)| match (*id, *v) { (2, Wire::Bytes(b)) => Some(b), _ => None });
        let Some(ty) = ty else { return Ok(None) };
        let tensor = wire_fields(ty)?.into_iter().find_map(|(id, v)| match (id, v) { (1, Wire::Bytes(b)) => Some(b), _ => None });
        let Some(tensor) = tensor else { return Ok(None) };
        let shape = wire_fields(tensor)?.into_iter().find_map(|(id, v)| match (id, v) { (2, Wire::Bytes(b)) => Some(b), _ => None });
        let Some(shape) = shape else { return Ok(None) };
        let dims: Vec<_> = wire_fields(shape)?.into_iter().filter(|(id, _)| *id == 1).collect();
        let Some((_, Wire::Bytes(dim))) = dims.get(1) else { return Ok(None) };
        return Ok(wire_fields(dim)?.into_iter().find_map(|(id, v)| match (id, v) { (1, Wire::Int(n)) => Some(n as i64), _ => None }));
    }
    Ok(None)
}

/// Returns the rewritten model and the number of ConvTranspose nodes replaced.
pub(super) fn dml_convtranspose_rewrite(bytes: &[u8]) -> Result<(Vec<u8>, usize)> {
    // Output phase a (0/1) and 3x3 tap d -> ConvTranspose kernel index.
    const KMAP: [[Option<usize>; 3]; 2] = [[Some(3), Some(1), None], [None, Some(2), Some(0)]];
    let model = fields(bytes)?;
    let graph = model.iter().find(|f| f.id == 7).ok_or_else(|| anyhow!("ONNX graph missing"))?;
    let graph_fields = fields(graph.data)?;
    let mut weights = HashMap::new();
    let mut parsed = Vec::new();
    let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut graph_outputs = HashSet::new();
    for field in &graph_fields {
        match field.id {
            1 => {
                let items = wire_fields(field.data)?;
                let index = parsed.len();
                for input in strings(&items, 1)? {
                    consumers.entry(input).or_default().push(index);
                }
                parsed.push(items);
            }
            5 => {
                let (name, weight) = parse_weight(field.data)?;
                if let Some(weight) = weight {
                    weights.insert(name, weight);
                }
            }
            12 => {
                graph_outputs.insert(name(&fields(field.data)?, 1)?.to_string());
            }
            _ => {}
        }
    }
    let op_of = |items: &[(u64, Wire<'_>)]| -> Result<String> {
        Ok(strings(items, 4)?.first().copied().unwrap_or_default().to_string())
    };
    // node index -> replacement node bytes (empty = drop), new initializers.
    let mut replace: HashMap<usize, Vec<u8>> = HashMap::new();
    let mut new_inits = Vec::new();
    let mut count = 0usize;
    for (index, items) in parsed.iter().enumerate() {
        if op_of(items)? != "ConvTranspose" || !strings(items, 7)?.iter().all(|d| d.is_empty()) {
            continue;
        }
        let attrs = node_attrs(items)?;
        let ints = |k: &str| attrs.get(k).map(|a| a.ints.clone());
        let auto_pad_ok = attrs.get("auto_pad").and_then(|a| a.s).is_none_or(|s| s == b"NOTSET");
        if ints("kernel_shape").as_deref() != Some(&[4, 4][..])
            || ints("strides").as_deref() != Some(&[2, 2][..])
            || ints("pads").as_deref() != Some(&[1, 1, 1, 1][..])
            || attrs.get("group").and_then(|a| a.i).is_some_and(|g| g != 1)
            || ints("dilations").is_some_and(|d| d != [1, 1])
            || attrs.get("output_padding").is_some_and(|a| a.ints.iter().any(|&v| v != 0))
            || attrs.contains_key("output_shape")
            || !auto_pad_ok
        {
            continue;
        }
        let inputs = strings(items, 1)?;
        let outputs = strings(items, 2)?;
        let (Some(&x), Some(&wname), Some(&out)) = (inputs.first(), inputs.get(1), outputs.first()) else { continue };
        let Some(weight) = weights.get(wname) else { continue };
        if weight.dims.len() != 4 || weight.dims[2] != 4 || weight.dims[3] != 4 {
            continue;
        }
        let (cin, cout) = (weight.dims[0] as usize, weight.dims[1] as usize);
        let elem = if weight.dtype == HALF { 2 } else { 4 };
        if weight.raw.len() != cin * cout * 16 * elem {
            continue;
        }
        let bias = match inputs.get(2).filter(|b| !b.is_empty()) {
            Some(b) => match weights.get(*b) {
                Some(bw) if bw.dtype == weight.dtype && bw.raw.len() == cout * elem => Some(bw),
                _ => continue,
            },
            None => None,
        };
        // Merge with a sole following DepthToSpace(2, CRD).
        let follower = consumers.get(out).filter(|c| c.len() == 1).map(|c| c[0]);
        let merge = match follower {
            Some(f) if !graph_outputs.contains(out) && cout % 4 == 0 && op_of(&parsed[f])? == "DepthToSpace" => {
                let a = node_attrs(&parsed[f])?;
                a.get("blocksize").and_then(|v| v.i) == Some(2)
                    && a.get("mode").and_then(|v| v.s) == Some(b"CRD")
                    && strings(&parsed[f], 1)?.len() == 1
            }
            _ => false,
        };
        let qcount = cout * 4;
        let mut kernel = vec![0u8; qcount * cin * 9 * elem];
        let mut bias_out = bias.map(|_| vec![0u8; qcount * elem]);
        let w_at = |c: usize, o: usize, ky: usize, kx: usize| ((c * cout + o) * 16 + ky * 4 + kx) * elem;
        let mut place = |q: usize, o: usize, a: usize, b: usize| {
            for c in 0..cin {
                for dy in 0..3 {
                    for dx in 0..3 {
                        if let (Some(ky), Some(kx)) = (KMAP[a][dy], KMAP[b][dx]) {
                            let src = w_at(c, o, ky, kx);
                            let dst = ((q * cin + c) * 9 + dy * 3 + dx) * elem;
                            kernel[dst..dst + elem].copy_from_slice(&weight.raw[src..src + elem]);
                        }
                    }
                }
            }
            if let (Some(out_bias), Some(bw)) = (bias_out.as_mut(), bias) {
                out_bias[q * elem..(q + 1) * elem].copy_from_slice(&bw.raw[o * elem..(o + 1) * elem]);
            }
        };
        let (block, final_out) = if merge {
            let c2 = cout / 4;
            for c in 0..c2 {
                for big_a in 0..2 {
                    for big_b in 0..2 {
                        let o = c * 4 + big_a * 2 + big_b;
                        for a in 0..2 {
                            for b in 0..2 {
                                let q = c * 16 + (2 * a + big_a) * 4 + (2 * b + big_b);
                                place(q, o, a, b);
                            }
                        }
                    }
                }
            }
            let f = follower.unwrap();
            (4u64, strings(&parsed[f], 2)?.first().copied().ok_or_else(|| anyhow!("DepthToSpace output missing"))?.to_string())
        } else {
            for o in 0..cout {
                for a in 0..2 {
                    for b in 0..2 {
                        place(o * 4 + a * 2 + b, o, a, b);
                    }
                }
            }
            (2u64, out.to_string())
        };
        let tag = format!("neo_dml_ct/{count}");
        let kname = format!("{tag}/k");
        let mid = format!("{tag}/sub");
        let mut tensor = Vec::new();
        for dim in [qcount, cin, 3, 3] {
            integer(&mut tensor, 1, dim as u64);
        }
        integer(&mut tensor, 2, weight.dtype);
        message(&mut tensor, 8, kname.as_bytes());
        message(&mut tensor, 9, &kernel);
        new_inits.push(tensor);
        let mut conv_inputs = vec![x.to_string(), kname];
        if let Some(raw) = bias_out {
            let bname = format!("{tag}/b");
            let mut tensor = Vec::new();
            integer(&mut tensor, 1, qcount as u64);
            integer(&mut tensor, 2, weight.dtype);
            message(&mut tensor, 8, bname.as_bytes());
            message(&mut tensor, 9, &raw);
            new_inits.push(tensor);
            conv_inputs.push(bname);
        }
        let ints_attr = |key: &str, values: &[u64]| {
            let mut a = Vec::new();
            message(&mut a, 1, key.as_bytes());
            integer(&mut a, 20, 7);
            for v in values {
                integer(&mut a, 8, *v);
            }
            a
        };
        let mut nodes = Vec::new();
        let mut conv = Vec::new();
        for input in &conv_inputs {
            message(&mut conv, 1, input.as_bytes());
        }
        message(&mut conv, 2, mid.as_bytes());
        message(&mut conv, 3, format!("{tag}/conv").as_bytes());
        message(&mut conv, 4, b"Conv");
        message(&mut conv, 5, &ints_attr("kernel_shape", &[3, 3]));
        message(&mut conv, 5, &ints_attr("pads", &[1, 1, 1, 1]));
        message(&mut conv, 5, &ints_attr("strides", &[1, 1]));
        message(&mut nodes, 1, &conv);
        let mut d2s = Vec::new();
        message(&mut d2s, 1, mid.as_bytes());
        message(&mut d2s, 2, final_out.as_bytes());
        message(&mut d2s, 3, format!("{tag}/d2s").as_bytes());
        message(&mut d2s, 4, b"DepthToSpace");
        let mut a = Vec::new();
        message(&mut a, 1, b"blocksize");
        integer(&mut a, 20, 2);
        integer(&mut a, 3, block);
        message(&mut d2s, 5, &a);
        let mut a = Vec::new();
        message(&mut a, 1, b"mode");
        integer(&mut a, 20, 3);
        message(&mut a, 4, b"CRD");
        message(&mut d2s, 5, &a);
        message(&mut nodes, 1, &d2s);
        replace.insert(index, nodes);
        if merge {
            replace.insert(follower.unwrap(), Vec::new());
        }
        count += 1;
    }
    if count == 0 {
        return Ok((bytes.to_vec(), 0));
    }
    let mut rewritten = Graph::default();
    let mut node_index = 0usize;
    for field in &graph_fields {
        if field.id == 1 {
            match replace.get(&node_index) {
                Some(nodes) => rewritten.bytes.extend_from_slice(nodes),
                None => rewritten.bytes.extend_from_slice(field.raw),
            }
            node_index += 1;
        } else {
            rewritten.bytes.extend_from_slice(field.raw);
        }
    }
    for tensor in &new_inits {
        message(&mut rewritten.bytes, 5, tensor);
    }
    let mut result = Vec::with_capacity(rewritten.bytes.len() + 64);
    for field in &model {
        if field.id == 7 {
            message(&mut result, 7, &rewritten.bytes);
        } else {
            result.extend_from_slice(field.raw);
        }
    }
    Ok((result, count))
}

/// v971: drop exact no-ops that DirectML still executes as full passes:
/// Resize whose constant scales are all 1.0 (RIFE's last block resizes 24, 4
/// and 13 channels at full resolution by 1.0) and Mul/Div by a scalar 1.0.
/// Consumers are rewired to the operand. Graph outputs are never removed.
pub(super) fn dml_identity_rewrite(bytes: &[u8]) -> Result<(Vec<u8>, usize)> {
    fn scalar_values(tensor: &[u8]) -> Result<Option<Vec<f32>>> {
        let items = wire_fields(tensor)?;
        let dtype = items.iter().find_map(|(id, v)| match (*id, *v) { (2, Wire::Int(t)) => Some(t), _ => None });
        if let Some((_, Wire::Bytes(raw))) = items.iter().find(|(id, _)| *id == 9) {
            return Ok(match dtype {
                Some(FLOAT) => Some(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                Some(HALF) => Some(raw.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()),
                _ => None,
            });
        }
        if dtype == Some(FLOAT) {
            let mut values = Vec::new();
            for (id, v) in &items {
                match (*id, *v) {
                    (4, Wire::Bytes(packed)) => values.extend(packed.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
                    (4, Wire::Int(_)) => return Ok(None),
                    _ => {}
                }
            }
            return Ok(Some(values));
        }
        Ok(None)
    }
    let model = fields(bytes)?;
    let graph = model.iter().find(|f| f.id == 7).ok_or_else(|| anyhow!("ONNX graph missing"))?;
    let graph_fields = fields(graph.data)?;
    let mut consts: HashMap<String, Vec<f32>> = HashMap::new();
    let mut graph_outputs = HashSet::new();
    for field in &graph_fields {
        match field.id {
            5 => {
                let items = wire_fields(field.data)?;
                let name = strings(&items, 8)?.first().copied().unwrap_or_default().to_string();
                if let Some(values) = scalar_values(field.data)? {
                    consts.insert(name, values);
                }
            }
            12 => {
                graph_outputs.insert(name(&fields(field.data)?, 1)?.to_string());
            }
            1 => {
                let items = wire_fields(field.data)?;
                if strings(&items, 4)?.first().copied() == Some("Constant") {
                    for (id, v) in &items {
                        let (5, Wire::Bytes(attr)) = (*id, *v) else { continue };
                        let attr_items = wire_fields(attr)?;
                        if strings(&attr_items, 1)?.first().copied() != Some("value") {
                            continue;
                        }
                        if let Some((_, Wire::Bytes(t))) = attr_items.iter().find(|(id, _)| *id == 5) {
                            if let (Some(out), Some(values)) = (strings(&items, 2)?.first(), scalar_values(t)?) {
                                consts.insert(out.to_string(), values);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let ones = |name: &str, scalar: bool| {
        consts.get(name).is_some_and(|v| !v.is_empty() && (!scalar || v.len() == 1) && v.iter().all(|&x| x == 1.0))
    };
    let mut alias: HashMap<String, String> = HashMap::new();
    let mut rewritten = Graph::default();
    let mut count = 0usize;
    for field in &graph_fields {
        if field.id != 1 {
            rewritten.bytes.extend_from_slice(field.raw);
            continue;
        }
        let items = wire_fields(field.data)?;
        let op = strings(&items, 4)?.first().copied().unwrap_or_default();
        let domain_default = strings(&items, 7)?.iter().all(|d| d.is_empty());
        let inputs: Vec<String> = strings(&items, 1)?.iter().map(|i| alias.get(*i).cloned().unwrap_or_else(|| i.to_string())).collect();
        let outputs = strings(&items, 2)?;
        let passthrough = if !domain_default || outputs.len() != 1 || graph_outputs.contains(outputs[0]) {
            None
        } else {
            match op {
                "Resize" if inputs.len() == 3 && !inputs[0].is_empty() && ones(&inputs[2], false) => Some(inputs[0].clone()),
                "Mul" | "Div" if inputs.len() == 2 && ones(&inputs[1], true) => Some(inputs[0].clone()),
                "Mul" if inputs.len() == 2 && ones(&inputs[0], true) => Some(inputs[1].clone()),
                _ => None,
            }
        };
        if let Some(source) = passthrough {
            alias.insert(outputs[0].to_string(), source);
            count += 1;
            continue;
        }
        // Re-emit the node with aliased input names.
        let mut node = Vec::new();
        let mut input_index = 0usize;
        for part in fields(field.data)? {
            if part.id == 1 {
                message(&mut node, 1, inputs[input_index].as_bytes());
                input_index += 1;
            } else {
                node.extend_from_slice(part.raw);
            }
        }
        message(&mut rewritten.bytes, 1, &node);
    }
    if count == 0 {
        return Ok((bytes.to_vec(), 0));
    }
    let mut result = Vec::with_capacity(rewritten.bytes.len() + 64);
    for field in &model {
        if field.id == 7 {
            message(&mut result, 7, &rewritten.bytes);
        } else {
            result.extend_from_slice(field.raw);
        }
    }
    Ok((result, count))
}

/// v972: WGL route input fusion. The route used a separate D3D12 compute pass
/// (own queue, CPU fence wait) to build the padded 7-channel FP16 input from
/// the two planar RGB history buffers and the phase. The same tensor is built
/// inside the graph instead: Concat(a, b) -> Pad(edge, bottom/right) ->
/// Concat(Expand(t)). Edge padding equals the pass's clamped addressing, so
/// the model sees identical values. Inputs: neo_wgl_a/neo_wgl_b [1,3,h,w]
/// FP16, neo_wgl_t [1,1,1,1] FP16; output declared [1,3,ph,pw] FP16.
pub(super) fn wgl_fused_input(bytes: &[u8], h: i64, w: i64, ph: i64, pw: i64) -> Result<Vec<u8>> {
    ensure!(h > 0 && w > 0 && ph >= h && pw >= w, "invalid WGL fused geometry");
    ensure!(input_is_half(bytes)?, "WGL fused input requires FP16 graph input");
    ensure!(input_channels(bytes)? == Some(7), "WGL fused input requires seven-channel RIFE");
    let model = fields(bytes)?;
    let graph = model.iter().find(|f| f.id == 7).ok_or_else(|| anyhow!("ONNX graph missing"))?;
    let original = fields(graph.data)?;
    let mut has_output = false;
    for field in original.iter().filter(|f| f.id == 12) {
        if name(&fields(field.data)?, 1)? == "output" {
            has_output = true;
        }
    }
    ensure!(has_output, "graph output 'output' missing");
    let mut g = Graph { bytes: Vec::new(), next: 2_000_000 };
    let dims = |v: &[i64]| v.iter().map(|&d| d as u64).collect::<Vec<_>>();
    g.value(11, "neo_wgl_a", HALF, &dims(&[1, 3, h, w]));
    g.value(11, "neo_wgl_b", HALF, &dims(&[1, 3, h, w]));
    g.value(11, "neo_wgl_t", HALF, &dims(&[1, 1, 1, 1]));
    let pair = g.node("Concat", &["neo_wgl_a", "neo_wgl_b"], Some(("axis", &[1], false)));
    let padded = if ph > h || pw > w {
        let pads = g.ints(&[0, 0, 0, 0, 0, 0, ph - h, pw - w]);
        let out = g.temp();
        let mut node = Vec::new();
        message(&mut node, 1, pair.as_bytes());
        message(&mut node, 1, pads.as_bytes());
        message(&mut node, 2, out.as_bytes());
        message(&mut node, 4, b"Pad");
        let mut mode = Vec::new();
        message(&mut mode, 1, b"mode");
        integer(&mut mode, 20, 3);
        message(&mut mode, 4, b"edge");
        message(&mut node, 5, &mode);
        message(&mut g.bytes, 1, &node);
        out
    } else {
        pair
    };
    let shape = g.ints(&[1, 1, ph, pw]);
    let time = g.node("Expand", &["neo_wgl_t", &shape], None);
    g.emit("input", "Concat", &[&padded, &time], Some(("axis", &[1], false)));
    for field in &original {
        match field.id {
            11 if name(&fields(field.data)?, 1)? == "input" => {}
            // Symbolic value_info would conflict with the fixed geometry.
            13 => {}
            12 if name(&fields(field.data)?, 1)? == "output" => {
                g.value(12, "output", HALF, &dims(&[1, 3, ph, pw]));
            }
            _ => g.bytes.extend_from_slice(field.raw),
        }
    }
    let mut result = Vec::with_capacity(g.bytes.len() + 64);
    for field in &model {
        if field.id == 7 {
            message(&mut result, 7, &g.bytes);
        } else {
            result.extend_from_slice(field.raw);
        }
    }
    Ok(result)
}

/// Compose the fixed-motion estimator and its per-phase heads into one ORT run.
/// Only the RGB endpoints and the small phase vector come from the host.
pub(super) fn fused_host_pair(bytes: &[u8], h: i64, w: i64, count: usize) -> Result<Vec<u8>> {
    ensure!((1..=4).contains(&count), "unsupported phase count");
    let (trunk, head) = split(bytes, h, w, false)?;
    let model = fields(&trunk)?;
    let graph_fields = |model: &[u8]| -> Result<Vec<u8>> {
        Ok(fields(model)?.into_iter().find(|f| f.id == 7)
            .ok_or_else(|| anyhow!("missing RIFE graph"))?.data.to_vec())
    };
    let trunk_graph = graph_fields(&trunk)?;
    let head_graph = graph_fields(&head)?;
    let mut combined = Graph::default();
    for f in fields(&trunk_graph)? {
        if matches!(f.id, 1 | 5 | 11) { combined.bytes.extend_from_slice(f.raw); }
    }
    // Use an independent namespace for the small phase-vector plumbing.
    combined.next = 1_000_000;
    combined.value(11, "phases", FLOAT, &[count as u64]);
    let rgb = combined.slice("input", 1, 0, 6);
    let plane_shape = combined.ints(&[1, 1, h, w]);
    for i in 0..count {
        let prefix = format!("neo_pair_head_{i}/");
        let phase = combined.slice("phases", 0, i as i64, i as i64 + 1);
        let plane = combined.node("Expand", &[&phase, &plane_shape], None);
        combined.emit(&format!("{prefix}input"), "Concat", &[&rgb, &plane], Some(("axis", &[1], false)));
        for f in fields(&head_graph)? {
            if !matches!(f.id, 1 | 5 | 12) { continue; }
            let mut renamed = Vec::new();
            for item in fields(f.data)? {
                let named = match f.id { 1 => matches!(item.id, 1..=3), 5 => item.id == 8, _ => item.id == 1 };
                if named && !item.data.is_empty() && !(f.id == 1 && item.id == 1 && item.data == b"state") {
                    let local = std::str::from_utf8(item.data)?;
                    message(&mut renamed,item.id,format!("{prefix}{local}").as_bytes());
                } else { renamed.extend_from_slice(item.raw); }
            }
            message(&mut combined.bytes,f.id,&renamed);
        }
    }
    Ok(assemble(&model,&combined,b"neo_dml_fused_rife_pair"))
}

#[cfg(test)]
mod native_head_tests {
    use super::*;
    fn model(name: &str) -> Option<Vec<u8>> {
        std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models/RIFE").join(name)).ok()
    }
    #[test]
    fn v983_lite_heads_are_native_candidates_and_full_models_are_not() {
        for (name, expected) in [
            ("NeoRIFE_v1_6_Lite50_fp16.onnx", true),
            ("NeoRIFE_v1_6_Lite75_fp16.onnx", true),
            ("NeoRIFE_v1_6_fp16.onnx", false),
            ("NeoRIFE_fp16.onnx", false),
        ] {
            let Some(bytes) = model(name) else { continue };
            let fp16 = input_is_half(&bytes).unwrap();
            assert_eq!(native_head_compatible(&bytes, fp16), expected, "{name}");
        }
    }
}