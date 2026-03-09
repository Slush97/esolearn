// SPDX-License-Identifier: MIT OR Apache-2.0
//! ONNX model export.
//!
//! Supports exporting trained models to ONNX format for cross-framework
//! inference with onnxruntime, TensorFlow, or PyTorch.
//!
//! Uses manual protobuf serialization — no external dependencies required.
//!
//! # Supported models
//!
//! - [`LinearRegression`](crate::linear::LinearRegression) — `MatMul + Add`
//! - [`LogisticRegression`](crate::linear::LogisticRegression) — `MatMul + Add + Sigmoid/Softmax`
//! - [`StandardScaler`](crate::preprocess::StandardScaler) — `Sub + Div`
//! - [`MLPClassifier`](crate::neural::MLPClassifier) — chain of `MatMul + Add + Activation`
//! - [`MLPRegressor`](crate::neural::MLPRegressor) — chain of `MatMul + Add + Activation`
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::prelude::*;
//! use scry_learn::onnx::ToOnnx;
//!
//! let mut model = LinearRegression::new();
//! model.fit(&data)?;
//! model.to_onnx("linear_model.onnx")?;
//! ```

// Protobuf dimensions require i64; wrapping is acceptable for feature counts.
#![allow(clippy::cast_possible_wrap)]

use std::path::Path;

use crate::error::{Result, ScryLearnError};
use crate::linear::{LinearRegression, LogisticRegression};
use crate::neural::{Activation, MLPClassifier, MLPRegressor};
use crate::preprocess::StandardScaler;
use crate::tree::{
    DecisionTreeClassifier, DecisionTreeRegressor, FlatTree, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    HistNodeView, RandomForestClassifier, RandomForestRegressor,
};

/// Trait for models that can be exported to ONNX format.
pub trait ToOnnx {
    /// Export this model to an ONNX file.
    fn to_onnx(&self, path: impl AsRef<Path>) -> Result<()> {
        let bytes = self.to_onnx_bytes()?;
        std::fs::write(path, bytes).map_err(ScryLearnError::Io)
    }

    /// Export this model to ONNX bytes (in-memory).
    fn to_onnx_bytes(&self) -> Result<Vec<u8>>;
}

// ── Protobuf encoding helpers ──────────────────────────────────────────────

/// Protobuf wire types.
const WIRE_VARINT: u32 = 0;
const WIRE_LEN: u32 = 2;

/// Encode a varint into a byte buffer.
fn encode_varint(buf: &mut Vec<u8>, mut val: u64) {
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

/// Encode a protobuf field tag.
fn encode_tag(buf: &mut Vec<u8>, field: u32, wire_type: u32) {
    encode_varint(buf, u64::from((field << 3) | wire_type));
}

/// Encode a varint field.
fn encode_varint_field(buf: &mut Vec<u8>, field: u32, val: u64) {
    if val != 0 {
        encode_tag(buf, field, WIRE_VARINT);
        encode_varint(buf, val);
    }
}

/// Encode a length-delimited bytes field.
fn encode_bytes_field(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
    encode_tag(buf, field, WIRE_LEN);
    encode_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

/// Encode a string field.
fn encode_string_field(buf: &mut Vec<u8>, field: u32, s: &str) {
    if !s.is_empty() {
        encode_bytes_field(buf, field, s.as_bytes());
    }
}

/// Encode a float field as a packed repeated float (wire type 2).
fn encode_packed_floats(buf: &mut Vec<u8>, field: u32, vals: &[f32]) {
    if vals.is_empty() {
        return;
    }
    encode_tag(buf, field, WIRE_LEN);
    encode_varint(buf, (vals.len() * 4) as u64);
    for &v in vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Encode a packed repeated int64 field.
fn encode_packed_i64s(buf: &mut Vec<u8>, field: u32, vals: &[i64]) {
    if vals.is_empty() {
        return;
    }
    let mut inner = Vec::new();
    for &v in vals {
        encode_varint(&mut inner, v as u64);
    }
    encode_bytes_field(buf, field, &inner);
}

// ── ONNX protobuf message builders ────────────────────────────────────────

/// ONNX IR version 8, opset 18.
const IR_VERSION: u64 = 8;
const OPSET_VERSION: u64 = 18;
const PRODUCER: &str = "scry-learn";

/// Build a complete ONNX ModelProto.
fn build_model(graph: &[u8], opset_domain: &str) -> Vec<u8> {
    let mut model = Vec::new();

    // field 1: ir_version (int64)
    encode_varint_field(&mut model, 1, IR_VERSION);

    // field 8: opset_import (repeated OpsetIdProto)
    // Standard opset
    {
        let mut opset = Vec::new();
        // field 1: domain (string) — empty string = standard
        // field 2: version (int64)
        encode_varint_field(&mut opset, 2, OPSET_VERSION);
        encode_bytes_field(&mut model, 8, &opset);
    }
    // ML opset (if needed)
    if !opset_domain.is_empty() {
        let mut opset = Vec::new();
        encode_string_field(&mut opset, 1, opset_domain);
        encode_varint_field(&mut opset, 2, 3); // ai.onnx.ml opset version 3
        encode_bytes_field(&mut model, 8, &opset);
    }

    // field 2: producer_name (string)
    encode_string_field(&mut model, 2, PRODUCER);

    // field 7: graph (GraphProto)
    encode_bytes_field(&mut model, 7, graph);

    model
}

/// Data types in ONNX TensorProto.
const ONNX_FLOAT: i64 = 1;

/// Build a TensorProto (used as an initializer).
fn build_tensor(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
    let mut t = Vec::new();
    // field 1: dims (repeated int64)
    encode_packed_i64s(&mut t, 1, dims);
    // field 2: data_type (int32, as varint)
    encode_varint_field(&mut t, 2, ONNX_FLOAT as u64);
    // field 4: float_data (repeated float, packed)
    encode_packed_floats(&mut t, 4, data);
    // field 8: name (string)
    encode_string_field(&mut t, 8, name);
    t
}

/// Build a ValueInfoProto for a tensor input/output.
fn build_value_info(name: &str, dims: &[i64]) -> Vec<u8> {
    // TypeProto.Tensor
    let mut tensor_type = Vec::new();
    // field 1: elem_type (int32)
    encode_varint_field(&mut tensor_type, 1, ONNX_FLOAT as u64);
    // field 2: shape (TensorShapeProto)
    {
        let mut shape = Vec::new();
        for &d in dims {
            let mut dim = Vec::new();
            if d < 0 {
                // Symbolic dim (use dim_param)
                encode_string_field(&mut dim, 2, "N");
            } else {
                // Fixed dim (dim_value)
                encode_varint_field(&mut dim, 1, d as u64);
            }
            encode_bytes_field(&mut shape, 1, &dim);
        }
        encode_bytes_field(&mut tensor_type, 2, &shape);
    }

    // TypeProto
    let mut type_proto = Vec::new();
    // field 1: tensor_type
    encode_bytes_field(&mut type_proto, 1, &tensor_type);

    // ValueInfoProto
    let mut vi = Vec::new();
    // field 1: name
    encode_string_field(&mut vi, 1, name);
    // field 2: type
    encode_bytes_field(&mut vi, 2, &type_proto);
    vi
}

/// Build a NodeProto.
fn build_node(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: &[(&str, AttrValue)],
    domain: &str,
) -> Vec<u8> {
    let mut node = Vec::new();
    // field 1: input (repeated string)
    for &inp in inputs {
        encode_string_field(&mut node, 1, inp);
    }
    // field 2: output (repeated string)
    for &out in outputs {
        encode_string_field(&mut node, 2, out);
    }
    // field 4: op_type (string)
    encode_string_field(&mut node, 4, op_type);
    // field 5: attribute (repeated AttributeProto)
    for (name, value) in attrs {
        let attr = build_attribute(name, value);
        encode_bytes_field(&mut node, 5, &attr);
    }
    // field 7: domain (string)
    encode_string_field(&mut node, 7, domain);
    node
}

/// ONNX attribute value types.
#[allow(dead_code)]
enum AttrValue<'a> {
    Int(i64),
    Ints(&'a [i64]),
    Float(f32),
    Floats(&'a [f32]),
    String(&'a str),
    Strings(&'a [&'a str]),
}

/// Build an AttributeProto.
fn build_attribute(name: &str, value: &AttrValue) -> Vec<u8> {
    let mut attr = Vec::new();
    // field 1: name
    encode_string_field(&mut attr, 1, name);
    match value {
        AttrValue::Int(v) => {
            // field 20: type = INT (2)
            encode_varint_field(&mut attr, 20, 2);
            // field 2: i (int64)
            encode_varint_field(&mut attr, 2, *v as u64);
        }
        AttrValue::Ints(vals) => {
            // field 20: type = INTS (7)
            encode_varint_field(&mut attr, 20, 7);
            // field 8: ints (repeated int64)
            for &v in *vals {
                encode_tag(&mut attr, 8, WIRE_VARINT);
                encode_varint(&mut attr, v as u64);
            }
        }
        AttrValue::Float(v) => {
            // field 20: type = FLOAT (1)
            encode_varint_field(&mut attr, 20, 1);
            // field 4: f (float32) — wire type 5 (32-bit)
            encode_tag(&mut attr, 4, 5);
            attr.extend_from_slice(&v.to_le_bytes());
        }
        AttrValue::Floats(vals) => {
            // field 20: type = FLOATS (6)
            encode_varint_field(&mut attr, 20, 6);
            // field 7: floats (repeated float)
            for &v in *vals {
                encode_tag(&mut attr, 7, 5);
                attr.extend_from_slice(&v.to_le_bytes());
            }
        }
        AttrValue::String(s) => {
            // field 20: type = STRING (3)
            encode_varint_field(&mut attr, 20, 3);
            // field 3: s (bytes)
            encode_bytes_field(&mut attr, 3, s.as_bytes());
        }
        AttrValue::Strings(vals) => {
            // field 20: type = STRINGS (8)
            encode_varint_field(&mut attr, 20, 8);
            // field 9: strings (repeated bytes)
            for s in *vals {
                encode_bytes_field(&mut attr, 9, s.as_bytes());
            }
        }
    }
    attr
}

/// Build a GraphProto.
fn build_graph(
    name: &str,
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
    nodes: &[Vec<u8>],
    initializers: &[Vec<u8>],
) -> Vec<u8> {
    let mut graph = Vec::new();
    // field 1: node (repeated NodeProto)
    for n in nodes {
        encode_bytes_field(&mut graph, 1, n);
    }
    // field 2: name
    encode_string_field(&mut graph, 2, name);
    // field 5: initializer (repeated TensorProto)
    for init in initializers {
        encode_bytes_field(&mut graph, 5, init);
    }
    // field 11: input (repeated ValueInfoProto)
    for inp in inputs {
        encode_bytes_field(&mut graph, 11, inp);
    }
    // field 12: output (repeated ValueInfoProto)
    for out in outputs {
        encode_bytes_field(&mut graph, 12, out);
    }
    graph
}

// ── ToOnnx implementations ────────────────────────────────────────────────

impl ToOnnx for LinearRegression {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let coeffs = self.coefficients();
        if coeffs.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        let n_features = coeffs.len();

        // Initializers: W [n_features, 1], b [1]
        let w_data: Vec<f32> = coeffs.iter().map(|&v| v as f32).collect();
        let b_data = [self.intercept() as f32];

        let w_tensor = build_tensor("W", &[n_features as i64, 1], &w_data);
        let b_tensor = build_tensor("b", &[1], &b_data);

        // Nodes: MatMul(X, W) -> tmp; Add(tmp, b) -> Y
        let matmul = build_node("MatMul", &["X", "W"], &["tmp"], &[], "");
        let add = build_node("Add", &["tmp", "b"], &["Y"], &[], "");

        let inputs = [build_value_info("X", &[-1, n_features as i64])];
        let outputs = [build_value_info("Y", &[-1, 1])];

        let graph = build_graph(
            "linear_regression",
            &inputs,
            &outputs,
            &[matmul, add],
            &[w_tensor, b_tensor],
        );
        Ok(build_model(&graph, ""))
    }
}

impl ToOnnx for LogisticRegression {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let weights = self.weights();
        if weights.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        let n_classes = weights.len();
        // Each weight vector is [bias, w1, w2, ..., wn]
        let n_features = weights[0].len() - 1;

        if n_classes <= 2 {
            // Binary classification: single output + Sigmoid
            let w_data: Vec<f32> = weights[0][1..].iter().map(|&v| v as f32).collect();
            let b_data = [weights[0][0] as f32];

            let w_tensor = build_tensor("W", &[n_features as i64, 1], &w_data);
            let b_tensor = build_tensor("b", &[1], &b_data);

            let matmul = build_node("MatMul", &["X", "W"], &["tmp"], &[], "");
            let add = build_node("Add", &["tmp", "b"], &["logits"], &[], "");
            let sigmoid = build_node("Sigmoid", &["logits"], &["Y"], &[], "");

            let inputs = [build_value_info("X", &[-1, n_features as i64])];
            let outputs = [build_value_info("Y", &[-1, 1])];

            let graph = build_graph(
                "logistic_regression",
                &inputs,
                &outputs,
                &[matmul, add, sigmoid],
                &[w_tensor, b_tensor],
            );
            Ok(build_model(&graph, ""))
        } else {
            // Multiclass: [N, n_features] @ [n_features, n_classes] + [n_classes] -> Softmax
            // Weight matrix: column j = coefficients for class j (excluding bias)
            let mut w_data = Vec::with_capacity(n_features * n_classes);
            let b_data: Vec<f32> = weights.iter().map(|w| w[0] as f32).collect();
            // Row-major: for each feature i, for each class j
            for i in 0..n_features {
                for w in weights {
                    w_data.push(w[i + 1] as f32);
                }
            }

            let w_tensor = build_tensor("W", &[n_features as i64, n_classes as i64], &w_data);
            let b_tensor = build_tensor("b", &[n_classes as i64], &b_data);

            let matmul = build_node("MatMul", &["X", "W"], &["tmp"], &[], "");
            let add = build_node("Add", &["tmp", "b"], &["logits"], &[], "");
            let softmax = build_node(
                "Softmax",
                &["logits"],
                &["Y"],
                &[("axis", AttrValue::Int(1))],
                "",
            );

            let inputs = [build_value_info("X", &[-1, n_features as i64])];
            let outputs = [build_value_info("Y", &[-1, n_classes as i64])];

            let graph = build_graph(
                "logistic_regression",
                &inputs,
                &outputs,
                &[matmul, add, softmax],
                &[w_tensor, b_tensor],
            );
            Ok(build_model(&graph, ""))
        }
    }
}

impl ToOnnx for StandardScaler {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(ScryLearnError::NotFitted);
        }
        let means = self.means();
        let stds = self.stds();
        let n_features = means.len();

        // Initializers: mean [1, n_features], std [1, n_features]
        let mean_data: Vec<f32> = means.iter().map(|&v| v as f32).collect();
        let std_data: Vec<f32> = stds
            .iter()
            .map(|&v| if v.abs() < 1e-12 { 1.0f32 } else { v as f32 })
            .collect();

        let mean_tensor = build_tensor("mean", &[1, n_features as i64], &mean_data);
        let std_tensor = build_tensor("scale", &[1, n_features as i64], &std_data);

        // Nodes: Sub(X, mean) -> centered; Div(centered, scale) -> Y
        let sub = build_node("Sub", &["X", "mean"], &["centered"], &[], "");
        let div = build_node("Div", &["centered", "scale"], &["Y"], &[], "");

        let inputs = [build_value_info("X", &[-1, n_features as i64])];
        let outputs = [build_value_info("Y", &[-1, n_features as i64])];

        let graph = build_graph(
            "standard_scaler",
            &inputs,
            &outputs,
            &[sub, div],
            &[mean_tensor, std_tensor],
        );
        Ok(build_model(&graph, ""))
    }
}

/// Map a scry-learn Activation to an ONNX op name.
fn activation_op(act: Activation) -> &'static str {
    match act {
        Activation::Relu => "Relu",
        Activation::Sigmoid => "Sigmoid",
        Activation::Tanh => "Tanh",
        // Identity or any future variant: pass-through
        _ => "Identity",
    }
}

/// Build ONNX graph for an MLP (shared logic for classifier and regressor).
fn build_mlp_onnx(
    graph_name: &str,
    n_features: usize,
    layer_weights: &[(Vec<f64>, Vec<f64>)],
    layer_dims: &[(usize, usize)],
    hidden_activation: Activation,
    output_activation: &str,
    output_dim: usize,
) -> Result<Vec<u8>> {
    if layer_weights.is_empty() {
        return Err(ScryLearnError::NotFitted);
    }
    let n_layers = layer_weights.len();

    let mut nodes = Vec::new();
    let mut initializers = Vec::new();

    let mut prev_output = "X".to_string();

    for (i, ((w_vec, b_vec), &(in_sz, out_sz))) in layer_weights.iter().zip(layer_dims).enumerate()
    {
        let w_name = format!("W{i}");
        let b_name = format!("b{i}");
        let mm_out = format!("mm{i}");
        let add_out = format!("add{i}");
        let act_out = format!("act{i}");

        // Weight tensor: [in_sz, out_sz], row-major
        let w_data: Vec<f32> = w_vec.iter().map(|&v| v as f32).collect();
        let b_data: Vec<f32> = b_vec.iter().map(|&v| v as f32).collect();

        initializers.push(build_tensor(
            &w_name,
            &[in_sz as i64, out_sz as i64],
            &w_data,
        ));
        initializers.push(build_tensor(&b_name, &[out_sz as i64], &b_data));

        nodes.push(build_node(
            "MatMul",
            &[&prev_output, &w_name],
            &[&mm_out],
            &[],
            "",
        ));
        nodes.push(build_node("Add", &[&mm_out, &b_name], &[&add_out], &[], ""));

        // Activation: hidden layers use hidden_activation, last layer uses output_activation
        let is_last = i == n_layers - 1;
        let op = if is_last {
            output_activation
        } else {
            activation_op(hidden_activation)
        };

        if op != "Identity" {
            let out = if is_last {
                "Y".to_string()
            } else {
                act_out.clone()
            };
            if op == "Softmax" {
                nodes.push(build_node(
                    op,
                    &[&add_out],
                    &[&out],
                    &[("axis", AttrValue::Int(1))],
                    "",
                ));
            } else {
                nodes.push(build_node(op, &[&add_out], &[&out], &[], ""));
            }
            prev_output = out;
        } else if is_last {
            // Identity on last layer — just rename
            nodes.push(build_node("Identity", &[&add_out], &["Y"], &[], ""));
            prev_output = "Y".to_string();
        } else {
            prev_output = add_out;
        }
    }

    let inputs = [build_value_info("X", &[-1, n_features as i64])];
    let outputs = [build_value_info("Y", &[-1, output_dim as i64])];

    let graph = build_graph(graph_name, &inputs, &outputs, &nodes, &initializers);
    Ok(build_model(&graph, ""))
}

impl ToOnnx for MLPClassifier {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let weights = self.weights();
        if weights.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        let n_classes = self.n_classes();
        let output_act = if n_classes <= 2 { "Sigmoid" } else { "Softmax" };
        let output_dim = if n_classes <= 2 { 1 } else { n_classes };

        build_mlp_onnx(
            "mlp_classifier",
            self.n_features(),
            weights,
            self.layer_dims(),
            self.activation_fn(),
            output_act,
            output_dim,
        )
    }
}

impl ToOnnx for MLPRegressor {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let weights = self.weights();
        if weights.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        build_mlp_onnx(
            "mlp_regressor",
            self.n_features(),
            weights,
            self.layer_dims(),
            self.activation_fn(),
            "Identity",
            1,
        )
    }
}

// ── ONNX ML tree ensemble helpers ────────────────────────────────────────

/// Sentinel value for leaf nodes in FlatTree (same as LEAF_SENTINEL in cart).
const FLAT_LEAF: u32 = u32::MAX;
const ML_DOMAIN: &str = "ai.onnx.ml";

/// Collected ONNX ML tree node arrays for one or more trees.
struct TreeEnsembleArrays {
    nodes_treeids: Vec<i64>,
    nodes_nodeids: Vec<i64>,
    nodes_featureids: Vec<i64>,
    nodes_values: Vec<f32>,
    nodes_modes: Vec<&'static str>,
    nodes_truenodeids: Vec<i64>,
    nodes_falsenodeids: Vec<i64>,
}

impl TreeEnsembleArrays {
    fn new() -> Self {
        Self {
            nodes_treeids: Vec::new(),
            nodes_nodeids: Vec::new(),
            nodes_featureids: Vec::new(),
            nodes_values: Vec::new(),
            nodes_modes: Vec::new(),
            nodes_truenodeids: Vec::new(),
            nodes_falsenodeids: Vec::new(),
        }
    }

    /// Append all nodes from a FlatTree, assigning them `tree_id`.
    fn append_flat_tree(&mut self, tree: &FlatTree, tree_id: i64) {
        for (idx, node) in tree.nodes.iter().enumerate() {
            self.nodes_treeids.push(tree_id);
            self.nodes_nodeids.push(idx as i64);
            if node.right == FLAT_LEAF {
                // Leaf node
                self.nodes_featureids.push(0);
                self.nodes_values.push(0.0);
                self.nodes_modes.push("LEAF");
                self.nodes_truenodeids.push(0);
                self.nodes_falsenodeids.push(0);
            } else {
                // Internal split node
                self.nodes_featureids.push(node.feature_idx as i64);
                self.nodes_values.push(node.threshold as f32);
                self.nodes_modes.push("BRANCH_LEQ");
                self.nodes_truenodeids.push((idx + 1) as i64); // left child
                self.nodes_falsenodeids.push(node.right as i64); // right child
            }
        }
    }

    /// Append all nodes from a HistTree (histogram-based), converting bin
    /// thresholds to raw thresholds using the binner's bin edges.
    fn append_hist_tree(
        &mut self,
        nodes: &[HistNodeView],
        tree_id: i64,
    ) {
        for (idx, node) in nodes.iter().enumerate() {
            self.nodes_treeids.push(tree_id);
            self.nodes_nodeids.push(idx as i64);
            match node {
                HistNodeView::Leaf { .. } => {
                    self.nodes_featureids.push(0);
                    self.nodes_values.push(0.0);
                    self.nodes_modes.push("LEAF");
                    self.nodes_truenodeids.push(0);
                    self.nodes_falsenodeids.push(0);
                }
                HistNodeView::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    self.nodes_featureids.push(*feature as i64);
                    self.nodes_values.push(*threshold as f32);
                    self.nodes_modes.push("BRANCH_LEQ");
                    self.nodes_truenodeids.push(*left as i64);
                    self.nodes_falsenodeids.push(*right as i64);
                }
            }
        }
    }
}

/// Leaf info for regressor tree ensembles.
struct RegressorLeaves {
    target_ids: Vec<i64>,
    target_nodeids: Vec<i64>,
    target_treeids: Vec<i64>,
    target_weights: Vec<f32>,
}

impl RegressorLeaves {
    fn new() -> Self {
        Self {
            target_ids: Vec::new(),
            target_nodeids: Vec::new(),
            target_treeids: Vec::new(),
            target_weights: Vec::new(),
        }
    }

    /// Extract leaf predictions from a FlatTree.
    fn append_flat_tree(&mut self, tree: &FlatTree, tree_id: i64, scale: f64) {
        for (idx, node) in tree.nodes.iter().enumerate() {
            if node.right == FLAT_LEAF {
                let li = node.feature_idx as usize;
                self.target_ids.push(0);
                self.target_nodeids.push(idx as i64);
                self.target_treeids.push(tree_id);
                self.target_weights.push((tree.predictions[li] * scale) as f32);
            }
        }
    }

    /// Extract leaf predictions from HistTree nodes.
    fn append_hist_leaves(
        &mut self,
        nodes: &[HistNodeView],
        tree_id: i64,
        scale: f64,
    ) {
        for (idx, node) in nodes.iter().enumerate() {
            if let HistNodeView::Leaf { value } = node {
                self.target_ids.push(0);
                self.target_nodeids.push(idx as i64);
                self.target_treeids.push(tree_id);
                self.target_weights.push((*value * scale) as f32);
            }
        }
    }
}

/// Leaf info for classifier tree ensembles.
struct ClassifierLeaves {
    class_ids: Vec<i64>,
    class_nodeids: Vec<i64>,
    class_treeids: Vec<i64>,
    class_weights: Vec<f32>,
}

impl ClassifierLeaves {
    fn new() -> Self {
        Self {
            class_ids: Vec::new(),
            class_nodeids: Vec::new(),
            class_treeids: Vec::new(),
            class_weights: Vec::new(),
        }
    }

    /// Extract leaf class probabilities from a FlatTree (decision tree classifier).
    fn append_flat_tree_proba(&mut self, tree: &FlatTree, tree_id: i64, n_classes: usize) {
        let nc = tree.n_classes_stored as usize;
        for (idx, node) in tree.nodes.iter().enumerate() {
            if node.right == FLAT_LEAF {
                let li = node.feature_idx as usize;
                let start = li * nc;
                for c in 0..n_classes {
                    self.class_ids.push(c as i64);
                    self.class_nodeids.push(idx as i64);
                    self.class_treeids.push(tree_id);
                    let w = if c < nc {
                        tree.leaf_probas[start + c]
                    } else {
                        0.0
                    };
                    self.class_weights.push(w);
                }
            }
        }
    }
}

/// Build ONNX bytes for a tree ensemble regressor.
fn build_tree_ensemble_regressor(
    graph_name: &str,
    n_features: usize,
    arrays: &TreeEnsembleArrays,
    leaves: &RegressorLeaves,
    base_values: &[f32],
    aggregate: &str,
) -> Result<Vec<u8>> {
    let modes_refs: Vec<&str> = arrays.nodes_modes.iter().copied().collect();
    let base_vals_owned: Vec<f32> = base_values.to_vec();

    let mut attrs: Vec<(&str, AttrValue)> = vec![
        ("nodes_treeids", AttrValue::Ints(&arrays.nodes_treeids)),
        ("nodes_nodeids", AttrValue::Ints(&arrays.nodes_nodeids)),
        ("nodes_featureids", AttrValue::Ints(&arrays.nodes_featureids)),
        ("nodes_values", AttrValue::Floats(&arrays.nodes_values)),
        ("nodes_modes", AttrValue::Strings(&modes_refs)),
        (
            "nodes_truenodeids",
            AttrValue::Ints(&arrays.nodes_truenodeids),
        ),
        (
            "nodes_falsenodeids",
            AttrValue::Ints(&arrays.nodes_falsenodeids),
        ),
        ("target_ids", AttrValue::Ints(&leaves.target_ids)),
        ("target_nodeids", AttrValue::Ints(&leaves.target_nodeids)),
        ("target_treeids", AttrValue::Ints(&leaves.target_treeids)),
        ("target_weights", AttrValue::Floats(&leaves.target_weights)),
        ("n_targets", AttrValue::Int(1)),
        ("post_transform", AttrValue::String("NONE")),
        ("aggregate_function", AttrValue::String(aggregate)),
    ];
    if !base_vals_owned.is_empty() {
        attrs.push(("base_values", AttrValue::Floats(&base_vals_owned)));
    }

    let node = build_node(
        "TreeEnsembleRegressor",
        &["X"],
        &["Y"],
        &attrs,
        ML_DOMAIN,
    );

    let inputs = [build_value_info("X", &[-1, n_features as i64])];
    let outputs = [build_value_info("Y", &[-1, 1])];

    let graph = build_graph(graph_name, &inputs, &outputs, &[node], &[]);
    Ok(build_model(&graph, ML_DOMAIN))
}

/// Build ONNX bytes for a tree ensemble classifier.
fn build_tree_ensemble_classifier(
    graph_name: &str,
    n_features: usize,
    n_classes: usize,
    arrays: &TreeEnsembleArrays,
    leaves: &ClassifierLeaves,
    post_transform: &str,
) -> Result<Vec<u8>> {
    let modes_refs: Vec<&str> = arrays.nodes_modes.iter().copied().collect();
    let class_labels: Vec<i64> = (0..n_classes as i64).collect();

    let attrs: Vec<(&str, AttrValue)> = vec![
        ("nodes_treeids", AttrValue::Ints(&arrays.nodes_treeids)),
        ("nodes_nodeids", AttrValue::Ints(&arrays.nodes_nodeids)),
        ("nodes_featureids", AttrValue::Ints(&arrays.nodes_featureids)),
        ("nodes_values", AttrValue::Floats(&arrays.nodes_values)),
        ("nodes_modes", AttrValue::Strings(&modes_refs)),
        (
            "nodes_truenodeids",
            AttrValue::Ints(&arrays.nodes_truenodeids),
        ),
        (
            "nodes_falsenodeids",
            AttrValue::Ints(&arrays.nodes_falsenodeids),
        ),
        ("class_ids", AttrValue::Ints(&leaves.class_ids)),
        ("class_nodeids", AttrValue::Ints(&leaves.class_nodeids)),
        ("class_treeids", AttrValue::Ints(&leaves.class_treeids)),
        ("class_weights", AttrValue::Floats(&leaves.class_weights)),
        ("classlabels_int64s", AttrValue::Ints(&class_labels)),
        ("post_transform", AttrValue::String(post_transform)),
    ];

    let node = build_node(
        "TreeEnsembleClassifier",
        &["X"],
        &["label", "probabilities"],
        &attrs,
        ML_DOMAIN,
    );

    let inputs = [build_value_info("X", &[-1, n_features as i64])];
    // TreeEnsembleClassifier outputs: label (int64) and probabilities (float)
    let label_out = build_value_info("label", &[-1]);
    let proba_out = build_value_info("probabilities", &[-1, n_classes as i64]);

    let graph = build_graph(
        graph_name,
        &inputs,
        &[label_out, proba_out],
        &[node],
        &[],
    );
    Ok(build_model(&graph, ML_DOMAIN))
}

// ── ToOnnx for Decision Tree models ─────────────────────────────────────

impl ToOnnx for DecisionTreeClassifier {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let ft = self.flat_tree().ok_or(ScryLearnError::NotFitted)?;
        let n_classes = self.n_classes();
        let n_features = self.n_features();

        let mut arrays = TreeEnsembleArrays::new();
        arrays.append_flat_tree(ft, 0);

        let mut leaves = ClassifierLeaves::new();
        leaves.append_flat_tree_proba(ft, 0, n_classes);

        build_tree_ensemble_classifier(
            "decision_tree_classifier",
            n_features,
            n_classes,
            &arrays,
            &leaves,
            "NONE",
        )
    }
}

impl ToOnnx for DecisionTreeRegressor {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let ft = self.flat_tree().ok_or(ScryLearnError::NotFitted)?;
        let n_features = self.n_features();

        let mut arrays = TreeEnsembleArrays::new();
        arrays.append_flat_tree(ft, 0);

        let mut leaves = RegressorLeaves::new();
        leaves.append_flat_tree(ft, 0, 1.0);

        build_tree_ensemble_regressor(
            "decision_tree_regressor",
            n_features,
            &arrays,
            &leaves,
            &[],
            "SUM",
        )
    }
}

// ── ToOnnx for Random Forest models ─────────────────────────────────────

impl ToOnnx for RandomForestClassifier {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let trees = self.trees();
        if trees.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        let n_classes = self.n_classes();
        let n_features = self.n_features();

        let mut arrays = TreeEnsembleArrays::new();
        let mut leaves = ClassifierLeaves::new();

        for (i, tree) in trees.iter().enumerate() {
            let ft = tree.flat_tree().ok_or(ScryLearnError::NotFitted)?;
            arrays.append_flat_tree(ft, i as i64);
            leaves.append_flat_tree_proba(ft, i as i64, n_classes);
        }

        build_tree_ensemble_classifier(
            "random_forest_classifier",
            n_features,
            n_classes,
            &arrays,
            &leaves,
            "NONE",
        )
    }
}

impl ToOnnx for RandomForestRegressor {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let trees = self.trees();
        if trees.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        let n_features = self.n_features();

        let mut arrays = TreeEnsembleArrays::new();
        let mut leaves = RegressorLeaves::new();

        for (i, tree) in trees.iter().enumerate() {
            let ft = tree.flat_tree().ok_or(ScryLearnError::NotFitted)?;
            arrays.append_flat_tree(ft, i as i64);
            leaves.append_flat_tree(ft, i as i64, 1.0);
        }

        build_tree_ensemble_regressor(
            "random_forest_regressor",
            n_features,
            &arrays,
            &leaves,
            &[],
            "AVERAGE",
        )
    }
}

// ── ToOnnx for Gradient Boosting models ─────────────────────────────────

impl ToOnnx for GradientBoostingRegressor {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let trees = self.trees();
        if trees.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        let n_features = self.n_features();
        let lr = self.learning_rate_val();

        let mut arrays = TreeEnsembleArrays::new();
        let mut leaves = RegressorLeaves::new();

        for (i, tree) in trees.iter().enumerate() {
            let ft = tree.flat_tree().ok_or(ScryLearnError::NotFitted)?;
            arrays.append_flat_tree(ft, i as i64);
            // Pre-scale leaf values by learning_rate for ONNX SUM aggregation.
            leaves.append_flat_tree(ft, i as i64, lr);
        }

        build_tree_ensemble_regressor(
            "gradient_boosting_regressor",
            n_features,
            &arrays,
            &leaves,
            &[self.init_prediction_val() as f32],
            "SUM",
        )
    }
}

impl ToOnnx for GradientBoostingClassifier {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let class_trees = self.class_trees();
        if class_trees.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        let n_classes = self.n_classes();
        let n_features = self.n_features();
        let lr = self.learning_rate_val();

        // GBT classifier: export as regressor (raw logits) since ONNX ML
        // TreeEnsembleClassifier expects probability-like weights.
        // We output raw scores and apply sigmoid/softmax externally.
        let mut arrays = TreeEnsembleArrays::new();
        let mut leaves = RegressorLeaves::new();

        let mut tree_id = 0i64;
        for class_seq in class_trees {
            for tree in class_seq {
                let ft = tree.flat_tree().ok_or(ScryLearnError::NotFitted)?;
                arrays.append_flat_tree(ft, tree_id);
                leaves.append_flat_tree(ft, tree_id, lr);
                tree_id += 1;
            }
        }

        let init_preds = self.init_predictions_val();
        let base_values: Vec<f32> = init_preds.iter().map(|&v| v as f32).collect();

        build_tree_ensemble_regressor(
            "gradient_boosting_classifier",
            n_features,
            &arrays,
            &leaves,
            &base_values,
            "SUM",
        )
    }
}

// ── ToOnnx for Histogram GBT models ────────────────────────────────────

impl ToOnnx for HistGradientBoostingRegressor {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let tree_views = self.tree_node_views();
        if tree_views.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        let n_features = self.n_features();
        let lr = self.learning_rate_val();

        let mut arrays = TreeEnsembleArrays::new();
        let mut leaves = RegressorLeaves::new();

        for (i, nodes) in tree_views.iter().enumerate() {
            arrays.append_hist_tree(nodes, i as i64);
            leaves.append_hist_leaves(nodes, i as i64, lr);
        }

        build_tree_ensemble_regressor(
            "hist_gradient_boosting_regressor",
            n_features,
            &arrays,
            &leaves,
            &[self.init_prediction_val() as f32],
            "SUM",
        )
    }
}

impl ToOnnx for HistGradientBoostingClassifier {
    fn to_onnx_bytes(&self) -> Result<Vec<u8>> {
        let class_tree_views = self.class_tree_node_views();
        if class_tree_views.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        let n_features = self.n_features();
        let lr = self.learning_rate_val();

        let mut arrays = TreeEnsembleArrays::new();
        let mut leaves = RegressorLeaves::new();

        let mut tree_id = 0i64;
        for class_seq in &class_tree_views {
            for nodes in class_seq {
                arrays.append_hist_tree(nodes, tree_id);
                leaves.append_hist_leaves(nodes, tree_id, lr);
                tree_id += 1;
            }
        }

        let init_preds = self.init_predictions_val();
        let base_values: Vec<f32> = init_preds.iter().map(|&v| v as f32).collect();

        build_tree_ensemble_regressor(
            "hist_gradient_boosting_classifier",
            n_features,
            &arrays,
            &leaves,
            &base_values,
            "SUM",
        )
    }
}

// ── Protobuf parsing helpers (for round-trip tests) ──────────────────────

#[cfg(test)]
mod proto_read {
    /// Read a varint from a byte slice, returning (value, bytes_consumed).
    pub(super) fn read_varint(data: &[u8]) -> Option<(u64, usize)> {
        let mut val: u64 = 0;
        let mut shift = 0;
        for (i, &byte) in data.iter().enumerate() {
            val |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Some((val, i + 1));
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
        None
    }

    /// Read a protobuf field: returns (field_number, wire_type, value_bytes, total_consumed).
    /// For varint: value_bytes contains the varint bytes.
    /// For length-delimited: value_bytes contains the payload.
    pub(super) fn read_field(data: &[u8]) -> Option<(u32, u32, &[u8], usize)> {
        let (tag, tag_len) = read_varint(data)?;
        let field = (tag >> 3) as u32;
        let wire = (tag & 0x7) as u32;
        let rest = &data[tag_len..];
        match wire {
            0 => {
                // Varint
                let (_, vlen) = read_varint(rest)?;
                Some((field, wire, &rest[..vlen], tag_len + vlen))
            }
            2 => {
                // Length-delimited
                let (len, llen) = read_varint(rest)?;
                let len = len as usize;
                let start = llen;
                Some((field, wire, &rest[start..start + len], tag_len + llen + len))
            }
            5 => {
                // 32-bit fixed
                Some((field, wire, &rest[..4], tag_len + 4))
            }
            1 => {
                // 64-bit fixed
                Some((field, wire, &rest[..8], tag_len + 8))
            }
            _ => None,
        }
    }

    /// Iterate over all fields in a protobuf message.
    pub(super) fn iter_fields(data: &[u8]) -> Vec<(u32, u32, Vec<u8>)> {
        let mut fields = Vec::new();
        let mut pos = 0;
        while pos < data.len() {
            if let Some((field, wire, value, consumed)) = read_field(&data[pos..]) {
                fields.push((field, wire, value.to_vec()));
                pos += consumed;
            } else {
                break;
            }
        }
        fields
    }

    /// Read packed floats from a length-delimited field.
    pub(super) fn read_packed_floats(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Read a varint from raw bytes.
    pub(super) fn decode_varint(data: &[u8]) -> u64 {
        read_varint(data).map_or(0, |(v, _)| v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::preprocess::Transformer;

    fn make_regression_data() -> Dataset {
        let mut rng = fastrand::Rng::with_seed(42);
        let n = 100;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let x0 = rng.f64() * 10.0;
            let x1 = rng.f64() * 10.0;
            f0.push(x0);
            f1.push(x1);
            target.push(3.0 * x0 + 2.0 * x1 + 1.0);
        }
        Dataset::new(vec![f0, f1], target, vec!["x0".into(), "x1".into()], "y")
    }

    fn make_classification_data() -> Dataset {
        let mut rng = fastrand::Rng::with_seed(42);
        let n = 200;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n / 2 {
            f0.push(-2.0 + rng.f64() * 2.0);
            f1.push(-2.0 + rng.f64() * 2.0);
            target.push(0.0);
        }
        for _ in n / 2..n {
            f0.push(2.0 + rng.f64() * 2.0);
            f1.push(2.0 + rng.f64() * 2.0);
            target.push(1.0);
        }
        Dataset::new(
            vec![f0, f1],
            target,
            vec!["x0".into(), "x1".into()],
            "class",
        )
    }

    #[test]
    fn linear_regression_onnx_export() {
        let data = make_regression_data();
        let mut model = LinearRegression::new();
        model.fit(&data).unwrap();
        let bytes = model.to_onnx_bytes().unwrap();
        assert!(!bytes.is_empty(), "ONNX output should be non-empty");
        // Should contain the graph name
        assert!(
            bytes.windows(17).any(|w| w == b"linear_regression"),
            "should contain graph name"
        );
    }

    #[test]
    fn logistic_regression_onnx_export() {
        let data = make_classification_data();
        let mut model = LogisticRegression::new().learning_rate(0.1).max_iter(100);
        model.fit(&data).unwrap();
        let bytes = model.to_onnx_bytes().unwrap();
        assert!(!bytes.is_empty());
        assert!(bytes.windows(7).any(|w| w == b"Sigmoid"));
    }

    #[test]
    fn standard_scaler_onnx_export() {
        let data = make_regression_data();
        let mut scaler = StandardScaler::new();
        scaler.fit(&data).unwrap();
        let bytes = scaler.to_onnx_bytes().unwrap();
        assert!(!bytes.is_empty());
        assert!(bytes.windows(15).any(|w| w == b"standard_scaler"));
    }

    #[test]
    fn mlp_classifier_onnx_export() {
        let data = make_classification_data();
        let mut model = MLPClassifier::new()
            .hidden_layers(&[8, 4])
            .max_iter(50)
            .learning_rate(0.01);
        model.fit(&data).unwrap();
        let bytes = model.to_onnx_bytes().unwrap();
        assert!(!bytes.is_empty());
        assert!(bytes.windows(14).any(|w| w == b"mlp_classifier"));
    }

    #[test]
    fn mlp_regressor_onnx_export() {
        let data = make_regression_data();
        let mut model = MLPRegressor::new()
            .hidden_layers(&[8, 4])
            .max_iter(50)
            .learning_rate(0.01);
        model.fit(&data).unwrap();
        let bytes = model.to_onnx_bytes().unwrap();
        assert!(!bytes.is_empty());
        assert!(bytes.windows(13).any(|w| w == b"mlp_regressor"));
    }

    #[test]
    fn onnx_round_trip_linear_weights() {
        let data = make_regression_data();
        let mut model = LinearRegression::new();
        model.fit(&data).unwrap();
        let bytes = model.to_onnx_bytes().unwrap();

        // Parse top-level ModelProto to find graph
        let model_fields = proto_read::iter_fields(&bytes);

        // field 1 = ir_version
        let ir_field = model_fields.iter().find(|(f, _, _)| *f == 1).unwrap();
        let ir_version = proto_read::decode_varint(&ir_field.2);
        assert_eq!(ir_version, 8, "IR version should be 8");

        // field 7 = graph
        let graph_data = model_fields
            .iter()
            .find(|(f, _, _)| *f == 7)
            .expect("graph field")
            .2
            .clone();
        let graph_fields = proto_read::iter_fields(&graph_data);

        // field 5 = initializer (tensors). First should be W.
        let initializers: Vec<_> = graph_fields.iter().filter(|(f, _, _)| *f == 5).collect();
        assert_eq!(initializers.len(), 2, "should have W and b initializers");

        // Parse W tensor
        let w_fields = proto_read::iter_fields(&initializers[0].2);
        let w_floats = w_fields
            .iter()
            .find(|(f, _, _)| *f == 4)
            .map(|(_, _, data)| proto_read::read_packed_floats(data))
            .unwrap_or_default();

        let expected: Vec<f32> = model.coefficients().iter().map(|&v| v as f32).collect();
        assert_eq!(w_floats.len(), expected.len());
        for (a, b) in w_floats.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "weight mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn onnx_file_write_read() {
        let data = make_regression_data();
        let mut model = LinearRegression::new();
        model.fit(&data).unwrap();

        let bytes = model.to_onnx_bytes().unwrap();
        let tmp = std::env::temp_dir().join("scry_test_model.onnx");
        model.to_onnx(&tmp).unwrap();

        let file_bytes = std::fs::read(&tmp).unwrap();
        assert_eq!(bytes, file_bytes, "file bytes should match in-memory bytes");

        // Clean up
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn untrained_model_errors() {
        let model = LinearRegression::new();
        let err = model.to_onnx_bytes();
        assert!(err.is_err());
        assert!(
            matches!(err.unwrap_err(), ScryLearnError::NotFitted),
            "should error with NotFitted"
        );

        let logreg = LogisticRegression::new();
        assert!(logreg.to_onnx_bytes().is_err());

        let scaler = StandardScaler::new();
        assert!(scaler.to_onnx_bytes().is_err());
    }

    #[test]
    fn onnx_opset_version() {
        let data = make_regression_data();
        let mut model = LinearRegression::new();
        model.fit(&data).unwrap();
        let bytes = model.to_onnx_bytes().unwrap();

        let model_fields = proto_read::iter_fields(&bytes);

        // field 8 = opset_import
        let opset_data = model_fields
            .iter()
            .find(|(f, _, _)| *f == 8)
            .expect("opset_import field")
            .2
            .clone();
        let opset_fields = proto_read::iter_fields(&opset_data);

        // field 2 in OpsetIdProto = version
        let version_field = opset_fields.iter().find(|(f, _, _)| *f == 2).unwrap();
        let version = proto_read::decode_varint(&version_field.2);
        assert_eq!(version, 18, "opset version should be 18");
    }

    #[test]
    fn onnx_protobuf_structure() {
        let data = make_regression_data();
        let mut model = LinearRegression::new();
        model.fit(&data).unwrap();
        let bytes = model.to_onnx_bytes().unwrap();

        let model_fields = proto_read::iter_fields(&bytes);

        // Check key fields exist
        assert!(
            model_fields.iter().any(|(f, _, _)| *f == 1),
            "should have ir_version (field 1)"
        );
        assert!(
            model_fields.iter().any(|(f, _, _)| *f == 7),
            "should have graph (field 7)"
        );
        assert!(
            model_fields.iter().any(|(f, _, _)| *f == 8),
            "should have opset_import (field 8)"
        );

        // Parse graph
        let graph_data = model_fields
            .iter()
            .find(|(f, _, _)| *f == 7)
            .unwrap()
            .2
            .clone();
        let graph_fields = proto_read::iter_fields(&graph_data);

        // field 1 = nodes, field 2 = name, field 5 = initializers, field 11 = inputs, field 12 = outputs
        let nodes: Vec<_> = graph_fields.iter().filter(|(f, _, _)| *f == 1).collect();
        assert_eq!(
            nodes.len(),
            2,
            "linear regression should have 2 nodes (MatMul + Add)"
        );

        let inputs: Vec<_> = graph_fields.iter().filter(|(f, _, _)| *f == 11).collect();
        assert_eq!(inputs.len(), 1, "should have 1 input");

        let outputs: Vec<_> = graph_fields.iter().filter(|(f, _, _)| *f == 12).collect();
        assert_eq!(outputs.len(), 1, "should have 1 output");

        // Check node op_types contain MatMul and Add
        let has_matmul = nodes.iter().any(|(_, _, data)| {
            let fields = proto_read::iter_fields(data);
            fields.iter().any(|(f, _, v)| *f == 4 && v == b"MatMul")
        });
        let has_add = nodes.iter().any(|(_, _, data)| {
            let fields = proto_read::iter_fields(data);
            fields.iter().any(|(f, _, v)| *f == 4 && v == b"Add")
        });
        assert!(has_matmul, "should have MatMul node");
        assert!(has_add, "should have Add node");
    }
}
