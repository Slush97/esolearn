// Shader for tessellated geometry (areas, paths).

struct Uniforms {
    viewport: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct PushConstants {
    color: vec4<f32>,
    transform: mat3x3<f32>,
};

@group(0) @binding(1)
var<uniform> push: PushConstants;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = push.transform * vec3<f32>(in.position, 1.0);

    let ndc_x = (world.x / uniforms.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (world.y / uniforms.viewport.w) * 2.0;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return push.color;
}
