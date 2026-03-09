// Text rendering via pre-rasterized glyph atlas (alpha-tested).
// Phase 2 uses a simple bitmap atlas; MSDF upgrade in Phase 4.

struct Uniforms {
    viewport: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var atlas_texture: texture_2d<f32>;

@group(0) @binding(2)
var atlas_sampler: sampler;

struct GlyphInstance {
    @location(0) rect: vec4<f32>,       // x, y, w, h (screen pixels)
    @location(1) uv_rect: vec4<f32>,    // u0, v0, u1, v1 (atlas UVs)
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

var<private> QUAD_POS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: GlyphInstance,
) -> VertexOutput {
    var out: VertexOutput;
    let q = QUAD_POS[vertex_index];

    let pixel_pos = instance.rect.xy + q * instance.rect.zw;

    let ndc_x = (pixel_pos.x / uniforms.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel_pos.y / uniforms.viewport.w) * 2.0;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    out.uv = instance.uv_rect.xy + q * (instance.uv_rect.zw - instance.uv_rect.xy);
    out.color = instance.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let atlas_val = textureSample(atlas_texture, atlas_sampler, in.uv);
    let alpha = atlas_val.r * in.color.a;
    if alpha < 0.01 {
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha);
}
