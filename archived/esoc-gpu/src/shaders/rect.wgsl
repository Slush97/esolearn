// Instanced rectangle rendering with rounded corners via SDF.

struct Uniforms {
    viewport: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct RectInstance {
    @location(0) rect: vec4<f32>,           // x, y, w, h
    @location(1) fill_color: vec4<f32>,     // RGBA
    @location(2) stroke_color: vec4<f32>,   // RGBA
    @location(3) params: vec4<f32>,         // corner_radius, stroke_width, 0, 0
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) rect_size: vec2<f32>,
    @location(2) fill_color: vec4<f32>,
    @location(3) stroke_color: vec4<f32>,
    @location(4) params: vec4<f32>,
};

// Unit quad vertices
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
    instance: RectInstance,
) -> VertexOutput {
    var out: VertexOutput;
    let quad_pos = QUAD_POS[vertex_index];

    // Expand by stroke width + 1px for AA
    let expand = instance.params.y + 1.0;
    let expanded_rect = vec4<f32>(
        instance.rect.x - expand,
        instance.rect.y - expand,
        instance.rect.z + expand * 2.0,
        instance.rect.w + expand * 2.0,
    );

    let pixel_pos = expanded_rect.xy + quad_pos * expanded_rect.zw;

    let ndc_x = (pixel_pos.x / uniforms.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel_pos.y / uniforms.viewport.w) * 2.0;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    // Local position relative to rect center
    out.local_pos = (quad_pos - 0.5) * expanded_rect.zw;
    out.rect_size = instance.rect.zw;
    out.fill_color = instance.fill_color;
    out.stroke_color = instance.stroke_color;
    out.params = instance.params;
    return out;
}

// Signed distance to a rounded rectangle
fn sdf_rounded_rect(p: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    let r = min(radius, min(half_size.x, half_size.y));
    let q = abs(p) - half_size + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let half_size = in.rect_size * 0.5;
    let radius = in.params.x;
    let stroke_width = in.params.y;

    let dist = sdf_rounded_rect(in.local_pos, half_size, radius);

    // Fill: inside the rectangle
    let fill_alpha = 1.0 - smoothstep(-0.5, 0.5, dist);
    var color = in.fill_color * fill_alpha;

    // Stroke: band around the edge
    if stroke_width > 0.0 {
        let stroke_dist = abs(dist + stroke_width * 0.5) - stroke_width * 0.5;
        let stroke_alpha = 1.0 - smoothstep(-0.5, 0.5, stroke_dist);
        color = mix(color, vec4<f32>(in.stroke_color.rgb, in.stroke_color.a * stroke_alpha), stroke_alpha);
    }

    if color.a < 0.001 {
        discard;
    }
    return color;
}
