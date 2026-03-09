// Instanced line segment rendering with SDF AA.

struct Uniforms {
    viewport: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct LineInstance {
    @location(0) endpoints: vec4<f32>,     // x0, y0, x1, y1
    @location(1) params: vec4<f32>,        // width, dash_offset, total_len, 0
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) half_length: f32,
    @location(3) half_width: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: LineInstance,
) -> VertexOutput {
    var out: VertexOutput;

    let p0 = instance.endpoints.xy;
    let p1 = instance.endpoints.zw;
    let width = instance.params.x;

    let dir = p1 - p0;
    let len = length(dir);
    let tangent = select(vec2<f32>(1.0, 0.0), dir / len, len > 0.001);
    let normal = vec2<f32>(-tangent.y, tangent.x);

    let half_len = len * 0.5;
    let half_w = width * 0.5 + 1.0; // +1 for AA

    // 6 vertices for a quad
    var quad_pos: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let q = quad_pos[vertex_index];
    let center = (p0 + p1) * 0.5;
    let offset = tangent * q.x * (half_len + 1.0) + normal * q.y * half_w;
    let pixel_pos = center + offset;

    let ndc_x = (pixel_pos.x / uniforms.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel_pos.y / uniforms.viewport.w) * 2.0;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    out.local_pos = vec2<f32>(q.x * (half_len + 1.0), q.y * half_w);
    out.color = instance.color;
    out.half_length = half_len;
    out.half_width = width * 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // SDF for a capsule/line segment
    let dx = max(abs(in.local_pos.x) - in.half_length, 0.0);
    let dy = abs(in.local_pos.y);
    let dist = length(vec2<f32>(dx, dy)) - in.half_width;

    let alpha = 1.0 - smoothstep(-0.5, 0.5, dist);
    if alpha < 0.001 {
        discard;
    }
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
