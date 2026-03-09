// Instanced point/marker rendering via SDF shapes.

struct Uniforms {
    viewport: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct PointInstance {
    @location(0) center_size: vec4<f32>,     // cx, cy, size, shape_type
    @location(1) fill_color: vec4<f32>,
    @location(2) stroke_color: vec4<f32>,
    @location(3) params: vec4<f32>,          // stroke_width, 0, 0, 0
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) fill_color: vec4<f32>,
    @location(2) stroke_color: vec4<f32>,
    @location(3) size: f32,
    @location(4) shape_type: f32,
    @location(5) stroke_width: f32,
};

var<private> QUAD_POS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: PointInstance,
) -> VertexOutput {
    var out: VertexOutput;
    let quad_pos = QUAD_POS[vertex_index];

    let half_size = instance.center_size.z * 0.5 + instance.params.x + 1.0;
    let pixel_pos = instance.center_size.xy + quad_pos * half_size;

    let ndc_x = (pixel_pos.x / uniforms.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel_pos.y / uniforms.viewport.w) * 2.0;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    out.local_pos = quad_pos * half_size;
    out.fill_color = instance.fill_color;
    out.stroke_color = instance.stroke_color;
    out.size = instance.center_size.z;
    out.shape_type = instance.center_size.w;
    out.stroke_width = instance.params.x;
    return out;
}

// SDF for circle
fn sdf_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

// SDF for square
fn sdf_square(p: vec2<f32>, r: f32) -> f32 {
    let d = abs(p) - vec2<f32>(r, r);
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

// SDF for diamond
fn sdf_diamond(p: vec2<f32>, r: f32) -> f32 {
    let d = abs(p);
    return (d.x + d.y - r * 1.2) * 0.707;
}

// SDF for triangle (pointing up)
fn sdf_triangle(p: vec2<f32>, r: f32) -> f32 {
    let k = sqrt(3.0);
    var q = vec2<f32>(abs(p.x) - r, p.y + r / k);
    if q.x + k * q.y > 0.0 {
        q = vec2<f32>(q.x - k * q.y, -k * q.x - q.y) / 2.0;
    }
    q.x -= clamp(q.x, -2.0 * r, 0.0);
    return -length(q) * sign(q.y);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r = in.size * 0.5;
    let shape = u32(in.shape_type + 0.5);

    var dist: f32;
    switch shape {
        case 0u: { dist = sdf_circle(in.local_pos, r); }     // Circle
        case 1u: { dist = sdf_square(in.local_pos, r); }     // Square
        case 2u: { dist = sdf_diamond(in.local_pos, r); }    // Diamond
        case 3u: { dist = sdf_triangle(in.local_pos, r); }   // TriangleUp
        case 4u: {
            let flipped = vec2<f32>(in.local_pos.x, -in.local_pos.y);
            dist = sdf_triangle(flipped, r);                   // TriangleDown
        }
        case 5u: {                                             // Cross
            let d = abs(in.local_pos);
            dist = min(d.x, d.y) - r * 0.25;
        }
        default: { dist = sdf_circle(in.local_pos, r); }
    }

    let fill_alpha = 1.0 - smoothstep(-0.5, 0.5, dist);
    var color = in.fill_color * fill_alpha;

    if in.stroke_width > 0.0 {
        let stroke_dist = abs(dist + in.stroke_width * 0.5) - in.stroke_width * 0.5;
        let stroke_alpha = 1.0 - smoothstep(-0.5, 0.5, stroke_dist);
        color = mix(color, vec4<f32>(in.stroke_color.rgb, in.stroke_color.a * stroke_alpha), stroke_alpha);
    }

    if color.a < 0.001 {
        discard;
    }
    return color;
}
