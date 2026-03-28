// Common uniforms shared by all passes.
struct Uniforms {
    viewport: vec4<f32>,  // x, y, width, height
    // Padding to align to 16 bytes
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// Convert from pixel coordinates to NDC [-1, 1].
fn pixel_to_ndc(pos: vec2<f32>) -> vec2<f32> {
    let x = (pos.x / uniforms.viewport.z) * 2.0 - 1.0;
    let y = 1.0 - (pos.y / uniforms.viewport.w) * 2.0;
    return vec2<f32>(x, y);
}
