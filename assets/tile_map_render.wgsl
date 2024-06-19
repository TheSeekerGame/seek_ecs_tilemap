

//@group(2) @binding(0) var<uniform> tilemap_texture: texture_2d_array<u32>;
//@group(2) @binding(1) var base_color_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertex(
    @builtin(vertex_index) vertex_id: u32,
) -> VertexOutput {

    let tile_index = vertex_id / 6u;
    let vertex_index = vertex_id % 6u;

    let tile_x = f32(tile_index % 2048u);
    let tile_y = f32(tile_index / 2048u);

    // probably a better way of doing this
    var vertex_pos = vec2<f32>(0.0, 0.0);
    switch (vertex_index) {
        case 0u: { vertex_pos = vec2<f32>(-0.5, -0.5); }
        case 1u: { vertex_pos = vec2<f32>(0.5, -0.5); }
        case 2u: { vertex_pos = vec2<f32>(-0.5, 0.5); }
        case 3u: { vertex_pos = vec2<f32>(-0.5, 0.5); }
        case 4u: { vertex_pos = vec2<f32>(0.5, -0.5); }
        case 5u: { vertex_pos = vec2<f32>(0.5, 0.5); }
        default: {}
    }

    var output: VertexOutput;
    output.position = vec4<f32>(
        tile_x + vertex_pos.x,
        tile_y + vertex_pos.y,
        0.0,
        1.0
    );

    return output;
}

@fragment
fn fragment(vert_output: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(1.0, 1.0, 1.0, 1.0);
}