// Vertex shader — quad fullscreen via indices (sem vertex buffer)
// Indices 0..5 formam dois triangulos cobrindo o clip space inteiro.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // UV normalizado [0,1] para calculo do gradiente
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Posicoes do quad fullscreen em clip space [-1, 1]
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    let pos = positions[vertex_index];

    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    // UV: mapeia [-1,1] para [0,1], Y invertido para origem top-left
    out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 1.0 - (pos.y * 0.5 + 0.5));
    return out;
}

// Fragment shader — gradiente radial do centro para as bordas

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Distancia do centro (0.5, 0.5) normalizada
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(in.uv - center) * 1.6;

    // Tom-sobre-tom azul: bordas azul-noite profundo
    let dark = vec3<f32>(0.035, 0.055, 0.085);

    // Centro: azul-teal mais claro (glow sutil)
    let glow = vec3<f32>(0.08, 0.15, 0.24);

    // Blend suave: glow no centro, grafite nas bordas
    let t = smoothstep(0.0, 1.0, dist);
    let color = mix(glow, dark, t);

    return vec4<f32>(color, 1.0);
}
