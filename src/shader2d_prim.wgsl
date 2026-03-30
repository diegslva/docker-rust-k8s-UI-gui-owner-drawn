// shader2d_prim.wgsl — geometria 2D colorida (linhas como quads, retangulos).
// Recebe posicao em NDC diretamente — nenhuma transformacao de camera.
// Usado para callout lines, box backgrounds e separadores do painel clinico.

struct VertIn {
    @location(0) pos: vec2<f32>,
    @location(1) col: vec4<f32>,
}

struct VertOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
}

@vertex
fn vs_main(v: VertIn) -> VertOut {
    var out: VertOut;
    out.clip = vec4<f32>(v.pos, 0.0, 1.0);
    out.col  = v.col;
    return out;
}

@fragment
fn fs_main(v: VertOut) -> @location(0) vec4<f32> {
    return v.col;
}
