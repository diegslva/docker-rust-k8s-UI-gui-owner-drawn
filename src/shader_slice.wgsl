// Shader do plano de corte MRI — samplea textura 3D do volume
// Renderiza um quad semi-transparente mostrando a fatia MRI no espaco 3D

struct CameraUniform {
    mvp:           mat4x4<f32>,
    model_normal:  mat4x4<f32>,
    light_dir:     vec3<f32>,
    roughness:     f32,
    tint:          vec3<f32>,
    alpha:         f32,
    sss_strength:  f32,
    use_texture:   f32,
    _pad:          vec2<f32>,
}

struct SliceParams {
    world_min:  vec3<f32>,
    _pad0:      f32,
    world_max:  vec3<f32>,
    alpha:      f32,        // transparencia do plano (0.85 default)
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var volume_tex: texture_3d<f32>;
@group(1) @binding(1)
var volume_sampler: sampler;
@group(1) @binding(2)
var<uniform> params: SliceParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) texcoord: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos  = camera.mvp * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Converter posicao world -> UVW [0,1]^3 para sampling do volume
    let range = params.world_max - params.world_min;
    var uvw = (in.world_pos - params.world_min) / range;

    // Correcao de lateralidade: NIfTI usa convencao radiologica (X invertido)
    uvw.x = 1.0 - uvw.x;

    // Clamp para evitar sampling fora do volume
    let uvw_clamped = clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0));

    // Sample do volume 3D (grayscale, canal R)
    let intensity = textureSample(volume_tex, volume_sampler, uvw_clamped).r;

    // Descarta pixels fora do cerebro (intensidade ~0 = ar/fundo)
    if intensity < 0.02 {
        discard;
    }

    // Grayscale com leve tom azul frio (estilo MRI)
    let color = vec3<f32>(
        intensity * 0.90,
        intensity * 0.93,
        intensity * 1.00,
    );

    return vec4<f32>(color, params.alpha * intensity);
}
