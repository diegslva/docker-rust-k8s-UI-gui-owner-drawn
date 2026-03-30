// Shader 3D -- iluminacao Phong com cor configuravel por mesh (tint)
// Suporta multi-mesh via dynamic uniform offset: cada mesh tem seu proprio tint.

struct CameraUniform {
    mvp:          mat4x4<f32>,
    model_normal: mat4x4<f32>,
    light_dir:    vec3<f32>,
    _pad:         f32,
    tint:         vec3<f32>,
    _pad2:        f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos     = camera.mvp * vec4<f32>(in.position, 1.0);
    // Transformar normal com a matriz normal para escala/rotacao corretas
    out.world_normal = normalize((camera.model_normal * vec4<f32>(in.normal, 0.0)).xyz);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let light  = normalize(camera.light_dir);

    // Phong: ambient + diffuse (sem specular para aparencia organica)
    let ambient  = 0.18;
    let diffuse  = max(dot(normal, light), 0.0) * 0.75;

    // Luz de preenchimento fraco vindo de baixo (evita sombra total)
    let fill     = max(dot(normal, -light * vec3<f32>(1.0, 0.3, 1.0)), 0.0) * 0.08;

    let intensity = ambient + diffuse + fill;

    // Cor configuravel por mesh via uniform (tint)
    return vec4<f32>(camera.tint * intensity, 1.0);
}
