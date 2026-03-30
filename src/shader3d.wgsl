// Shader 3D -- Blinn-Phong + rim light + alpha configuravel por mesh
// Usado tanto para o cerebro (semi-transparente) quanto para os tumores (opacos).

struct CameraUniform {
    mvp:          mat4x4<f32>,
    model_normal: mat4x4<f32>,
    light_dir:    vec3<f32>,
    _pad:         f32,
    tint:         vec3<f32>,
    alpha:        f32,   // 1.0 = opaco, <1.0 = semi-transparente
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
    @location(1)       view_dir:     vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos     = camera.mvp * vec4<f32>(in.position, 1.0);
    out.world_normal = normalize((camera.model_normal * vec4<f32>(in.normal, 0.0)).xyz);
    // View direction aproximada (suficiente para Blinn-Phong sem eye pos no uniform)
    out.view_dir     = normalize(-out.clip_pos.xyz);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let light  = normalize(camera.light_dir);
    let view   = normalize(in.view_dir);

    // --- Blinn-Phong ---
    let ambient  = 0.15;
    let diffuse  = max(dot(normal, light), 0.0) * 0.70;

    // Specular (Blinn-Phong half-vector)
    let half_vec  = normalize(light + view);
    let spec_pow  = 32.0;
    let specular  = pow(max(dot(normal, half_vec), 0.0), spec_pow) * 0.25;

    // Luz de preenchimento (fill light) -- evita sombra total no lado oposto
    let fill = max(dot(normal, vec3<f32>(-light.x * 0.4, -light.y * 0.2, -light.z * 0.4)), 0.0) * 0.10;

    // Rim light -- borda luminosa para dar profundidade medica
    let rim_strength = 1.0 - max(dot(normal, view), 0.0);
    let rim = pow(rim_strength, 3.0) * 0.20;

    let intensity = ambient + diffuse + fill + specular + rim;

    return vec4<f32>(camera.tint * intensity, camera.alpha);
}
