// Shader 3D -- Medical-grade rendering
// Blinn-Phong + Fresnel + Subsurface Scattering + Ambient Occlusion
// Three-point lighting (key + fill + rim) para visualizacao anatomica.

struct CameraUniform {
    mvp:           mat4x4<f32>,
    model_normal:  mat4x4<f32>,
    light_dir:     vec3<f32>,
    roughness:     f32,         // 0.0=espelho, 1.0=fosco
    tint:          vec3<f32>,
    alpha:         f32,         // 1.0=opaco, <1.0=semi-transparente
    sss_strength:  f32,         // subsurface scattering (0.0=off, 0.15=brain)
    _pad:          vec3<f32>,
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
    out.view_dir     = normalize(-out.clip_pos.xyz);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let L = normalize(camera.light_dir);
    let V = normalize(in.view_dir);
    let H = normalize(L + V);

    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0);
    let NdotH = max(dot(N, H), 0.0);

    // --- Ambient Occlusion aproximado (cavity darkening) ---
    // Normais apontando para baixo (sulcos cerebrais) recebem menos luz ambiente.
    let ao = mix(0.65, 1.0, N.y * 0.5 + 0.5);

    // --- Ambient ---
    let ambient = 0.18 * ao;

    // --- Key light (diffuse) ---
    let diffuse = NdotL * 0.65;

    // --- Specular (Blinn-Phong com roughness variavel) ---
    let spec_power = mix(8.0, 128.0, 1.0 - camera.roughness);
    let specular = pow(NdotH, spec_power) * mix(0.15, 0.40, 1.0 - camera.roughness);

    // --- Fill light (lado oposto, tom frio, mais suave) ---
    let fill_dir = normalize(vec3<f32>(-L.x, L.y * 0.3, -L.z));
    let fill = max(dot(N, fill_dir), 0.0) * 0.12;

    // --- Fresnel (Schlick approximation) ---
    // F0 = 0.04 para dieletricos (tecido biologico)
    let F0 = 0.04;
    let fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    // --- Rim light (reforçado pelo Fresnel) ---
    let rim = fresnel * 0.35;

    // --- Subsurface Scattering simulado ---
    // Luz que "atravessa" o tecido translucido (hemisferios cerebrais).
    // Simulado como diffuse invertido com tom rosado quente (sangue sob tecido).
    let sss_dot = max(dot(-N, L), 0.0);
    let sss = sss_dot * camera.sss_strength;
    let sss_color = vec3<f32>(1.0, 0.75, 0.65); // tom rosado (hemoglobina)

    // --- Composicao final ---
    let direct_light = ambient + diffuse + fill + specular + rim;
    let color = camera.tint * direct_light + sss_color * sss;

    return vec4<f32>(color, camera.alpha);
}
