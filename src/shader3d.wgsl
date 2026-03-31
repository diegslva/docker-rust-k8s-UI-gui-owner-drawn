// Shader 3D -- Medical-grade rendering with optional texture mapping
// Blinn-Phong + Fresnel + SSS + AO + texture support for anatomical models

struct CameraUniform {
    mvp:           mat4x4<f32>,
    model_normal:  mat4x4<f32>,
    light_dir:     vec3<f32>,
    roughness:     f32,
    tint:          vec3<f32>,
    alpha:         f32,
    sss_strength:  f32,
    use_texture:   f32,   // > 0.5 = sample texture, <= 0.5 = use tint
    _pad:          vec2<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var diffuse_texture: texture_2d<f32>;
@group(1) @binding(1)
var diffuse_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) texcoord: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_normal: vec3<f32>,
    @location(1)       view_dir:     vec3<f32>,
    @location(2)       uv:          vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos     = camera.mvp * vec4<f32>(in.position, 1.0);
    out.world_normal = normalize((camera.model_normal * vec4<f32>(in.normal, 0.0)).xyz);
    out.view_dir     = normalize(-out.clip_pos.xyz);
    out.uv           = in.texcoord;
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

    // --- Base color: textura ou tint ---
    var base_color: vec3<f32>;
    if camera.use_texture > 0.5 {
        // UV flip Y: OBJ convencao (0,0)=bottom-left, GPU (0,0)=top-left
        let uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
        base_color = textureSample(diffuse_texture, diffuse_sampler, uv).rgb;
        // sRGB -> linear (aproximacao gamma 2.2)
        base_color = pow(base_color, vec3<f32>(2.2, 2.2, 2.2));
    } else {
        base_color = camera.tint;
    }

    // --- Ambient Occlusion aproximado ---
    let ao = mix(0.65, 1.0, N.y * 0.5 + 0.5);

    // --- Ambient ---
    let ambient = 0.18 * ao;

    // --- Key light (diffuse) ---
    let diffuse = NdotL * 0.65;

    // --- Specular (Blinn-Phong com roughness) ---
    let spec_power = mix(8.0, 128.0, 1.0 - camera.roughness);
    let specular = pow(NdotH, spec_power) * mix(0.15, 0.40, 1.0 - camera.roughness);

    // --- Fill light ---
    let fill_dir = normalize(vec3<f32>(-L.x, L.y * 0.3, -L.z));
    let fill = max(dot(N, fill_dir), 0.0) * 0.12;

    // --- Fresnel (Schlick) ---
    let F0 = 0.04;
    let fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
    let rim = fresnel * 0.35;

    // --- Subsurface Scattering ---
    let sss_dot = max(dot(-N, L), 0.0);
    let sss = sss_dot * camera.sss_strength;
    let sss_color = vec3<f32>(1.0, 0.75, 0.65);

    // --- Final ---
    let direct_light = ambient + diffuse + fill + specular + rim;
    let color = base_color * direct_light + sss_color * sss;

    // Linear -> sRGB output (se usando textura, precisa do gamma correto)
    var out_color: vec3<f32>;
    if camera.use_texture > 0.5 {
        out_color = pow(color, vec3<f32>(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
    } else {
        out_color = color;
    }

    return vec4<f32>(out_color, camera.alpha);
}
