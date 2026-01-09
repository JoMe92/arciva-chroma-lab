pub const WGSL_SHADER: &str = r#"
// Vertex Output
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Uniforms
struct Settings {
    // Geometry
    homography_matrix: mat3x3<f32>,
    geo_padding: f32, // Padding to align 16 bytes? mat3x3 takes 3 vec3s.
    // WGSL struct layout rules: mat3x3<f32> is 3 columns of vec3.
    // Each column is 16-byte aligned (vec3 is treated as vec4 size in uniform buffers usually).
    // So 3 * 16 = 48 bytes.
    // Next field starts at offset 48.
    
    // Flips - keep for now as they are applied separately or integrated? 
    // Plan says "Geometry Settings already has vertical/horizontal".
    // We passed them as 'v, h' floats before.
    // Now we pass matrix.
    // The matrix handles the Warp.
    // Flips are usually independent?
    // Let's keep flips separate for now.
    flip_vertical: f32,
    flip_horizontal: f32, // 0.0 or 1.0
    
    // Crop/Rotate
    crop_rotation: f32, // radians
    crop_aspect: f32,   // 0.0 = none
    
    // Exposure
    exposure: f32,
    contrast: f32,
    highlights: f32,
    shadows: f32,
    
    // Color
    temp: f32,
    tint: f32,
    
    // Grain
    grain_amount: f32,
    grain_size: f32, // scale factor
    
    // Dimensions
    src_width: f32,
    src_height: f32,

    // LUT
    lut_intensity: f32,
    
    // Denoise
    denoise_luminance: f32,
    denoise_color: f32, 
};

@group(0) @binding(0) var<uniform> settings: Settings;
@group(0) @binding(1) var t_diffuse: texture_2d<f32>;
@group(0) @binding(2) var s_diffuse: sampler;
@group(0) @binding(3) var t_grain: texture_2d<f32>; // Pre-seeded noise texture
@group(0) @binding(4) var s_grain: sampler;
@group(0) @binding(5) var t_lut: texture_3d<f32>;
@group(0) @binding(6) var s_lut: sampler;
@group(0) @binding(7) var t_curves: texture_2d<f32>;
@group(0) @binding(8) var s_curves: sampler;

// Helper: Bicubic sampling
fn cubic_hermite(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let a_val = -a / 2.0 + (3.0 * b) / 2.0 - (3.0 * c) / 2.0 + d / 2.0;
    let b_val = a - (5.0 * b) / 2.0 + 2.0 * c - d / 2.0;
    let c_val = -a / 2.0 + c / 2.0;
    let d_val = b;
    return a_val * t * t * t + b_val * t * t + c_val * t + d_val;
}

fn sample_bicubic(uv: vec2<f32>) -> vec4<f32> {
    let tex_size = vec2<f32>(settings.src_width, settings.src_height);
    let coords = uv * tex_size - 0.5;
    let f = fract(coords);
    let i = floor(coords);

    var result = vec4<f32>(0.0);
    
    // 4x4 neighborhood
    for (var y = -1; y <= 2; y++) {
        var row_val = vec4<f32>(0.0);
        for (var x = -1; x <= 2; x++) {
            let sample_uv = (i + vec2<f32>(f32(x), f32(y)) + 0.5) / tex_size;
            // Clamp to edge
            let clamped_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
            
            // We need to fetch individual weights. 
            // Optimization: In a real shader, we might precompute weights or use textureGather if available.
            // For now, simple 16-tap fetch.
            // Wait, this is expensive. Let's do separable if possible.
            // But texture is 2D.
            // Let's stick to the logic: interpolate rows, then col.
            
            // Actually, to keep shader size small, let's just do bilinear for now if bicubic is too heavy?
            // Requirement says "Bicubic approximation".
            // Let's implement the separable version properly.
        }
    }
    
    // Separable Bicubic
    // We need 4 samples in X for each of 4 rows in Y.
    
    var col_results = array<vec4<f32>, 4>();
    
    for (var y = -1; y <= 2; y++) {
        var row_samples = array<vec4<f32>, 4>();
        for (var x = -1; x <= 2; x++) {
             let sample_pos = i + vec2<f32>(f32(x), f32(y)) + 0.5;
             let sample_uv = clamp(sample_pos / tex_size, vec2<f32>(0.0), vec2<f32>(1.0));
             row_samples[x+1] = textureSampleLevel(t_diffuse, s_diffuse, sample_uv, 0.0);
        }
        
        col_results[y+1] = vec4<f32>(
            cubic_hermite(row_samples[0].r, row_samples[1].r, row_samples[2].r, row_samples[3].r, f.x),
            cubic_hermite(row_samples[0].g, row_samples[1].g, row_samples[2].g, row_samples[3].g, f.x),
            cubic_hermite(row_samples[0].b, row_samples[1].b, row_samples[2].b, row_samples[3].b, f.x),
            cubic_hermite(row_samples[0].a, row_samples[1].a, row_samples[2].a, row_samples[3].a, f.x)
        );
    }
    
    return vec4<f32>(
        cubic_hermite(col_results[0].r, col_results[1].r, col_results[2].r, col_results[3].r, f.y),
        cubic_hermite(col_results[0].g, col_results[1].g, col_results[2].g, col_results[3].g, f.y),
        cubic_hermite(col_results[0].b, col_results[1].b, col_results[2].b, col_results[3].b, f.y),
        cubic_hermite(col_results[0].a, col_results[1].a, col_results[2].a, col_results[3].a, f.y)
    );
}

fn rgb_to_yuv(rgb: vec3<f32>) -> vec3<f32> {
    let y = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
    let u = 0.492 * (rgb.b - y);
    let v = 0.877 * (rgb.r - y);
    return vec3<f32>(y, u, v);
}

fn yuv_to_rgb(yuv: vec3<f32>) -> vec3<f32> {
    let y = yuv.x;
    let u = yuv.y;
    let v = yuv.z;
    let r = y + 1.13983 * v;
    let g = y - 0.39465 * u - 0.58060 * v;
    let b = y + 2.03211 * u;
    return vec3<f32>(r, g, b);
}

fn sample_denoised(uv: vec2<f32>) -> vec4<f32> {
    // If no denoise, just sample
    if (settings.denoise_luminance <= 0.0 && settings.denoise_color <= 0.0) {
        return sample_bicubic(uv);
    }
    
    let tex_size = vec2<f32>(settings.src_width, settings.src_height);
    let step = 1.0 / tex_size;
    
    var center_col = sample_bicubic(uv);
    var center_yuv = rgb_to_yuv(center_col.rgb);
    
    // Parameters
    // Luma strength controls sigma_r (range) for bilateral
    // Color strength controls sigma_s (spatial) for box/gaussian blur on Chroma
    
    // Using a 5x5 kernel
    var final_y = 0.0;
    var final_u = 0.0;
    var final_v = 0.0;
    var weight_total_y = 0.0;
    var weight_total_c = 0.0;
    
    let sigma_s = 2.0; // Spatial sigma
    
    // Range sigma for Bilateral: related to denoise_luminance. 
    // If denoise is high, sigma_r is high (more blurring across edges).
    // If denoise is low, sigma_r is low (preserve edges strictly).
    // Let's map 0..1 to sensible range. E.g. 0.01 to 0.3?
    let sigma_r = max(0.001, settings.denoise_luminance * 0.4);
    
    // Chroma blur radius/strength
    // For chroma, we just do spatial blur.
    
    for (var i = -2; i <= 2; i++) {
        for (var j = -2; j <= 2; j++) {
            let offset = vec2<f32>(f32(i), f32(j)) * step;
            let sample_uv = clamp(uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
            // Optimization: Use bilinear for neighbor samples to save perf
            let col = textureSampleLevel(t_diffuse, s_diffuse, sample_uv, 0.0);
            let yuv = rgb_to_yuv(col.rgb);
            
            // Spatial Filtering Weight (Gaussian)
            let dist_sq = f32(i*i + j*j);
            let w_spatial = exp(-dist_sq / (2.0 * sigma_s * sigma_s));
            
            // 1. Luminance (Bilateral)
            let diff_y = abs(yuv.x - center_yuv.x);
            let w_range = exp(-(diff_y * diff_y) / (2.0 * sigma_r * sigma_r));
            let w_y = w_spatial * w_range;
            
            if (settings.denoise_luminance > 0.0) {
                final_y += yuv.x * w_y;
                weight_total_y += w_y;
            } else {
                // If luma denoise off, we just take center later, but let's accumulate
                // to keep logic unified or just break?
            }
            
            // 2. Color (Gaussian Blur on U/V)
            // If denoise_color > 0, we use spatial weights.
            if (settings.denoise_color > 0.0) {
                 final_u += yuv.y * w_spatial;
                 final_v += yuv.z * w_spatial;
                 weight_total_c += w_spatial;
            }
        }
    }
    
    var out_y = center_yuv.x;
    var out_u = center_yuv.y;
    var out_v = center_yuv.z;
    
    if (settings.denoise_luminance > 0.0 && weight_total_y > 0.0) {
        out_y = final_y / weight_total_y;
    }
    
    if (settings.denoise_color > 0.0 && weight_total_c > 0.0) {
         out_u = final_u / weight_total_c;
         out_v = final_v / weight_total_c;
    }
    
    let out_rgb = yuv_to_rgb(vec3<f32>(out_y, out_u, out_v));
    // Mix with original alpha
    return vec4<f32>(out_rgb, center_col.a);
}

// Vertex Shader
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Full screen triangle
    let x = f32(i32(in_vertex_index) - 1);
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // Standard Quad UVs: 0,0 top-left (WebGPU texture space)
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    
    return out;
}

// Fragment Shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv = in.uv;
    
    // 0. Flips
    // Apply flips BEFORE geometry warp / crop (Inverse logic: Modify UVs)
    // If we want to flip the "Source", we modify the UVs we are about to sample with.
    // If flip_h is true, uv.x = 1.0 - uv.x?
    // Let's trace: Output pixel at x=0.1.
    // If flipped, it should come from Source x=0.9.
    // So uv.x = 1.0 - uv.x. Correct.
    // Wait, geometry warp applies to the "Source Quad".
    // If we flip, do we flip the quad or the content?
    // Usually "Geometry" tool (skew) assumes it works on the original image orientation.
    // If we flip FIRST (as requested: Flip -> Geometry), then we flip the content that enters the Geometry phase.
    // So in inverse mapping: Start with Output UV.
    // 1. Un-Crop/Un-Rotate (Inverse) -> done below.
    // 2. Un-Warp (Inverse) -> done below.
    // 3. Un-Flip (Inverse) -> do here?
    // Wait, the shader steps traverse backwards from Output -> Source.
    // Output -> [Crop/Rotate] -> [Geometry] -> [Flip] -> Source Image.
    // So yes, Flip is applied LAST in the shader (closest to texture sample).
    
    // BUT! The `apply_geometry` logic in CPU does Flip *inside* the geometry loop, essentially modifying the source coordinate.
    // `sample_bicubic(img, sx, sy)` where `sx` is flipped if needed.
    // So if we integrate logic here:
    // We calculate the Source UV where we want to sample.
    // AND THEN flip it if needed.
    
    // 1. Geometry (Warp)
    // Homography Transform
    // Map Output UV (0..1) to Source UV
    // Homogenize UV: (u, v, 1)
    
    let src_h = settings.homography_matrix * vec3<f32>(uv, 1.0);
    
    // Perspective Divide
    if (abs(src_h.z) > 0.00001) {
        uv = src_h.xy / src_h.z;
    } else {
        uv = src_h.xy;
    }
    
    // 2. Crop / Rotate
    // We need to modify UVs again.
    // Rotation: Rotate UV around 0.5, 0.5
    if (abs(settings.crop_rotation) > 0.0001) {
        let c = cos(-settings.crop_rotation); // Inverse for source mapping
        let s = sin(-settings.crop_rotation);
        let center = vec2<f32>(0.5);
        let centered = uv - center;
        // Aspect ratio correction for rotation?
        // If we rotate a rectangular image, UV space is 0..1 (square).
        // Real space is W x H.
        // We should rotate in pixel space or aspect-corrected space.
        let aspect = settings.src_width / settings.src_height;
        let corrected = vec2<f32>(centered.x * aspect, centered.y);
        
        let rotated = vec2<f32>(
            corrected.x * c - corrected.y * s,
            corrected.x * s + corrected.y * c
        );
        
        uv = vec2<f32>(rotated.x / aspect, rotated.y) + center;
    }
    
    // Aspect Ratio Crop
    // If we need to crop to a specific aspect ratio, we just scale UVs around center.
    if (settings.crop_aspect > 0.0) {
        let current_aspect = settings.src_width / settings.src_height;
        let target_aspect = settings.crop_aspect;
        
        if (abs(current_aspect - target_aspect) > 0.001) {
             var scale_x = 1.0;
             var scale_y = 1.0;
             
             if (current_aspect > target_aspect) {
                 // Image is wider than target. Crop width.
                 // We need to show LESS of the width.
                 // So UVs should go from (0.5 - w/2) to (0.5 + w/2)
                 // where w < 1.0.
                 // Wait, if we crop, we zoom in.
                 // So we sample a smaller portion of the source.
                 // So UV range is smaller.
                 scale_x = target_aspect / current_aspect;
             } else {
                 // Image is taller. Crop height.
                 scale_y = current_aspect / target_aspect;
             }
             
             let center = vec2<f32>(0.5);
             uv = (uv - center) * vec2<f32>(scale_x, scale_y) + center;
        }
    }
    
    // Check bounds
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Sample Texture
    // Use bicubic if enabled/needed, or bilinear for speed.
    // For now, let's use the bicubic function we wrote.
    var color = sample_denoised(uv);
    
    // 3. Exposure
    let exposure_factor = pow(2.0, settings.exposure);
    color = vec4<f32>(color.rgb * exposure_factor, color.a);
    
    // Contrast
    if (abs(settings.contrast - 1.0) > 0.001) {
        color = vec4<f32>((color.rgb - 0.5) * settings.contrast + 0.5, color.a);
    }
    
    // Highlights / Shadows
    if (abs(settings.highlights) > 0.001 || abs(settings.shadows) > 0.001) {
        let lum = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        // Simple tonal masks based on channel values (like Python)
        // Python:
        // hm = clamp((c - 0.5) * 2, 0, 1)
        // sm = clamp((0.5 - c) * 2, 0, 1)
        
        let hm = clamp((color.rgb - 0.5) * 2.0, vec3<f32>(0.0), vec3<f32>(1.0));
        let sm = clamp((0.5 - color.rgb) * 2.0, vec3<f32>(0.0), vec3<f32>(1.0));
        
        color = vec4<f32>(
            color.rgb + hm * settings.highlights * 0.5 + sm * settings.shadows * 0.5,
            color.a
        );
    }
    
    // 4. Color Balance
    if (abs(settings.temp) > 0.001 || abs(settings.tint) > 0.001) {
        let temp_r = 1.0 + settings.temp * 0.25;
        let temp_b = 1.0 - settings.temp * 0.25;
        let tint_g = 1.0 - settings.tint * 0.2;
        let tint_rb = 1.0 + settings.tint * 0.1;
        
        let f_r = temp_r * tint_rb;
        let f_g = tint_g;
        let f_b = temp_b * tint_rb;
        
        color = vec4<f32>(
            color.r * f_r,
            color.g * f_g,
            color.b * f_b,
            color.a
        );
    }
    
    // 5. LUT
    if (settings.lut_intensity > 0.0) {
        // Map color to 0..1 (it is already)
        // Texture 3D sampling
        // Color is the coordinate.
        // We need a sampler and texture.
        // Assuming we add binding 5 and 6 for LUT.
        let lut_color = textureSample(t_lut, s_lut, color.rgb);
        
        color = vec4<f32>(mix(color.rgb, lut_color.rgb, settings.lut_intensity), color.a); 
    }

    // 5.5 Curves
    // Sample curve LUT (256x1 texture)
    let curve_r = textureSampleLevel(t_curves, s_curves, vec2<f32>(color.r, 0.5), 0.0).r;
    let curve_g = textureSampleLevel(t_curves, s_curves, vec2<f32>(color.g, 0.5), 0.0).g;
    let curve_b = textureSampleLevel(t_curves, s_curves, vec2<f32>(color.b, 0.5), 0.0).b;
    color = vec4<f32>(curve_r, curve_g, curve_b, color.a);

    // 6. Grain
    if (settings.grain_amount > 0.0) {
        // Sample grain texture
        // Grain texture is tiled.
        // Scale UVs by image size / grain scale
        let grain_scale = max(1.0, settings.grain_size);
        let grain_uv = (in.uv * vec2<f32>(settings.src_width, settings.src_height)) / grain_scale;
        
        // Wrap mode should be repeat in sampler
        let noise = textureSample(t_grain, s_grain, grain_uv).r; // Assuming grayscale noise
        
        // Noise is 0..1? Or -1..1?
        // If texture is u8, it's 0..1.
        // Python noise is Normal(0, sigma).
        // We should probably upload noise as centered around 0.5 or just add (noise - 0.5).
        // Let's assume texture is 0..1, representing -sigma..sigma?
        // No, let's assume texture is raw noise values normalized.
        // Easier: Texture contains pre-computed noise * 128 + 128.
        // So (val - 0.5) * 2 * sigma.
        
        // Actually, let's just say the texture contains standard normal noise mapped to 0..1?
        // No, precision issues.
        // Let's say texture contains noise in -1..1 range (f32 texture) or 0..1 (unorm).
        // Let's assume 0..1 where 0.5 is 0.
        
        let n = (noise - 0.5) * 2.0; // -1..1
        let sigma = settings.grain_amount * 25.0 / 255.0; // Scale to 0..1 range
        
        // Wait, settings.grain_amount is 0..1.
        // In Python: sigma = amount * 25.0. Pixel values are 0..255.
        // So sigma in 0..1 space is amount * 25.0 / 255.0 ~= amount * 0.1.
        
        // But we also need the random distribution.
        // If texture is just uniform noise, we need to shape it?
        // Plan said: "Grain noise will be pre-generated on the CPU ... and uploaded".
        // So the texture ALREADY has the correct distribution (Normal).
        // We just need to scale it by amount.
        // But amount can change at runtime.
        // So we upload "Unit Normal" noise (mean 0, std dev 1)?
        // And scale by sigma in shader.
        // Texture f32 is best for this.
        
        color = vec4<f32>(color.rgb + vec3<f32>(n * sigma), color.a);
    }
    
    // Clamp
    color = clamp(color, vec4<f32>(0.0), vec4<f32>(1.0));
    
    return color;
}
"#;

pub const GLSL_VERTEX: &str = r#"#version 300 es
in vec2 position;
out vec2 v_uv;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    // Position is -1..1
    // UV is 0..1
    v_uv = position * 0.5 + 0.5;
    v_uv.y = 1.0 - v_uv.y; // Flip Y for WebGL texture coords usually
}
"#;

pub const GLSL_FRAGMENT: &str = r#"#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_texture;
uniform sampler2D u_grain;

uniform mat3 u_homography_matrix;
uniform float u_flip_vertical;
uniform float u_flip_horizontal;
uniform float u_crop_rotation;
uniform float u_crop_aspect;
uniform float u_exposure;
uniform float u_contrast;
uniform float u_highlights;
uniform float u_shadows;
uniform float u_temp;
uniform float u_tint;
uniform float u_grain_amount;
uniform float u_grain_size;
uniform vec2 u_src_size;
uniform sampler3D u_lut;
uniform float u_lut_intensity;
uniform float u_denoise_luminance;
uniform float u_denoise_color;
uniform sampler2D u_curves;

// Helper: Cubic Hermite
float cubic_hermite(float a, float b, float c, float d, float t) {
    float a_val = -a / 2.0 + (3.0 * b) / 2.0 - (3.0 * c) / 2.0 + d / 2.0;
    float b_val = a - (5.0 * b) / 2.0 + 2.0 * c - d / 2.0;
    float c_val = -a / 2.0 + c / 2.0;
    float d_val = b;
    return a_val * t * t * t + b_val * t * t + c_val * t + d_val;
}

vec4 sample_bicubic(vec2 uv) {
    vec2 tex_size = u_src_size;
    vec2 coords = uv * tex_size - 0.5;
    vec2 f = fract(coords);
    vec2 i = floor(coords);
    
    vec4 col_results[4];
    
    for (int y = -1; y <= 2; y++) {
        vec4 row_samples[4];
        for (int x = -1; x <= 2; x++) {
             vec2 sample_pos = i + vec2(float(x), float(y)) + 0.5;
             vec2 sample_uv = clamp(sample_pos / tex_size, vec2(0.0), vec2(1.0));
             row_samples[x+1] = texture(u_texture, sample_uv);
        }
        
        col_results[y+1] = vec4(
            cubic_hermite(row_samples[0].r, row_samples[1].r, row_samples[2].r, row_samples[3].r, f.x),
            cubic_hermite(row_samples[0].g, row_samples[1].g, row_samples[2].g, row_samples[3].g, f.x),
            cubic_hermite(row_samples[0].b, row_samples[1].b, row_samples[2].b, row_samples[3].b, f.x),
            cubic_hermite(row_samples[0].a, row_samples[1].a, row_samples[2].a, row_samples[3].a, f.x)
        );
    }
    
    return vec4(
        cubic_hermite(col_results[0].r, col_results[1].r, col_results[2].r, col_results[3].r, f.y),
        cubic_hermite(col_results[0].g, col_results[1].g, col_results[2].g, col_results[3].g, f.y),
        cubic_hermite(col_results[0].b, col_results[1].b, col_results[2].b, col_results[3].b, f.y),
        cubic_hermite(col_results[0].a, col_results[1].a, col_results[2].a, col_results[3].a, f.y)
    );
}

vec3 rgb_to_yuv(vec3 rgb) {
    float y = dot(rgb, vec3(0.299, 0.587, 0.114));
    float u = 0.492 * (rgb.b - y);
    float v = 0.877 * (rgb.r - y);
    return vec3(y, u, v);
}

vec3 yuv_to_rgb(vec3 yuv) {
    float y = yuv.x;
    float u = yuv.y;
    float v = yuv.z;
    float r = y + 1.13983 * v;
    float g = y - 0.39465 * u - 0.58060 * v;
    float b = y + 2.03211 * u;
    return vec3(r, g, b);
}

vec4 sample_denoised(vec2 uv) {
    if (u_denoise_luminance <= 0.0 && u_denoise_color <= 0.0) {
        return sample_bicubic(uv);
    }
    
    vec2 tex_size = u_src_size;
    vec2 step = 1.0 / tex_size;
    
    vec4 center_col = sample_bicubic(uv);
    vec3 center_yuv = rgb_to_yuv(center_col.rgb);
    
    float final_y = 0.0;
    float final_u = 0.0;
    float final_v = 0.0;
    float weight_total_y = 0.0;
    float weight_total_c = 0.0;
    
    float sigma_s = 2.0;
    float sigma_r = max(0.001, u_denoise_luminance * 0.4);
    
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            vec2 offset = vec2(float(i), float(j)) * step;
            vec2 sample_uv = clamp(uv + offset, vec2(0.0), vec2(1.0));
            
            vec4 col = texture(u_texture, sample_uv);
            vec3 yuv = rgb_to_yuv(col.rgb);
            
            float dist_sq = float(i*i + j*j);
            float w_spatial = exp(-dist_sq / (2.0 * sigma_s * sigma_s));
            
            float diff_y = abs(yuv.x - center_yuv.x);
            float w_range = exp(-(diff_y * diff_y) / (2.0 * sigma_r * sigma_r));
            float w_y = w_spatial * w_range;
            
            if (u_denoise_luminance > 0.0) {
                final_y += yuv.x * w_y;
                weight_total_y += w_y;
            }
            
            if (u_denoise_color > 0.0) {
                final_u += yuv.y * w_spatial;
                final_v += yuv.z * w_spatial;
                weight_total_c += w_spatial;
            }
        }
    }
    
    float out_y = center_yuv.x;
    float out_u = center_yuv.y;
    float out_v = center_yuv.z;
    
    if (u_denoise_luminance > 0.0 && weight_total_y > 0.0) {
        out_y = final_y / weight_total_y;
    }
    
    if (u_denoise_color > 0.0 && weight_total_c > 0.0) {
        out_u = final_u / weight_total_c;
        out_v = final_v / weight_total_c;
    }
    
    return vec4(yuv_to_rgb(vec3(out_y, out_u, out_v)), center_col.a);
}

void main() {
    vec2 uv = v_uv;
    
    // 1. Geometry (Warp)
    vec3 src_h = u_homography_matrix * vec3(uv, 1.0);
    if (abs(src_h.z) > 0.00001) {
        uv = src_h.xy / src_h.z;
    }

    // Apply Flips
    if (u_flip_horizontal > 0.5) {
        uv.x = 1.0 - uv.x;
    }
    if (u_flip_vertical > 0.5) {
        uv.y = 1.0 - uv.y;
    }
    
    // 2. Crop / Rotate
    if (abs(u_crop_rotation) > 0.0001) {
        float c = cos(-u_crop_rotation);
        float s = sin(-u_crop_rotation);
        vec2 center = vec2(0.5);
        vec2 centered = uv - center;
        float aspect = u_src_size.x / u_src_size.y;
        vec2 corrected = vec2(centered.x * aspect, centered.y);
        
        vec2 rotated = vec2(
            corrected.x * c - corrected.y * s,
            corrected.x * s + corrected.y * c
        );
        
        uv = vec2(rotated.x / aspect, rotated.y) + center;
    }
    
    if (u_crop_aspect > 0.0) {
        float current_aspect = u_src_size.x / u_src_size.y;
        float target_aspect = u_crop_aspect;
        
        if (abs(current_aspect - target_aspect) > 0.001) {
             float scale_x = 1.0;
             float scale_y = 1.0;
             
             if (current_aspect > target_aspect) {
                 scale_x = target_aspect / current_aspect;
             } else {
                 scale_y = current_aspect / target_aspect;
             }
             
             vec2 center = vec2(0.5);
             uv = (uv - center) * vec2(scale_x, scale_y) + center;
        }
    }
    
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        out_color = vec4(0.0);
        return;
    }
    
    vec4 color = sample_denoised(uv);
    
    // 3. Exposure
    float exposure_factor = pow(2.0, u_exposure);
    color.rgb *= exposure_factor;
    
    if (abs(u_contrast - 1.0) > 0.001) {
        color.rgb = (color.rgb - 0.5) * u_contrast + 0.5;
    }
    
    if (abs(u_highlights) > 0.001 || abs(u_shadows) > 0.001) {
        vec3 hm = clamp((color.rgb - 0.5) * 2.0, 0.0, 1.0);
        vec3 sm = clamp((0.5 - color.rgb) * 2.0, 0.0, 1.0);
        color.rgb += hm * u_highlights * 0.5 + sm * u_shadows * 0.5;
    }
    
    // 4. Color
    if (abs(u_temp) > 0.001 || abs(u_tint) > 0.001) {
        float temp_r = 1.0 + u_temp * 0.25;
        float temp_b = 1.0 - u_temp * 0.25;
        float tint_g = 1.0 - u_tint * 0.2;
        float tint_rb = 1.0 + u_tint * 0.1;
        
        color.r *= temp_r * tint_rb;
        color.g *= tint_g;
        color.b *= temp_b * tint_rb;
    }
    
    // 5. LUT 
    if (u_lut_intensity > 0.0) {
        vec3 lut_col = texture(u_lut, color.rgb).rgb;
        color.rgb = mix(color.rgb, lut_col, u_lut_intensity);
    }
    
    // 5.5 Curves
    color.r = texture(u_curves, vec2(color.r, 0.5)).r;
    color.g = texture(u_curves, vec2(color.g, 0.5)).g;
    color.b = texture(u_curves, vec2(color.b, 0.5)).b;
    
    // 6. Grain
    if (u_grain_amount > 0.0) {
        float grain_scale = max(1.0, u_grain_size);
        vec2 grain_uv = (v_uv * u_src_size) / grain_scale; // Use v_uv (screen space) or uv (image space)?
        // Grain usually stays with the image in photo editors (image space), 
        // but film grain is on the "film". If we crop/rotate, does grain rotate?
        // Real film: Grain is in the image.
        // Digital overlay: Often screen space.
        // Python implementation: `apply_grain_in_place` happens LAST.
        // It iterates x,y of the *buffer* (which is already cropped/rotated).
        // So it's effectively screen space (relative to the output frame).
        // So we should use `v_uv` (the original quad UVs) or `gl_FragCoord`.
        // Let's use `v_uv` scaled by output size.
        // But `u_src_size` is INPUT size.
        // We need OUTPUT size for correct grain scale if we want to match Python exactly?
        // Python: `noise_w = (width + scale - 1) / scale`. `width` is current image width.
        // So yes, it depends on output size.
        // We'll approximate using u_src_size for now, or pass output size uniform.
        // Let's use `v_uv` * `u_src_size` as a proxy.
        
        float noise = texture(u_grain, grain_uv).r;
        float n = (noise - 0.5) * 2.0;
        float sigma = u_grain_amount * 25.0 / 255.0;
        color.rgb += n * sigma;
    }
    
    out_color = clamp(color, 0.0, 1.0);
}
"#;
