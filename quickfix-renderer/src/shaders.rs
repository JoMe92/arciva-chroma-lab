pub const WGSL_SHADER: &str = r#"
// Vertex Output
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Uniforms
struct Settings {
    // Geometry
    geo_vertical: f32,
    geo_horizontal: f32,
    // Flips
    flip_vertical: f32,   // 0.0 or 1.0
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
};

@group(0) @binding(0) var<uniform> settings: Settings;
@group(0) @binding(1) var t_diffuse: texture_2d<f32>;
@group(0) @binding(2) var s_diffuse: sampler;
@group(0) @binding(3) var t_grain: texture_2d<f32>; // Pre-seeded noise texture
@group(0) @binding(4) var s_grain: sampler;

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
    // Inverse mapping: find where current UV comes from in source
    // Bilinear warp approximation
    let v = clamp(settings.geo_vertical, -1.0, 1.0);
    let h = clamp(settings.geo_horizontal, -1.0, 1.0);
    
    if (abs(v) > 0.0001 || abs(h) > 0.0001) {
        let max_x_offset = 0.25;
        let max_y_offset = 0.25;
        
        let top_inset = v * max_x_offset;
        let bottom_inset = -v * max_x_offset;
        let left_y = h * max_y_offset;
        let right_y = -h * max_y_offset;
        
        // Corners in UV space (0..1)
        // We want to map the unit square UV to the distorted quad.
        // Actually, we want the inverse: For a pixel in the output (unit square), where is it in the source?
        // The source is the distorted quad? No, the source is the original image (unit square).
        // The output is the distorted image.
        // So we map Output UV -> Source UV.
        // If we pull the top corners in (positive vertical), we are "zooming out" the top.
        // So we need to sample a WIDER area of the source at the top.
        // So Source UV range at top > 1.0?
        // Wait, the Python implementation:
        // "Calculate quad corners (source coordinates mapped to destination)"
        // It maps Source (0..W, 0..H) to Destination Quad.
        // Then it iterates Destination pixels and interpolates Source coordinates.
        // So we do the same here.
        // We interpolate the Source UVs based on current UV.
        
        let ul = vec2<f32>(0.0 + top_inset, 0.0 + left_y);
        let ll = vec2<f32>(0.0 + bottom_inset, 1.0 - left_y); // 1.0 because UV y is 0..1
        let lr = vec2<f32>(1.0 - bottom_inset, 1.0 - right_y);
        let ur = vec2<f32>(1.0 - top_inset, 0.0 + right_y);
        
        // Bilinear interpolation of these corners based on current UV
        // P(u,v) = mix(mix(ul, ur, u), mix(ll, lr, u), v)
        let top = mix(ul, ur, uv.x);
        let bottom = mix(ll, lr, uv.x);
        uv = mix(top, bottom, uv.y);
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
    var color = sample_bicubic(uv);
    
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
    
    // 5. Grain
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

uniform float u_geo_vertical;
uniform float u_geo_horizontal;
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

void main() {
    vec2 uv = v_uv;
    
    // 1. Geometry (Warp)
    float v = clamp(u_geo_vertical, -1.0, 1.0);
    float h = clamp(u_geo_horizontal, -1.0, 1.0);
    
    if (abs(v) > 0.0001 || abs(h) > 0.0001) {
        float max_x_offset = 0.25;
        float max_y_offset = 0.25;
        
        float top_inset = v * max_x_offset;
        float bottom_inset = -v * max_x_offset;
        float left_y = h * max_y_offset;
        float right_y = -h * max_y_offset;
        
        vec2 ul = vec2(0.0 + top_inset, 0.0 + left_y);
        vec2 ll = vec2(0.0 + bottom_inset, 1.0 - left_y);
        vec2 lr = vec2(1.0 - bottom_inset, 1.0 - right_y);
        vec2 ur = vec2(1.0 - top_inset, 0.0 + right_y);
        
        vec2 top = mix(ul, ur, uv.x);
        vec2 bottom = mix(ll, lr, uv.x);
        uv = mix(top, bottom, uv.y);
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
    
    vec4 color = sample_bicubic(uv);
    
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
    
    // 5. Grain
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
