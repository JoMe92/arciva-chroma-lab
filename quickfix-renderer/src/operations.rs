use crate::{
    ClaritySettings, ColorSettings, CropSettings, CurvesSettings, DehazeSettings, DenoiseSettings,
    ExposureSettings, GeometrySettings, GrainSettings, HslSettings, QuickFixAdjustments,
    SharpenSettings, SplitToningSettings, VignetteSettings, LensDistortionSettings,
};
use image::{ImageBuffer, Rgba, RgbaImage};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::cmp::max;

// Helper to clamp values
fn clamp(v: f32, min_val: f32, max_val: f32) -> f32 {
    v.max(min_val).min(max_val)
}

fn clamp_u8(v: f32) -> u8 {
    clamp(v, 0.0, 255.0) as u8
}

use crate::api_types::Lut3DSettings;
// New imports for potential test module or general use
use super::*;
use crate::{CropRect, ChannelCurve, CurvePoint};


pub fn compute_histogram(data: &[u8]) -> Vec<u32> {
    let mut histogram = vec![0u32; 256 * 3]; // R, G, B concatenated

    // Each pixel is 4 bytes: R, G, B, A
    for chunk in data.chunks_exact(4) {
        let r = chunk[0] as usize;
        let g = chunk[1] as usize;
        let b = chunk[2] as usize;
        // Alpha ignored

        histogram[r] += 1;
        histogram[256 + g] += 1;
        histogram[512 + b] += 1;
    }

    histogram
}

pub fn process_frame_internal(
    data: &mut [u8],
    width: u32,
    height: u32,
    adjustments: &QuickFixAdjustments,
    lut_buffer: Option<(&[f32], u32)>,
) -> Result<(Vec<u8>, u32, u32, Vec<u32>), String> {
    // Convert raw bytes to ImageBuffer
    let mut img: RgbaImage =
        ImageBuffer::from_raw(width, height, data.to_vec()).ok_or("Invalid buffer size")?;

    // 0. Lens Distortion
    if let Some(distortion) = &adjustments.distortion {
        img = apply_lens_distortion(&img, distortion);
    }

    // 1. Geometry
    if let Some(geo) = &adjustments.geometry {
        img = apply_geometry(&img, geo);
    }

    // 2. Crop/Rotate
    if let Some(crop) = &adjustments.crop {
        img = apply_crop_rotate(&img, crop);
    }

    // 3. Denoise
    if let Some(denoise) = &adjustments.denoise {
        apply_denoise_in_place(&mut img, denoise);
    }

    // 4. Exposure
    if let Some(exp) = &adjustments.exposure {
        apply_exposure_in_place(&mut img, exp);
    }

    // 5. Color
    if let Some(col) = &adjustments.color {
        apply_color_balance_in_place(&mut img, col);
    }

    // 6. Curves
    if let Some(curves) = &adjustments.curves {
        apply_curves_in_place(&mut img, curves);
    }

    // 7. HSL
    if let Some(hsl) = &adjustments.hsl {
        apply_hsl_in_place(&mut img, hsl);
    }

    // 7.5 Split Toning
    if let Some(st) = &adjustments.split_toning {
        apply_split_toning_in_place(&mut img, st);
    }

    // 7.6 Dehaze
    if let Some(dehaze) = &adjustments.dehaze {
        apply_dehaze_in_place(&mut img, dehaze);
    }

    // 7.7 Clarity
    if let Some(clarity) = &adjustments.clarity {
        apply_clarity_in_place(&mut img, clarity);
    }

    // 8. LUT
    if let (Some(lut_settings), Some((lut_data, lut_size))) = (&adjustments.lut, lut_buffer) {
        apply_lut_in_place(&mut img, lut_data, lut_size, lut_settings);
    }

    // 8.5 Vignette
    if let Some(vignette) = &adjustments.vignette {
        apply_vignette_in_place(&mut img, vignette);
    }

    // 8.6 Sharpen
    if let Some(sharpen) = &adjustments.sharpen {
        apply_sharpen_in_place(&mut img, sharpen);
    }

    // 9. Grain
    if let Some(grain) = &adjustments.grain {
        apply_grain_in_place(&mut img, grain);
    }

    let (w, h) = img.dimensions();
    let raw = img.into_raw();
    let histogram = compute_histogram(&raw);
    Ok((raw, w, h, histogram))
}

// Bicubic interpolation helper
fn cubic_hermite(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let a_val = -a / 2.0 + (3.0 * b) / 2.0 - (3.0 * c) / 2.0 + d / 2.0;
    let b_val = a - (5.0 * b) / 2.0 + 2.0 * c - d / 2.0;
    let c_val = -a / 2.0 + c / 2.0;
    let d_val = b;

    a_val * t * t * t + b_val * t * t + c_val * t + d_val
}

fn get_pixel_clamped(img: &RgbaImage, x: i32, y: i32) -> Rgba<u8> {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let x = x.max(0).min(w - 1);
    let y = y.max(0).min(h - 1);
    *img.get_pixel(x as u32, y as u32)
}

fn sample_bicubic(img: &RgbaImage, u: f32, v: f32) -> Rgba<u8> {
    let x = u - 0.5;
    let y = v - 0.5;
    let x_int = x.floor() as i32;
    let y_int = y.floor() as i32;
    let x_frac = x - x.floor();
    let y_frac = y - y.floor();

    let mut result = [0.0; 4];

    for c in 0..4 {
        let mut col_values = [0.0; 4];
        for (i, dy) in (-1..=2).enumerate() {
            let mut row_values = [0.0; 4];
            for (j, dx) in (-1..=2).enumerate() {
                let px = get_pixel_clamped(img, x_int + dx, y_int + dy);
                row_values[j] = px[c] as f32;
            }
            col_values[i] = cubic_hermite(
                row_values[0],
                row_values[1],
                row_values[2],
                row_values[3],
                x_frac,
            );
        }
        result[c] = cubic_hermite(
            col_values[0],
            col_values[1],
            col_values[2],
            col_values[3],
            y_frac,
        );
    }

    Rgba([
        clamp_u8(result[0]),
        clamp_u8(result[1]),
        clamp_u8(result[2]),
        clamp_u8(result[3]),
    ])
}

fn apply_geometry(img: &RgbaImage, settings: &GeometrySettings) -> RgbaImage {
    let v = clamp(settings.vertical.unwrap_or(0.0), -1.0, 1.0);
    let h = clamp(settings.horizontal.unwrap_or(0.0), -1.0, 1.0);
    let flip_v = settings.flip_vertical.unwrap_or(false);
    let flip_h = settings.flip_horizontal.unwrap_or(false);

    // Fast path: No perspective distortion
    if v.abs() < 1e-5 && h.abs() < 1e-5 {
        if !flip_h && !flip_v {
            return img.clone();
        }
        let mut res = img.clone();
        if flip_h {
            res = image::imageops::flip_horizontal(&res);
        }
        if flip_v {
            res = image::imageops::flip_vertical(&res);
        }
        return res;
    }

    let width = img.width();
    let height = img.height();

    // Homography Calculation
    let corners = crate::geometry::calculate_distortion_state(v, h);
    let matrix = crate::geometry::calculate_homography_from_unit_square(&corners);

    // Apply flip logic separately if needed or integrate.
    // The shader applies flip AFTER warp (closer to texture sample).
    // Let's match shader logic: Output UV -> (Warp) -> Source UV -> (Flip) -> Sample.

    let mut new_img = RgbaImage::new(width, height);

    let w_f32 = width as f32;
    let h_f32 = height as f32;

    for y in 0..height {
        let v_coord = y as f32 / h_f32; // Normalised Output V
        for x in 0..width {
            let u_coord = x as f32 / w_f32; // Normalised Output U

            // 1. Homography Transform
            // src_h = H * (u, v, 1)
            let src_x_h = matrix[0] * u_coord + matrix[1] * v_coord + matrix[2];
            let src_y_h = matrix[3] * u_coord + matrix[4] * v_coord + matrix[5];
            let src_z_h = matrix[6] * u_coord + matrix[7] * v_coord + matrix[8];

            // Perspective Divide
            let mut sx_norm = if src_z_h.abs() > 1e-6 {
                src_x_h / src_z_h
            } else {
                src_x_h
            };

            let mut sy_norm = if src_z_h.abs() > 1e-6 {
                src_y_h / src_z_h
            } else {
                src_y_h
            };

            // 2. Apply Flips
            if flip_h {
                sx_norm = 1.0 - sx_norm;
            }
            if flip_v {
                sy_norm = 1.0 - sy_norm;
            }

            // Check bounds (0..1)
            if !(0.0..=1.0).contains(&sx_norm) || !(0.0..=1.0).contains(&sy_norm) {
                // Out of bounds - transparent
                new_img.put_pixel(x, y, Rgba([0, 0, 0, 0]));
                continue;
            }

            // Map back to pixel coordinates
            let sx = sx_norm * w_f32;
            let sy = sy_norm * h_f32;

            // Sample
            let px = sample_bicubic(img, sx, sy);
            new_img.put_pixel(x, y, px);
        }
    }

    new_img
}

fn apply_crop_rotate(img: &RgbaImage, settings: &CropSettings) -> RgbaImage {
    let mut result = img.clone();

    // Rotation
    if let Some(rot) = settings.rotation {
        if rot.abs() > 1e-5 {
            // Python: result.rotate(-settings.rotation, resample=Image.Resampling.BICUBIC, expand=False)
            // We need to rotate around center, keeping size same.
            // We can reuse the same sampling logic as Geometry!
            // Just map output pixels to input pixels via rotation matrix.

            let angle_rad = -rot.to_radians(); // Python uses degrees
            let cos_a = angle_rad.cos();
            let sin_a = angle_rad.sin();

            let w = result.width();
            let h = result.height();
            let cx = w as f32 / 2.0;
            let cy = h as f32 / 2.0;

            let mut rotated = RgbaImage::new(w, h);

            for y in 0..h {
                for x in 0..w {
                    // Output coords relative to center
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;

                    // Rotate backwards to find source
                    // x_src = x_dst * cos - y_dst * sin
                    // y_src = x_dst * sin + y_dst * cos
                    // (Inverse rotation is -angle)
                    // Wait, we want to map Dest -> Source.
                    // If we rotate image by Theta, then Source = Rotate(-Theta) * Dest
                    // So we use -angle_rad.
                    // But `angle_rad` is already `-rot`.
                    // So we use `-angle_rad` = `rot`.

                    let src_dx = dx * cos_a + dy * sin_a;
                    let src_dy = -dx * sin_a + dy * cos_a;

                    let src_x = src_dx + cx;
                    let src_y = src_dy + cy;

                    rotated.put_pixel(x, y, sample_bicubic(&result, src_x, src_y));
                }
            }
            result = rotated;
        }
    }

    // Explicit Rect Crop
    if let Some(rect) = &settings.rect {
        // Validation: rect should be within 0..1
        let rx = clamp(rect.x, 0.0, 1.0);
        let ry = clamp(rect.y, 0.0, 1.0);
        let rw = clamp(rect.width, 0.0, 1.0 - rx); // Ensure validation fits in bounds
        let rh = clamp(rect.height, 0.0, 1.0 - ry);

        if rw > 0.0 && rh > 0.0 {
            let (w, h) = result.dimensions();
            let x = (w as f32 * rx).round() as u32;
            let y = (h as f32 * ry).round() as u32;
            let new_w = max(1, (w as f32 * rw).round() as u32);
            let new_h = max(1, (h as f32 * rh).round() as u32);

            // Re-check bounds to avoid OOB due to rounding
            let x = x.min(w - 1);
            let y = y.min(h - 1);
            let new_w = new_w.min(w - x);
            let new_h = new_h.min(h - y);

            result = image::imageops::crop_imm(&result, x, y, new_w, new_h).to_image();
        }
    }
    // Fallback to Aspect Ratio Crop if no explicit rect
    else if let Some(ar) = settings.aspect_ratio {
        if ar > 0.0 {
            let (w, h) = result.dimensions();
            let current_ratio = w as f32 / h as f32;

            if (current_ratio - ar).abs() > 1e-3 {
                let (new_w, new_h) = if current_ratio > ar {
                    // Too wide, crop width
                    let nw = (h as f32 * ar).round() as u32;
                    (nw, h)
                } else {
                    // Too tall, crop height
                    let nh = (w as f32 / ar).round() as u32;
                    (w, nh)
                };

                let x = (w - new_w) / 2;
                let y = (h - new_h) / 2;

                result = image::imageops::crop_imm(&result, x, y, new_w, new_h).to_image();
            }
        }
    }

    result
}


fn apply_lens_distortion(img: &RgbaImage, settings: &LensDistortionSettings) -> RgbaImage {
    if settings.k1.abs() < 1e-5 && settings.k2.abs() < 1e-5 {
        return img.clone();
    }

    let width = img.width();
    let height = img.height();
    let mut new_img = RgbaImage::new(width, height);

    let w_f32 = width as f32;
    let h_f32 = height as f32;
    let center_x = 0.5;
    let center_y = 0.5;

    // We iterate over OUTPUT pixels (straight image) and map to INPUT pixels (distorted source)
    for y in 0..height {
        let v = y as f32 / h_f32;
        for x in 0..width {
            let u = x as f32 / w_f32;

            // Distort UV
            // r^2 = x^2 + y^2 (relative to center)
            let rel_x = u - center_x;
            let rel_y = v - center_y;
            let r2 = rel_x * rel_x + rel_y * rel_y;
            let scaling = 1.0 + settings.k1 * r2 + settings.k2 * r2 * r2;

            let src_u = center_x + rel_x * scaling;
            let src_v = center_y + rel_y * scaling;

            // Check bounds
            if src_u < 0.0 || src_u > 1.0 || src_v < 0.0 || src_v > 1.0 {
                // Black
                new_img.put_pixel(x, y, Rgba([0, 0, 0, 255])); // Alpha 255 for opaque black? Or 0? Let's use 0 for transparent border.
                continue;
            }

            let src_x = src_u * w_f32;
            let src_y = src_v * h_f32;
            
            new_img.put_pixel(x, y, sample_bicubic(img, src_x, src_y));
        }
    }

    new_img
}

fn apply_exposure_in_place(img: &mut RgbaImage, settings: &ExposureSettings) {
    let exposure = settings.exposure.unwrap_or(0.0);
    let contrast = settings.contrast.unwrap_or(1.0);
    let highlights = settings.highlights.unwrap_or(0.0);
    let shadows = settings.shadows.unwrap_or(0.0);

    let exposure_factor = 2.0_f32.powf(exposure);

    for pixel in img.pixels_mut() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;
        let a = pixel[3];

        // Exposure
        let mut r = r * exposure_factor;
        let mut g = g * exposure_factor;
        let mut b = b * exposure_factor;

        // Contrast
        if (contrast - 1.0).abs() > 1e-3 {
            r = (r - 0.5) * contrast + 0.5;
            g = (g - 0.5) * contrast + 0.5;
            b = (b - 0.5) * contrast + 0.5;
        }

        // Highlights / Shadows
        if highlights != 0.0 {
            let hm_r = clamp((r - 0.5) * 2.0, 0.0, 1.0);
            let hm_g = clamp((g - 0.5) * 2.0, 0.0, 1.0);
            let hm_b = clamp((b - 0.5) * 2.0, 0.0, 1.0);

            r += hm_r * highlights * 0.5;
            g += hm_g * highlights * 0.5;
            b += hm_b * highlights * 0.5;
        }

        if shadows != 0.0 {
            let sm_r = clamp((0.5 - r) * 2.0, 0.0, 1.0);
            let sm_g = clamp((0.5 - g) * 2.0, 0.0, 1.0);
            let sm_b = clamp((0.5 - b) * 2.0, 0.0, 1.0);

            r += sm_r * shadows * 0.5;
            g += sm_g * shadows * 0.5;
            b += sm_b * shadows * 0.5;
        }

        pixel[0] = clamp_u8(r * 255.0);
        pixel[1] = clamp_u8(g * 255.0);
        pixel[2] = clamp_u8(b * 255.0);
        pixel[3] = a;
    }
}

fn apply_color_balance_in_place(img: &mut RgbaImage, settings: &ColorSettings) {
    let temp = clamp(settings.temperature.unwrap_or(0.0), -1.0, 1.0);
    let tint = clamp(settings.tint.unwrap_or(0.0), -1.0, 1.0);

    if temp.abs() < 1e-5 && tint.abs() < 1e-5 {
        return;
    }

    let temp_r = 1.0 + temp * 0.25;
    let temp_b = 1.0 - temp * 0.25;
    let tint_g = 1.0 - tint * 0.2;
    let tint_rb = 1.0 + tint * 0.1;

    let f_r = temp_r * tint_rb;
    let f_g = tint_g;
    let f_b = temp_b * tint_rb;

    for pixel in img.pixels_mut() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;

        pixel[0] = clamp_u8(r * f_r);
        pixel[1] = clamp_u8(g * f_g);
        pixel[2] = clamp_u8(b * f_b);
    }
}

fn apply_grain_in_place(img: &mut RgbaImage, settings: &GrainSettings) {
    if settings.amount <= 0.0 {
        return;
    }

    let scale = match settings.size.as_str() {
        "medium" => 2,
        "coarse" => 4,
        _ => 1,
    };

    let (width, height) = img.dimensions();
    let noise_w = max(1, width.div_ceil(scale));
    let noise_h = max(1, height.div_ceil(scale));

    let sigma = clamp(settings.amount, 0.0, 1.0) * 25.0;

    // Deterministic RNG
    let seed = settings.seed.unwrap_or(42);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = match Normal::new(0.0, sigma) {
        Ok(n) => n,
        Err(_) => return, // Should not happen given check above, but safe fallback
    };

    // Generate noise buffer
    let mut noise_buffer = vec![0.0f32; (noise_w * noise_h) as usize];
    for x in noise_buffer.iter_mut() {
        *x = normal.sample(&mut rng);
    }

    // Apply noise
    for y in 0..height {
        for x in 0..width {
            let nx = x / scale;
            let ny = y / scale;
            let noise_val = noise_buffer[(ny * noise_w + nx) as usize];

            let pixel = img.get_pixel_mut(x, y);
            let r = pixel[0] as f32 + noise_val;
            let g = pixel[1] as f32 + noise_val;
            let b = pixel[2] as f32 + noise_val;

            pixel[0] = clamp_u8(r);
            pixel[1] = clamp_u8(g);
            pixel[2] = clamp_u8(b);
        }
    }
}

fn apply_denoise_in_place(img: &mut RgbaImage, settings: &DenoiseSettings) {
    if settings.luminance <= 0.0 && settings.color <= 0.0 {
        return;
    }

    let (width, height) = img.dimensions();
    let source = img.clone(); // Need source for reading neighbors

    for y in 0..height {
        for x in 0..width {
            let cx = x as i32;
            let cy = y as i32;

            let center_px = source.get_pixel(x, y);
            let center_rgb = [
                center_px[0] as f32,
                center_px[1] as f32,
                center_px[2] as f32,
            ];
            let center_yuv = rgb_to_yuv(center_rgb);

            let mut final_y = 0.0;
            let mut weight_y = 0.0;

            let mut final_u = 0.0;
            let mut final_v = 0.0;
            let mut weight_c = 0.0;

            // Kernel size 5x5 (radius 2)
            let radius = 2;
            // Sigma parameters (approximate shader logic)
            let sigma_s = 2.0;
            let sigma_r = 0.4 * settings.luminance.max(0.001) * 255.0; // Scale to 0..255 range

            for j in -radius..=radius {
                for i in -radius..=radius {
                    let nx = cx + i;
                    let ny = cy + j;

                    // Clamped lookup
                    let px = if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        source.get_pixel(nx as u32, ny as u32)
                    } else {
                        // Clamp to edge
                        let cnx = nx.max(0).min(width as i32 - 1);
                        let cny = ny.max(0).min(height as i32 - 1);
                        source.get_pixel(cnx as u32, cny as u32)
                    };

                    let rgb = [px[0] as f32, px[1] as f32, px[2] as f32];
                    let yuv = rgb_to_yuv(rgb);

                    let dist_sq = (i * i + j * j) as f32;
                    let w_spatial = (-(dist_sq) / (2.0 * sigma_s * sigma_s)).exp();

                    // Luminance: Bilateral
                    if settings.luminance > 0.0 {
                        let diff = (yuv[0] - center_yuv[0]).abs();
                        let w_range = (-(diff * diff) / (2.0 * sigma_r * sigma_r)).exp();
                        let w = w_spatial * w_range;
                        final_y += yuv[0] * w;
                        weight_y += w;
                    }

                    // Color: Spatial only
                    if settings.color > 0.0 {
                        final_u += yuv[1] * w_spatial;
                        final_v += yuv[2] * w_spatial;
                        weight_c += w_spatial;
                    }
                }
            }

            let out_y = if settings.luminance > 0.0 && weight_y > 0.0 {
                final_y / weight_y
            } else {
                center_yuv[0]
            };

            let out_u = if settings.color > 0.0 && weight_c > 0.0 {
                final_u / weight_c
            } else {
                center_yuv[1]
            };

            let out_v = if settings.color > 0.0 && weight_c > 0.0 {
                final_v / weight_c
            } else {
                center_yuv[2]
            };

            let out_rgb = yuv_to_rgb([out_y, out_u, out_v]);
            let res_px = Rgba([
                clamp_u8(out_rgb[0]),
                clamp_u8(out_rgb[1]),
                clamp_u8(out_rgb[2]),
                center_px[3],
            ]);
            img.put_pixel(x, y, res_px);
        }
    }
}

// Helpers
fn rgb_to_yuv(rgb: [f32; 3]) -> [f32; 3] {
    let y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
    let u = 0.492 * (rgb[2] - y);
    let v = 0.877 * (rgb[0] - y);
    [y, u, v]
}

fn yuv_to_rgb(yuv: [f32; 3]) -> [f32; 3] {
    let y = yuv[0];
    let u = yuv[1];
    let v = yuv[2];
    let r = y + 1.13983 * v;
    let g = y - 0.39465 * u - 0.58060 * v;
    let b = y + 2.03211 * u;
    [r, g, b]
}

fn rgb_to_hsl(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let mut h = 0.0;
    if delta > 0.0 {
        if max == r {
            h = (g - b) / delta % 6.0;
        } else if max == g {
            h = (b - r) / delta + 2.0;
        } else {
            h = (r - g) / delta + 4.0;
        }
        h /= 6.0;
        if h < 0.0 {
            h += 1.0;
        }
    }

    let l = (max + min) / 2.0;
    let s = if delta == 0.0 {
        0.0
    } else {
        delta / (1.0 - (2.0 * l - 1.0).abs())
    };

    [h, s, l]
}

fn hsl_to_rgb(hsl: [f32; 3]) -> [f32; 3] {
    let h = hsl[0];
    let s = hsl[1];
    let l = hsl[2];

    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r, g, b) = if h < 1.0 / 6.0 {
        (c, x, 0.0)
    } else if h < 2.0 / 6.0 {
        (x, c, 0.0)
    } else if h < 3.0 / 6.0 {
        (0.0, c, x)
    } else if h < 4.0 / 6.0 {
        (0.0, x, c)
    } else if h < 5.0 / 6.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [r + m, g + m, b + m]
}

pub(crate) fn apply_hsl_in_place(img: &mut RgbaImage, settings: &HslSettings) {
    // Range centers in Hue (0..1)
    // Red (0), Orange (30), Yellow (60), Green (120), Aqua (180), Blue (240), Purple (270), Magenta (300)
    // Actually standard 8-range is often:
    // Red: 0/360
    // Orange: 30
    // Yellow: 60
    // Green: 120
    // Aqua/Cyan: 180
    // Blue: 240
    // Purple/Violet: 270
    // Magenta: 300

    let ranges = [
        (0.0 / 360.0, &settings.red),
        (30.0 / 360.0, &settings.orange),
        (60.0 / 360.0, &settings.yellow),
        (120.0 / 360.0, &settings.green),
        (180.0 / 360.0, &settings.aqua),
        (240.0 / 360.0, &settings.blue),
        (270.0 / 360.0, &settings.purple),
        (300.0 / 360.0, &settings.magenta),
    ];

    for pixel in img.pixels_mut() {
        let r_in = pixel[0] as f32 / 255.0;
        let g_in = pixel[1] as f32 / 255.0;
        let b_in = pixel[2] as f32 / 255.0;
        let a = pixel[3];

        let mut hsl = rgb_to_hsl([r_in, g_in, b_in]);
        let h = hsl[0];

        let mut delta_h = 0.0;
        let mut delta_s = 0.0;
        let mut delta_l = 0.0;

        for (center, s) in ranges.iter() {
            // Hue distance with wrapping
            let mut dist = (h - center).abs();
            if dist > 0.5 {
                dist = 1.0 - dist;
            }

            // Falloff: using a simple bell curve or linear falloff
            // Width of influence. Typically ranges overlap.
            // Let's use 60 deg (1/6) as total width of influence (30 deg each side).
            let width = 60.0 / 360.0;
            if dist < width {
                // Quadratic falloff for smoothness
                let t = dist / width;
                let weight = 1.0 - t * t * (3.0 - 2.0 * t); // Smoothstep-like falloff

                delta_h += s.hue * weight;
                delta_s += s.saturation * weight;
                delta_l += s.luminance * weight;
            }
        }

        // Apply adjustments
        // Hue: wrap around
        hsl[0] = (hsl[0] + delta_h * (30.0 / 360.0)).rem_euclid(1.0);

        // Saturation/Luminosity: offset + clamp
        hsl[1] = clamp(hsl[1] + delta_s, 0.0, 1.0);
        hsl[2] = clamp(hsl[2] + delta_l, 0.0, 1.0);

        let rgb_out = hsl_to_rgb(hsl);
        pixel[0] = clamp_u8(rgb_out[0] * 255.0);
        pixel[1] = clamp_u8(rgb_out[1] * 255.0);
        pixel[2] = clamp_u8(rgb_out[2] * 255.0);
        pixel[3] = a;
    }
}

pub(crate) fn apply_curves_in_place(img: &mut RgbaImage, settings: &CurvesSettings) {
    let master_lut = settings
        .master
        .as_ref()
        .map(|c| generate_curve_lut(&c.points))
        .unwrap_or_else(|| (0..256).map(|i| i as f32 / 255.0).collect());

    let red_lut = settings
        .red
        .as_ref()
        .map(|c| generate_curve_lut(&c.points))
        .unwrap_or_else(|| (0..256).map(|i| i as f32 / 255.0).collect());

    let green_lut = settings
        .green
        .as_ref()
        .map(|c| generate_curve_lut(&c.points))
        .unwrap_or_else(|| (0..256).map(|i| i as f32 / 255.0).collect());

    let blue_lut = settings
        .blue
        .as_ref()
        .map(|c| generate_curve_lut(&c.points))
        .unwrap_or_else(|| (0..256).map(|i| i as f32 / 255.0).collect());

    // Combine Master and RGB luts
    let mut combined_r = [0u8; 256];
    let mut combined_g = [0u8; 256];
    let mut combined_b = [0u8; 256];

    let intensity = settings.intensity;

    for i in 0..256 {
        let original_f = i as f32 / 255.0;
        // Apply master first, then channel curve
        let m = master_lut[i];
        // For channel curves, we need to interpolate because master might map to non-integer
        let r = sample_lut_linear(&red_lut, m);
        let g = sample_lut_linear(&green_lut, m);
        let b = sample_lut_linear(&blue_lut, m);

        // Mix with original if intensity < 1.0
        let r_final = original_f * (1.0 - intensity) + r * intensity;
        let g_final = original_f * (1.0 - intensity) + g * intensity;
        let b_final = original_f * (1.0 - intensity) + b * intensity;

        combined_r[i] = clamp_u8(r_final * 255.0);
        combined_g[i] = clamp_u8(g_final * 255.0);
        combined_b[i] = clamp_u8(b_final * 255.0);
    }

    for pixel in img.pixels_mut() {
        pixel[0] = combined_r[pixel[0] as usize];
        pixel[1] = combined_g[pixel[1] as usize];
        pixel[2] = combined_b[pixel[2] as usize];
    }
}

fn sample_lut_linear(lut: &[f32], x: f32) -> f32 {
    let x = clamp(x, 0.0, 1.0) * 255.0;
    let i = x.floor() as usize;
    let f = x - i as f32;
    if i >= 255 {
        lut[255]
    } else {
        lut[i] * (1.0 - f) + lut[i + 1] * f
    }
}

pub fn generate_curve_lut(points: &[crate::CurvePoint]) -> Vec<f32> {
    if points.is_empty() {
        return (0..256).map(|i| i as f32 / 255.0).collect();
    }

    // Sort points by X
    let mut sorted_points = points.to_vec();
    sorted_points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

    // Ensure we have 0 and 1
    if sorted_points[0].x > 0.0 {
        sorted_points.insert(
            0,
            crate::CurvePoint {
                x: 0.0,
                y: sorted_points[0].y,
            },
        );
    }
    if sorted_points.last().unwrap().x < 1.0 {
        let last_y = sorted_points.last().unwrap().y;
        sorted_points.push(crate::CurvePoint { x: 1.0, y: last_y });
    }

    if sorted_points.len() < 2 {
        return (0..256).map(|i| i as f32 / 255.0).collect();
    }

    // Natural Cubic Spline
    let n = sorted_points.len();
    let mut h = vec![0.0; n - 1];
    for i in 0..n - 1 {
        h[i] = sorted_points[i + 1].x - sorted_points[i].x;
        if h[i] == 0.0 {
            h[i] = 1e-6;
        } // Avoid division by zero
    }

    let mut alpha = vec![0.0; n - 1];
    for i in 1..n - 1 {
        alpha[i] = 3.0 / h[i] * (sorted_points[i + 1].y - sorted_points[i].y)
            - 3.0 / h[i - 1] * (sorted_points[i].y - sorted_points[i - 1].y);
    }

    let mut l = vec![1.0; n];
    let mut mu = vec![0.0; n];
    let mut z = vec![0.0; n];

    for i in 1..n - 1 {
        l[i] = 2.0 * (sorted_points[i + 1].x - sorted_points[i - 1].x) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];

    for j in (0..n - 1).rev() {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (sorted_points[j + 1].y - sorted_points[j].y) / h[j]
            - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
    }

    let mut lut = Vec::with_capacity(256);
    for i in 0..256 {
        let x = i as f32 / 255.0;
        // Find interval
        let mut idx = n - 2;
        for j in 0..n - 1 {
            if x <= sorted_points[j + 1].x {
                idx = j;
                break;
            }
        }

        let dx = x - sorted_points[idx].x;
        let y = sorted_points[idx].y + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx;
        lut.push(clamp(y, 0.0, 1.0));
    }

    lut
}

pub(crate) fn apply_lut_in_place(
    img: &mut RgbaImage,
    lut_data: &[f32],
    size: u32,
    settings: &Lut3DSettings,
) {
    if settings.intensity <= 0.0 {
        return;
    }

    let size_f = size as f32;
    let max_idx = size_f - 1.0;
    let intensity = clamp(settings.intensity, 0.0, 1.0);

    for pixel in img.pixels_mut() {
        let r_in = pixel[0] as f32 / 255.0;
        let g_in = pixel[1] as f32 / 255.0;
        let b_in = pixel[2] as f32 / 255.0;
        // alpha ignored for LUT

        // Map 0..1 to 0..size-1 (LUT coordinate space)
        let r_c = r_in * max_idx;
        let g_c = g_in * max_idx;
        let b_c = b_in * max_idx;

        // Trilinear interpolation
        // Indices
        let r0 = r_c.floor() as u32;
        let g0 = g_c.floor() as u32;
        let b0 = b_c.floor() as u32;

        let r1 = (r0 + 1).min(size - 1);
        let g1 = (g0 + 1).min(size - 1);
        let b1 = (b0 + 1).min(size - 1);

        // Weights
        let rw = r_c - r0 as f32;
        let gw = g_c - g0 as f32;
        let bw = b_c - b0 as f32;

        let sample = |r, g, b| {
            let idx = ((b * size + g) * size + r) as usize * 3;
            // .cube format is (size*size*size) lines.
            // Standard order: Red changes fastest, then Green, then Blue.
            // i.e. loop B, loop G, loop R.
            // Index = b * size*size + g * size + r.
            if idx + 2 < lut_data.len() {
                [lut_data[idx], lut_data[idx + 1], lut_data[idx + 2]]
            } else {
                [r_in, g_in, b_in] // Fallback (shouldn't happen if size is correct)
            }
        };

        // 8 corners
        let c000 = sample(r0, g0, b0);
        let c100 = sample(r1, g0, b0);
        let c010 = sample(r0, g1, b0);
        let c001 = sample(r0, g0, b1);
        let c110 = sample(r1, g1, b0);
        let c101 = sample(r1, g0, b1);
        let c011 = sample(r0, g1, b1);
        let c111 = sample(r1, g1, b1);

        // Interpolate along R
        let c00 = [
            c000[0] * (1.0 - rw) + c100[0] * rw,
            c000[1] * (1.0 - rw) + c100[1] * rw,
            c000[2] * (1.0 - rw) + c100[2] * rw,
        ];
        let c10 = [
            c010[0] * (1.0 - rw) + c110[0] * rw,
            c010[1] * (1.0 - rw) + c110[1] * rw,
            c010[2] * (1.0 - rw) + c110[2] * rw,
        ];
        let c01 = [
            c001[0] * (1.0 - rw) + c101[0] * rw,
            c001[1] * (1.0 - rw) + c101[1] * rw,
            c001[2] * (1.0 - rw) + c101[2] * rw,
        ];
        let c11 = [
            c011[0] * (1.0 - rw) + c111[0] * rw,
            c011[1] * (1.0 - rw) + c111[1] * rw,
            c011[2] * (1.0 - rw) + c111[2] * rw,
        ];

        // Interpolate along G
        let c0 = [
            c00[0] * (1.0 - gw) + c10[0] * gw,
            c00[1] * (1.0 - gw) + c10[1] * gw,
            c00[2] * (1.0 - gw) + c10[2] * gw,
        ];
        let c1 = [
            c01[0] * (1.0 - gw) + c11[0] * gw,
            c01[1] * (1.0 - gw) + c11[1] * gw,
            c01[2] * (1.0 - gw) + c11[2] * gw,
        ];

        // Interpolate along B
        let c = [
            c0[0] * (1.0 - bw) + c1[0] * bw,
            c0[1] * (1.0 - bw) + c1[1] * bw,
            c0[2] * (1.0 - bw) + c1[2] * bw,
        ];

        // Mix with original (intensity)
        let r_out = r_in * (1.0 - intensity) + c[0] * intensity;
        let g_out = g_in * (1.0 - intensity) + c[1] * intensity;
        let b_out = b_in * (1.0 - intensity) + c[2] * intensity;

        pixel[0] = clamp_u8(r_out * 255.0);
        pixel[1] = clamp_u8(g_out * 255.0);
        pixel[2] = clamp_u8(b_out * 255.0);
    }
}
pub(crate) fn apply_split_toning_in_place(img: &mut RgbaImage, settings: &SplitToningSettings) {
    let shadow_h = settings.shadow_hue / 360.0;
    let shadow_s = clamp(settings.shadow_sat, 0.0, 1.0);
    let highlight_h = settings.highlight_hue / 360.0;
    let highlight_s = clamp(settings.highlight_sat, 0.0, 1.0);
    let balance = clamp(settings.balance, -1.0, 1.0);

    let shadow_rgb = hsl_to_rgb([shadow_h, shadow_s, 0.5]);
    let highlight_rgb = hsl_to_rgb([highlight_h, highlight_s, 0.5]);

    for pixel in img.pixels_mut() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        let lum = 0.299 * r + 0.587 * g + 0.114 * b;

        // Balance shifts the midpoint. Midpoint is usually 0.5.
        // If balance is -1, midpoint is 0. If balance is 1, midpoint is 1.
        let mid = 0.5 + balance * 0.5;

        // Shadow/Highlight masks with smooth transition
        let highlight_mask = clamp((lum - mid) / (1.0 - mid).max(0.01), 0.0, 1.0);
        let shadow_mask = 1.0 - clamp(lum / mid.max(0.01), 0.0, 1.0);

        // Blending (Soft Light behavior)
        fn soft_light(base: f32, blend: f32) -> f32 {
            if blend < 0.5 {
                base - (1.0 - 2.0 * blend) * base * (1.0 - base)
            } else {
                base + (2.0 * blend - 1.0)
                    * (if base <= 0.25 {
                        ((16.0 * base - 12.0) * base + 4.0) * base
                    } else {
                        base.sqrt()
                    } - base)
            }
        }

        let mut r_out = r;
        let mut g_out = g;
        let mut b_out = b;

        // Apply Shadow Tint
        if shadow_s > 0.0 {
            r_out = soft_light(r_out, shadow_rgb[0]) * shadow_mask + r_out * (1.0 - shadow_mask);
            g_out = soft_light(g_out, shadow_rgb[1]) * shadow_mask + g_out * (1.0 - shadow_mask);
            b_out = soft_light(b_out, shadow_rgb[2]) * shadow_mask + b_out * (1.0 - shadow_mask);
        }

        // Apply Highlight Tint
        if highlight_s > 0.0 {
            r_out = soft_light(r_out, highlight_rgb[0]) * highlight_mask
                + r_out * (1.0 - highlight_mask);
            g_out = soft_light(g_out, highlight_rgb[1]) * highlight_mask
                + g_out * (1.0 - highlight_mask);
            b_out = soft_light(b_out, highlight_rgb[2]) * highlight_mask
                + b_out * (1.0 - highlight_mask);
        }

        pixel[0] = clamp_u8(r_out * 255.0);
        pixel[1] = clamp_u8(g_out * 255.0);
        pixel[2] = clamp_u8(b_out * 255.0);
    }
}

pub(crate) fn apply_vignette_in_place(img: &mut RgbaImage, settings: &VignetteSettings) {
    if settings.amount.abs() < 1e-4 {
        return;
    }

    let (width, height) = img.dimensions();
    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;

    // Normalize coordinates to [0, 1] relative to center
    let radius_max_x = center_x;
    let radius_max_y = center_y;

    let roundness = clamp(settings.roundness, -1.0, 1.0);
    // +1.0 = Rectangle, 0.0 = Oval

    let feather = clamp(settings.feather, 0.0, 1.0);
    let amount = clamp(settings.amount, -1.0, 1.0);
    let midpoint = clamp(settings.midpoint, 0.0, 1.0);

    for y in 0..height {
        let dy = (y as f32 - center_y) / radius_max_y; // -1..1
        let dy_abs = dy.abs();
        let dy_sq = dy * dy;

        for x in 0..width {
            let dx = (x as f32 - center_x) / radius_max_x; // -1..1
            let dx_abs = dx.abs();
            let dx_sq = dx * dx;

            // Distance calculation
            let dist_oval = (dx_sq + dy_sq).sqrt();
            let dist_rect = dx_abs.max(dy_abs);

            let dist = if roundness >= 0.0 {
                // Blend Oval -> Rect
                dist_oval * (1.0 - roundness) + dist_rect * roundness
            } else {
                dist_oval
            };

            // Vignette Falloff
            let low = midpoint * (1.0 - 0.9 * feather);
            let high = 1.0 + feather;

            let t = (dist - low) / (high - low).max(1e-5);
            let t = clamp(t, 0.0, 1.0);

            // Smoothstep
            let mask = t * t * (3.0 - 2.0 * t);

            if mask > 0.0 {
                let pixel = img.get_pixel_mut(x, y);

                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;

                let mut r_out = r;
                let mut g_out = g;
                let mut b_out = b;

                if amount < 0.0 {
                    // Darken
                    let factor = 1.0 - mask * amount.abs();
                    r_out *= factor;
                    g_out *= factor;
                    b_out *= factor;
                } else {
                    // Lighten (White vignette)
                    let factor = mask * amount;
                    r_out = r_out + (1.0 - r_out) * factor;
                    g_out = g_out + (1.0 - g_out) * factor;
                    b_out = b_out + (1.0 - b_out) * factor;
                }

                pixel[0] = clamp_u8(r_out * 255.0);
                pixel[1] = clamp_u8(g_out * 255.0);
                pixel[2] = clamp_u8(b_out * 255.0);
            }
        }
    }
}

pub(crate) fn apply_sharpen_in_place(img: &mut RgbaImage, settings: &SharpenSettings) {
    if settings.amount <= 0.0 {
        return;
    }

    let radius = settings.radius.max(0.1);
    // Use standard gaussian blur for sharpening (usually small radius)
    // image::imageops::blur uses Gaussian
    let blurred = image::imageops::blur(img, radius);

    let threshold = settings.threshold;

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let blur_px = blurred.get_pixel(x, y);

        for c in 0..3 {
            let val = pixel[c] as f32;
            let blur_val = blur_px[c] as f32;

            let diff = val - blur_val;
            if diff.abs() * 255.0 >= threshold {
                let new_val = val + diff * settings.amount;
                pixel[c] = clamp_u8(new_val);
            }
        }
    }
}

pub(crate) fn apply_clarity_in_place(img: &mut RgbaImage, settings: &ClaritySettings) {
    if settings.amount.abs() < 1e-4 {
        return;
    }

    let (width, height) = img.dimensions();
    // Large radius for local contrast
    let radius = (width.min(height) as f32 / 64.0).clamp(8.0, 100.0);

    // Use fast box blur approximation (1 pass is often enough for clarity effect,
    // but 3 passes is smoother. Let's do 2 passes for speed/quality balance)
    let blurred = fast_box_blur(img, radius as u32);
    // let blurred = fast_box_blur(&blurred, radius as u32); // Second pass if needed

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let blur_px = blurred.get_pixel(x, y);

        // Apply to RGB roughly or specific channel?
        // Simple Unsharp Mask logic
        for c in 0..3 {
            let val = pixel[c] as f32;
            let blur_val = blur_px[c] as f32;
            let diff = val - blur_val;

            // Clarity typically adds mid-tone contrast using Soft Light or Overlay blending of High Pass
            // Or just straight unsharp mask.
            let new_val = val + diff * settings.amount;
            pixel[c] = clamp_u8(new_val);
        }
    }
}

pub(crate) fn apply_dehaze_in_place(img: &mut RgbaImage, settings: &DehazeSettings) {
    if settings.amount <= 0.0 {
        return;
    }

    // Simple Dehaze: "De-fog"
    // Concept: Haze adds a white veil. Dark regions get brighter.
    // 1. Estimate Veil: Dark Channel.
    // 2. Subtract Veil.

    let amount = clamp(settings.amount, 0.0, 1.0) * 0.95; // Limit max amount

    // Optimisation: We can just do per-pixel operation for speed
    // T(x) = 1 - w * min(r,g,b)
    // J(x) = (I(x) - A) / max(T(x), 0.1) + A
    // Taking A = 1.0 (255) approx.

    for pixel in img.pixels_mut() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        let dark = r.min(g).min(b);
        let transm = 1.0 - amount * dark;
        let transm = transm.max(0.1); // Prevent div by zero

        // Recover: J = (I - 1) / t + 1
        // (Assuming Atmosphere is white)
        let r_new = (r - 1.0) / transm + 1.0;
        let g_new = (g - 1.0) / transm + 1.0;
        let b_new = (b - 1.0) / transm + 1.0;

        pixel[0] = clamp_u8(r_new * 255.0);
        pixel[1] = clamp_u8(g_new * 255.0);
        pixel[2] = clamp_u8(b_new * 255.0);
    }
}

fn fast_box_blur(img: &RgbaImage, radius: u32) -> RgbaImage {
    let (width, height) = img.dimensions();
    if radius == 0 {
        return img.clone();
    }

    let r = radius as i32;

    // Simpler separative blur implementation
    // Creates intermediate buffer
    let mut h_blur = RgbaImage::new(width, height);

    // Horizontal
    for y in 0..height {
        let mut sum_r: u32 = 0;
        let mut sum_g: u32 = 0;
        let mut sum_b: u32 = 0;
        let mut count: u32 = 0;

        // Initial window for x=0
        for i in -r..=r {
            let ix = i.clamp(0, width as i32 - 1) as u32;
            let px = img.get_pixel(ix, y);
            sum_r += px[0] as u32;
            sum_g += px[1] as u32;
            sum_b += px[2] as u32;
            count += 1;
        }

        // This naïve window re-summing is O(R*W). Sliding window is O(W).
        // For radius ~20, O(R*W) is okay but let's try to be efficient.
        // But implementing correct sliding window with boundary clamping is verbose.
        // Let's do O(W) sliding window.

        // Prepare first window [0-r, 0+r].
        // current sum is sum([0-r..0+r]).
        // Actually, clamp means we read pixel 0 for negative indices.
        // sum_r already computed above with clamp.

        h_blur.put_pixel(
            0,
            y,
            Rgba([
                (sum_r / count) as u8,
                (sum_g / count) as u8,
                (sum_b / count) as u8,
                255,
            ]),
        );

        for x in 1..width {
            // Remove x-r-1
            let out_idx = (x as i32 - r - 1).clamp(0, width as i32 - 1) as u32;
            let p_out = img.get_pixel(out_idx, y);
            sum_r -= p_out[0] as u32;
            sum_g -= p_out[1] as u32;
            sum_b -= p_out[2] as u32;

            // Add x+r
            let in_idx = (x as i32 + r).clamp(0, width as i32 - 1) as u32;
            let p_in = img.get_pixel(in_idx, y);
            sum_r += p_in[0] as u32;
            sum_g += p_in[1] as u32;
            sum_b += p_in[2] as u32;

            h_blur.put_pixel(
                x,
                y,
                Rgba([
                    (sum_r / count) as u8,
                    (sum_g / count) as u8,
                    (sum_b / count) as u8,
                    255,
                ]),
            );
        }
    }

    let mut v_blur = RgbaImage::new(width, height);
    // Vertical
    for x in 0..width {
        let mut sum_r: u32 = 0;
        let mut sum_g: u32 = 0;
        let mut sum_b: u32 = 0;
        let mut count: u32 = 0;

        for i in -r..=r {
            let iy = i.clamp(0, height as i32 - 1) as u32;
            let px = h_blur.get_pixel(x, iy);
            sum_r += px[0] as u32;
            sum_g += px[1] as u32;
            sum_b += px[2] as u32;
            count += 1;
        }

        v_blur.put_pixel(
            x,
            0,
            Rgba([
                (sum_r / count) as u8,
                (sum_g / count) as u8,
                (sum_b / count) as u8,
                255,
            ]),
        );

        for y in 1..height {
            let out_idx = (y as i32 - r - 1).clamp(0, height as i32 - 1) as u32;
            let p_out = h_blur.get_pixel(x, out_idx);
            sum_r -= p_out[0] as u32;
            sum_g -= p_out[1] as u32;
            sum_b -= p_out[2] as u32;

            let in_idx = (y as i32 + r).clamp(0, height as i32 - 1) as u32;
            let p_in = h_blur.get_pixel(x, in_idx);
            sum_r += p_in[0] as u32;
            sum_g += p_in[1] as u32;
            sum_b += p_in[2] as u32;

            let p_orig = img.get_pixel(x, y); // Preserve original alpha
            v_blur.put_pixel(
                x,
                y,
                Rgba([
                    (sum_r / count) as u8,
                    (sum_g / count) as u8,
                    (sum_b / count) as u8,
                    p_orig[3],
                ]),
            );
        }
    }

    v_blur
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ClaritySettings, ColorSettings, CropSettings, DehazeSettings, ExposureSettings,
        GeometrySettings, GrainSettings, HslSettings, LensDistortionSettings,
        QuickFixAdjustments, SharpenSettings, SplitToningSettings, VignetteSettings,
    };
    use crate::{ChannelCurve, CropRect, CurvePoint};

    fn create_test_image(width: u32, height: u32, color: [u8; 4]) -> RgbaImage {
        let mut img = RgbaImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = Rgba(color);
        }
        img
    }

    #[test]
    fn test_apply_exposure() {
        let mut img = create_test_image(10, 10, [100, 100, 100, 255]);
        let settings = ExposureSettings {
            exposure: Some(1.0), // +1 stop = double values
            ..Default::default()
        };
        apply_exposure_in_place(&mut img, &settings);

        let px = img.get_pixel(0, 0);
        // 100/255 * 2 = 200/255 approx.
        // 100 * 2 = 200.
        assert!(px[0] >= 199 && px[0] <= 201);
    }

    #[test]
    fn test_apply_color_temperature() {
        let mut img = create_test_image(10, 10, [128, 128, 128, 255]);
        let settings = ColorSettings {
            temperature: Some(0.5), // Warmer: more red, less blue
            ..Default::default()
        };
        apply_color_balance_in_place(&mut img, &settings);

        let px = img.get_pixel(0, 0);
        assert!(px[0] > 128); // Red increased
        assert!(px[2] < 128); // Blue decreased
    }

    #[test]
    fn test_apply_crop() {
        let img = create_test_image(100, 100, [255, 0, 0, 255]);
        let settings = CropSettings {
            aspect_ratio: Some(2.0), // 2:1 ratio
            ..Default::default()
        };
        let res = apply_crop_rotate(&img, &settings);

        assert_eq!(res.width(), 100);
        assert_eq!(res.height(), 50); // Should be cropped to height 50
    }

    #[test]
    fn test_apply_crop_rect() {
        let img = create_test_image(100, 100, [255, 0, 0, 255]);
        let settings = CropSettings {
            rect: Some(CropRect {
                x: 0.25,
                y: 0.25,
                width: 0.5,
                height: 0.5,
            }),
            ..Default::default()
        };
        let res = apply_crop_rotate(&img, &settings);

        assert_eq!(res.width(), 50);
        assert_eq!(res.height(), 50);
    }

    #[test]
    fn test_apply_crop_rect_out_of_bounds() {
        let img = create_test_image(100, 100, [255, 0, 0, 255]);
        let settings = CropSettings {
            rect: Some(CropRect {
                x: 0.8,
                y: 0.8,
                width: 0.5,  // Should be clamped to 0.2
                height: 0.5, // Should be clamped to 0.2
            }),
            ..Default::default()
        };
        let res = apply_crop_rotate(&img, &settings);

        assert_eq!(res.width(), 20); // 100 * 0.2
        assert_eq!(res.height(), 20);
    }

    #[test]
    fn test_apply_geometry_no_panic() {
        let img = create_test_image(50, 50, [0, 255, 0, 255]);
        let settings = GeometrySettings {
            vertical: Some(0.5),
            horizontal: Some(-0.2),
            ..Default::default()
        };
        let res = apply_geometry(&img, &settings);
        assert_eq!(res.width(), 50);
        assert_eq!(res.height(), 50);
    }

    #[test]
    fn test_flip_horizontal() {
        // Create an image where left side is red, right side is blue
        let width = 10;
        let height = 10;
        let mut img = RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                if x < width / 2 {
                    img.put_pixel(x, y, Rgba([255, 0, 0, 255]));
                } else {
                    img.put_pixel(x, y, Rgba([0, 0, 255, 255]));
                }
            }
        }

        let settings = GeometrySettings {
            flip_horizontal: Some(true),
            ..Default::default()
        };
        let res = apply_geometry(&img, &settings);

        // Check if flipped: Left should be blue, Right should be red
        let left_px = res.get_pixel(0, 0);
        let right_px = res.get_pixel(width - 1, 0);

        assert_eq!(left_px[0], 0); // Blue component is at index 2, Red at 0
        assert_eq!(left_px[2], 255);
        assert_eq!(right_px[0], 255);
        assert_eq!(right_px[2], 0);
    }

    #[test]
    fn test_flip_vertical() {
        // Top red, Bottom blue
        let width = 10;
        let height = 10;
        let mut img = RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                if y < height / 2 {
                    img.put_pixel(x, y, Rgba([255, 0, 0, 255]));
                } else {
                    img.put_pixel(x, y, Rgba([0, 0, 255, 255]));
                }
            }
        }

        let settings = GeometrySettings {
            flip_vertical: Some(true),
            ..Default::default()
        };
        let res = apply_geometry(&img, &settings);

        // Top should be blue, Bottom should be red
        let top_px = res.get_pixel(0, 0);
        let bottom_px = res.get_pixel(0, height - 1);

        assert_eq!(top_px[0], 0);
        assert_eq!(top_px[2], 255);
        assert_eq!(bottom_px[0], 255);
        assert_eq!(bottom_px[2], 0);
    }

    #[test]
    fn test_flip_and_geometry() {
        // If we flip AND apply geometry, it should work without panic.
        // Verifying exact pixel values with perspective + flip is hard,
        // so we check structural properties or basic color preservation.
        let img = create_test_image(20, 20, [100, 100, 100, 255]);
        let settings = GeometrySettings {
            vertical: Some(0.1),
            flip_horizontal: Some(true),
            ..Default::default()
        };
        let res = apply_geometry(&img, &settings);
        assert_eq!(res.width(), 20);
        assert_eq!(res.height(), 20);

        // Check center pixel is preserved (approx)
        let cx = 10;
        let cy = 10;
        let px = res.get_pixel(cx, cy);
        assert!(px[0] > 90 && px[0] < 110);
    }
    #[test]
    fn test_compute_histogram() {
        // R=255, G=0, B=100
        let data = vec![255, 0, 100, 255, 255, 0, 100, 255]; // 2 pixels
        let hist = compute_histogram(&data);

        assert_eq!(hist.len(), 768);
        assert_eq!(hist[255], 2); // Red bin 255
        assert_eq!(hist[256 + 0], 2); // Green bin 0
        assert_eq!(hist[512 + 100], 2); // Blue bin 100
        assert_eq!(hist[0], 0); // Red bin 0 should be 0
    }

    #[test]
    fn test_apply_denoise() {
        // Create 3x3 image with a center outlier
        let mut img = create_test_image(3, 3, [100, 100, 100, 255]);
        // Set center pixel to bright
        img.put_pixel(1, 1, Rgba([200, 100, 100, 255]));

        let settings = DenoiseSettings {
            luminance: 1.0, // Strong bilateral
            color: 0.0,
        };
        apply_denoise_in_place(&mut img, &settings);

        let center = img.get_pixel(1, 1);
        // Bilateral should preserve edge if sigma_r is small, but if sigma_r is large (luminance=1.0)
        // it should smooth it towards neighbors.
        // Neighbors are 100. Center is 200.
        // With high luminance denoise, center should drop significantly towards 100.
        // Let's assert it changed significantly
        assert!(center[0] < 195);
    }

    #[test]
    fn test_spline_identity() {
        // Curve points: (0,0), (0.5, 0.5), (1,1)
        let points = vec![
            crate::CurvePoint { x: 0.0, y: 0.0 },
            crate::CurvePoint { x: 0.5, y: 0.5 },
            crate::CurvePoint { x: 1.0, y: 1.0 },
        ];
        let lut = generate_curve_lut(&points);
        assert_eq!(lut.len(), 256);
        // Check few points. Index 128 is ~0.5.
        // 128 / 255 = 0.5019...
        assert!((lut[0] - 0.0).abs() < 1e-5);
        assert!((lut[255] - 1.0).abs() < 1e-5);
        assert!((lut[128] - (128.0 / 255.0)).abs() < 1e-5);
    }

    #[test]
    fn test_spline_nonlinear() {
        // Curve that boosts shadows: (0,0), (0.25, 0.5), (1,1)
        let points = vec![
            crate::CurvePoint { x: 0.0, y: 0.0 },
            crate::CurvePoint { x: 0.25, y: 0.5 },
            crate::CurvePoint { x: 1.0, y: 1.0 },
        ];
        let lut = generate_curve_lut(&points);
        // At 0.25 (index ~64), value should be ~0.5
        let idx = (0.25 * 255.0) as usize;
        assert!(lut[idx] > 0.45 && lut[idx] < 0.55);
    }

    #[test]
    fn test_apply_curves() {
        let mut img = create_test_image(10, 10, [64, 64, 64, 255]);
        // Master curve that darkens: (0,0), (1, 0.5)
        let settings = CurvesSettings {
            intensity: 1.0,
            master: Some(ChannelCurve {
                points: vec![CurvePoint { x: 0.0, y: 0.0 }, CurvePoint { x: 1.0, y: 0.5 }],
            }),
            ..Default::default()
        };
        apply_curves_in_place(&mut img, &settings);
        let px = img.get_pixel(0, 0);
        // 64 -> ~32
        assert!(px[0] < 40);
        assert!(px[1] < 40);
        assert!(px[2] < 40);
    }

    #[test]
    fn test_apply_curves_intensity() {
        let mut img = create_test_image(1, 1, [100, 100, 100, 255]);
        // Curve that would map 100 to ~200
        let settings = CurvesSettings {
            intensity: 0.5,
            master: Some(ChannelCurve {
                points: vec![
                    CurvePoint { x: 0.0, y: 0.0 },
                    CurvePoint { x: 1.0, y: 1.0 },
                    CurvePoint { x: 0.392, y: 0.784 }, // 100/255 -> 200/255 approx
                ],
            }),
            ..Default::default()
        };
        apply_curves_in_place(&mut img, &settings);
        let px = img.get_pixel(0, 0);
        // Original: 100, Full curve: 200. Intensity 0.5 -> 150.
        assert!(px[0] > 140 && px[0] < 160);
    }

    #[test]
    fn test_rgb_hsl_conversion() {
        let rgb = [1.0, 0.0, 0.0]; // Red
        let hsl = rgb_to_hsl(rgb);
        assert!((hsl[0] - 0.0).abs() < 1e-5);
        assert!((hsl[1] - 1.0).abs() < 1e-5);
        assert!((hsl[2] - 0.5).abs() < 1e-5);

        let rgb_back = hsl_to_rgb(hsl);
        assert!((rgb_back[0] - 1.0).abs() < 1e-5);
        assert!((rgb_back[1] - 0.0).abs() < 1e-5);
        assert!((rgb_back[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_hsl_red_saturation() {
        let mut img = create_test_image(1, 1, [255, 0, 0, 255]); // Pure Red
        let mut settings = HslSettings::default();
        settings.red.saturation = 0.5; // Boost saturation (even though already 1.0, test rounding/clamping)

        // Let's test desaturation instead to be sure
        settings.red.saturation = -1.0;
        apply_hsl_in_place(&mut img, &settings);
        let px = img.get_pixel(0, 0);
        // Red is at hue 0. Should be gray now.
        assert!(px[0] == px[1] && px[1] == px[2]);
        assert!(px[0] > 120 && px[0] < 130); // 0.5 luminance -> ~127
    }

    #[test]
    fn test_apply_hsl_green_luminance() {
        let mut img = create_test_image(1, 1, [0, 255, 0, 255]); // Pure Green
        let mut settings = HslSettings::default();
        settings.green.luminance = -0.5; // Darken
        apply_hsl_in_place(&mut img, &settings);
        let px = img.get_pixel(0, 0);
        // Should be darker green
        assert!(px[1] < 200);
        assert!(px[0] == 0);
        assert!(px[2] == 0);
    }
    #[test]
    fn test_apply_split_toning() {
        let mut img = create_test_image(1, 1, [128, 128, 128, 255]); // Mid gray
        let settings = SplitToningSettings {
            shadow_hue: 200.0, // Teal
            shadow_sat: 0.5,
            highlight_hue: 30.0, // Orange
            highlight_sat: 0.5,
            balance: 0.0,
        };
        apply_split_toning_in_place(&mut img, &settings);
        let px = img.get_pixel(0, 0);

        // Mid gray should be affected by both (or neutral if masks are 0 at midpoint,
        // but our masks overlap mid).
        // Let's just verify it changed.
        assert!(px[0] != 128 || px[1] != 128 || px[2] != 128);
    }

    #[test]
    fn test_apply_vignette() {
        // Create white image
        let width = 100;
        let height = 100;
        let mut img = create_test_image(width, height, [255, 255, 255, 255]);

        let settings = VignetteSettings {
            amount: -1.0, // Full darkness
            midpoint: 0.5,
            roundness: 0.0, // Oval
            feather: 0.5,
        };

        apply_vignette_in_place(&mut img, &settings);

        let center = img.get_pixel(width / 2, height / 2);
        let corner = img.get_pixel(0, 0);

        // Center should be largely unaffected (white)
        assert!(center[0] > 240);

        // Corner should be significantly darkened
        assert!(corner[0] < 100);

        // Verify output is grayish (keeps saturation balance on white image)
        assert_eq!(corner[0], corner[1]);
        assert_eq!(corner[1], corner[2]);
    }
    #[test]
    fn test_apply_sharpen() {
        // Create a blurry image or just an edge.
        // Let's create a step function (edge).
        let width = 10;
        let height = 10;
        let mut img = create_test_image(width, height, [100, 100, 100, 255]);
        // Right half bright
        for y in 0..height {
            for x in width / 2..width {
                img.put_pixel(x, y, Rgba([200, 200, 200, 255]));
            }
        }

        let settings = SharpenSettings {
            amount: 1.0,
            radius: 1.0,
            threshold: 0.0,
        };

        // Sharpening should increase contrast at the edge.
        // The pixel just to the left of the edge (x=4) should get darker (100 -> <100)
        // The pixel just to the right of the edge (x=5) should get brighter (200 -> >200)

        // Before: x=4 is 100. x=5 is 200.
        apply_sharpen_in_place(&mut img, &settings);

        let p_left = img.get_pixel(4, 5);
        let p_right = img.get_pixel(5, 5);

        assert!(p_left[0] < 100);
        assert!(p_right[0] > 200);
    }

    #[test]
    fn test_apply_lens_distortion() {
        let width = 100;
        let height = 100;
        let mut img = create_test_image(width, height, [255, 255, 255, 255]);
        
        // Draw a straight line at x=50 (width 3: 49, 50, 51)
        for y in 0..height {
            img.put_pixel(49, y, Rgba([0, 0, 0, 255]));
            img.put_pixel(50, y, Rgba([0, 0, 0, 255]));
            img.put_pixel(51, y, Rgba([0, 0, 0, 255]));
        }

        let settings = LensDistortionSettings {
            k1: 0.5, 
            k2: 0.0,
        };

        let res = apply_lens_distortion(&img, &settings);
        
        // Center should stay at 50 (cx=0.5 -> x=50).
        let center_px = res.get_pixel(50, 50);
        assert_eq!(center_px[0], 0); // Should be black

        // Point not at center (x=25) should look different (shifted).
        // Vertical line at x=25.
        let mut img_vert = create_test_image(width, height, [255, 255, 255, 255]);
        for y in 0..height {
            img_vert.put_pixel(25, y, Rgba([0, 0, 0, 255]));
        }
        
        let res_vert = apply_lens_distortion(&img_vert, &settings);
        
        // At center y=50, the line shifts inwards?
        // u=0.25 maps to src_u=0.242.
        // So Res(25) looks at Src(24). Src(24) is white.
        // Src(25) is black.
        // So Res(25) is white.
        // The black line effectively moved to x=26 (where src_u maps to 0.25).
        assert!(res_vert.get_pixel(25, 50)[0] > 100); // Moved away
    }
    #[test]
    fn test_apply_clarity() {
        // Clarity boosts local contrast.
        // Similar to sharpen but large radius.
        let width = 20;
        let height = 20;
        let mut img = create_test_image(width, height, [100, 100, 100, 255]);
        // Center bright spot
        for y in 5..15 {
            for x in 5..15 {
                img.put_pixel(x, y, Rgba([150, 150, 150, 255]));
            }
        }

        let settings = ClaritySettings { amount: 0.5 };
        apply_clarity_in_place(&mut img, &settings);

        // Center pixel (10,10) is 150. Surround is 100.
        // Blur will average them (e.g. 125).
        // Difference = 150 - 125 = +25.
        // New = 150 + 25*0.5 = 162.5.
        // So center should get brighter.
        let center = img.get_pixel(10, 10);
        assert!(center[0] > 150);

        // Verify it doesn't effect uniform area much?
        // Fast box blur might have edge effects but internal uniform area should be close to uniform blur = pixel.
        // so diff = 0.
        // But our test image is small, so radius might cover everything.
    }

    #[test]
    fn test_apply_dehaze() {
        // Create an image with haze (elevated dark channel)
        // e.g. RGB(200, 200, 210) - kinda whitish/grayish
        let mut img = create_test_image(1, 1, [200, 200, 210, 255]);

        let settings = DehazeSettings { amount: 0.5 };
        apply_dehaze_in_place(&mut img, &settings);

        let px = img.get_pixel(0, 0);

        // Dark channel of [200, 200, 210] is 200 (approx 0.78).
        // Transm = 1.0 - 0.5 * 0.78 = 0.61.
        // Recover: (0.78 - 1.0) / 0.61 + 1.0 = -0.22 / 0.61 + 1.0 = -0.36 + 1.0 = 0.64 -> 163.
        // So values should drop (contrast stretching downward).
        assert!(px[0] < 200);
        assert!(px[1] < 200);
        assert!(px[2] < 210);
    }
}
