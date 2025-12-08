use crate::{
    ColorSettings, CropSettings, ExposureSettings, GeometrySettings, GrainSettings,
    QuickFixAdjustments,
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

pub fn process_frame_internal(
    data: &mut [u8],
    width: u32,
    height: u32,
    adjustments: &QuickFixAdjustments,
) -> Result<(Vec<u8>, u32, u32), String> {
    // Convert raw bytes to ImageBuffer
    let mut img: RgbaImage =
        ImageBuffer::from_raw(width, height, data.to_vec()).ok_or("Invalid buffer size")?;

    // 1. Geometry
    if let Some(geo) = &adjustments.geometry {
        img = apply_geometry(&img, geo);
    }

    // 2. Crop/Rotate
    if let Some(crop) = &adjustments.crop {
        img = apply_crop_rotate(&img, crop);
    }

    // 3. Exposure
    if let Some(exp) = &adjustments.exposure {
        apply_exposure_in_place(&mut img, exp);
    }

    // 4. Color
    if let Some(col) = &adjustments.color {
        apply_color_balance_in_place(&mut img, col);
    }

    // 5. Grain
    if let Some(grain) = &adjustments.grain {
        apply_grain_in_place(&mut img, grain);
    }

    let (w, h) = img.dimensions();
    Ok((img.into_raw(), w, h))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColorSettings, CropRect, CropSettings, ExposureSettings, GeometrySettings};

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
}
