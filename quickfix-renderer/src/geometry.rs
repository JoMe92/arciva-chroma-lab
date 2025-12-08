/// Calculates the largest axis-aligned rectangle with the same aspect ratio as the original
/// that fits inside the rotated original rectangle.
///
/// * `w` - Original width
/// * `h` - Original height
/// * `angle_rad` - Rotation angle in radians
///
/// Returns (new_width, new_height)
pub fn calculate_largest_interior_rect(w: f32, h: f32, angle_rad: f32) -> (f32, f32) {
    if w <= 0.0 || h <= 0.0 {
        return (0.0, 0.0);
    }

    let angle = angle_rad.abs();

    // If angle is 0, return original
    if angle < 1e-5 {
        return (w, h);
    }

    // If angle is 90 degrees (PI/2), we can't fit the original aspect ratio
    // without scaling down to 0 unless it's a square, but typically we want the largest *box*.
    // However, the standard formula for "largest rectangle with SAME ASPECT RATIO" is:
    // Scale factor k = (w*h) / (w*h*cos(a) + (w^2 + h^2)*sin(a)) ??
    // Actually, there is a simpler geometric derivation.
    //
    // Let the original aspect ratio be aspect = w / h.
    // We want to find a rectangle of size (k*w, k*h) that fits inside the rotated rectangle.
    //
    // For a rectangle (w, h) rotated by angle A:
    // The width of the bounding box is W_bb = w * cos(A) + h * sin(A)
    // The height of the bounding box is H_bb = w * sin(A) + h * cos(A)
    //
    // But we are looking for the inner rectangle.

    // A known formula for the scale factor 'k' to fit a rectangle (w, h) inside itself rotated by angle alpha:
    // where aspect ratio is preserved.
    //
    // k = w*h / (w*h*cos(alpha) + w*w*sin(alpha))  ... IF w <= h ? No, it depends on which side hits.

    // Let's use the symmetric formula which handles arbitrary w, h.
    // Adapted from: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

    // Ensure we are working with 0 <= angle <= 90 for symmetry (rectangle symmetry)
    // The input angle might be anything, but for a rectangle, result is periodic for 180, and symmetric for 90?
    // Actually, for aspect-preserving, we just care about the perturbation from 0.
    // Let's normalize angle to [0, PI/2].
    // If angle > PI/2, we map it back.
    // A rotated rect by 100 deg is same shape as rotated by 80 deg (relative to bounding box),
    // but the aspect ratio constraint is relative to the *unrotated* image.
    // If we rotate a 2:1 image by 90 degrees, we have a 1:2 oriented image.
    // We cannot fit a 2:1 image inside it typically unless scaled very small.
    //
    // Wait, the user wants "straighten the horizon". This usually means small angles (< 45 deg).
    // If the user rotates 90 degrees, they probably intend to rotate the aspect ratio too?
    // "Black/transparent triangles" implies we just want to cut out the empty space.
    //
    // The "Largest Interior Rectangle" usually implies creating an axis-aligned rectangle
    // inside the rotated bounds.
    //
    // The user's prompt said: "Implement 'Largest Interior Rectangle' calculation for a rotated rectangle."
    // And "Integration: Provide a utility function get_opaque_crop(rotation, aspect)".
    //
    // If we look at standard implementations for "straighten":
    // We strictly reduce the scale to cut corners.

    // Formula from a reliable source for "Largest Rectangle of ratio w:h inside rotated w:h":

    let sin_a = angle.sin();
    let cos_a = angle.cos();

    let num1 = w;
    let den1 = w * cos_a + h * sin_a;

    let num2 = h;
    let den2 = w * sin_a + h * cos_a;

    let k1 = num1 / den1;
    let k2 = num2 / den2;

    let k = k1.min(k2);

    (w * k, h * k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_zero_rotation() {
        let (w, h) = calculate_largest_interior_rect(100.0, 50.0, 0.0);
        assert!((w - 100.0).abs() < 1e-4);
        assert!((h - 50.0).abs() < 1e-4);
    }

    #[test]
    fn test_90_degree_rotation() {
        // If we rotate 100x50 by 90deg, we have a 50x100 vertical rect.
        // We want to fit a 2:1 rect inside 50x100.
        // interior width = new_w, interior height = new_w/2.
        // Box is width 50, height 100.
        // max new_w = 50. -> height = 25.
        // Verify formula:
        // k1 = 100 / (100*0 + 50*1) = 2.0
        // k2 = 50 / (100*1 + 50*0) = 0.5
        // k = 0.5
        // new_w = 100 * 0.5 = 50.
        // new_h = 50 * 0.5 = 25.
        // Correct.

        let (w, h) = calculate_largest_interior_rect(100.0, 50.0, PI / 2.0);
        assert!((w - 50.0).abs() < 1e-4);
        assert!((h - 25.0).abs() < 1e-4);
    }

    #[test]
    fn test_small_rotation() {
        // Rotate square 100x100 by small angle.
        // k = min( 100/(100c+100s), 100/(100s+100c) )
        //   = 1/(c+s) in both cases.
        // 45 degrees: c=s=0.707
        // k = 1 / 1.414 = 0.707.
        // w = 70.7.
        // Max square in diamonds is side * sin(45)? No, side / sqrt(2) = side * 0.707.
        // Correct.

        let (w, _h) = calculate_largest_interior_rect(100.0, 100.0, PI / 4.0);
        assert!((w - 70.7106).abs() < 1e-3);
    }
}
