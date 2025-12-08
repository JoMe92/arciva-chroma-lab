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

/// Point struct for geometry calculations
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

/// Calculate the 4 corners of the source image that correspond to the output unit square
/// based on the tilt settings.
/// Returns [TopLeft, TopRight, BottomRight, BottomLeft]
/// Coordinates are normalized 0..1 relative to the source image.
pub fn calculate_distortion_state(vertical: f32, horizontal: f32) -> [Point; 4] {
    let v = vertical.clamp(-1.0, 1.0);
    let h = horizontal.clamp(-1.0, 1.0);

    // Reuse logic from operations.rs but normalized
    let max_offset = 0.25;

    let top_inset = v * max_offset;
    let bottom_inset = -v * max_offset;
    let left_y = h * max_offset;
    let right_y = -h * max_offset;

    // TL
    let ul = Point {
        x: 0.0 + top_inset,
        y: 0.0 + left_y,
    };
    // BL
    let ll = Point {
        x: 0.0 + bottom_inset,
        y: 1.0 - left_y,
    };
    // BR
    let lr = Point {
        x: 1.0 - bottom_inset,
        y: 1.0 - right_y,
    };
    // TR
    let ur = Point {
        x: 1.0 - top_inset,
        y: 0.0 + right_y,
    };

    [ul, ur, lr, ll]
}

/// Calculate the 3x3 Homography Matrix that maps a point in the destination (unit square)
/// to the source distorted quad.
///
/// Mapping:
/// (0,0) -> corners[0] (TL)
/// (1,0) -> corners[1] (TR)
/// (1,1) -> corners[2] (BR)
/// (0,1) -> corners[3] (BL)
///
/// Returns a flattened 3x3 matrix (row-major).
pub fn calculate_homography_from_unit_square(corners: &[Point; 4]) -> [f32; 9] {
    // Source points (Destination unit square)
    // u, v
    let src = [
        Point { x: 0.0, y: 0.0 }, // TL
        Point { x: 1.0, y: 0.0 }, // TR
        Point { x: 1.0, y: 1.0 }, // BR
        Point { x: 0.0, y: 1.0 }, // BL
    ];

    // Destination points (Source Quad) -> We map FROM output TO input
    // x, y
    let dst = corners;

    compute_homography(&src, dst)
}

/// Solves for H such that H * src = dst
/// Where src and dst are 4 points.
fn compute_homography(src: &[Point; 4], dst: &[Point; 4]) -> [f32; 9] {
    // Basic DLT (Direct Linear Transformation)
    // For each point correspondence:
    // x' = (h11 x + h12 y + h13) / (h31 x + h32 y + h33)
    // y' = (h21 x + h22 y + h23) / (h31 x + h32 y + h33)
    //
    // Rearranging:
    // x (h31 x' - h11) + y (h32 x' - h12) + (h33 x' - h13) = 0
    // x (h31 y' - h21) + y (h32 y' - h22) + (h33 y' - h23) = 0
    //
    // We assume h33 = 1.0. The system is 8x8.
    // matrix A * h = b
    // where h = [h11, h12, h13, h21, h22, h23, h31, h32]^T
    //
    // A has 8 rows (2 per point).
    // Row 2i:   [-x, -y, -1,  0,  0,  0, x*x', y*x']
    // Row 2i+1: [ 0,  0,  0, -x, -y, -1, x*y', y*y']
    //
    // b has 8 elements.
    // b[2i]   = -x'
    // b[2i+1] = -y'

    let mut a = [[0.0f32; 8]; 8];
    let mut b = [0.0f32; 8];

    for i in 0..4 {
        let x = src[i].x;
        let y = src[i].y;
        let xp = dst[i].x;
        let yp = dst[i].y;

        // Row 1
        a[2 * i][0] = -x;
        a[2 * i][1] = -y;
        a[2 * i][2] = -1.0;
        a[2 * i][3] = 0.0;
        a[2 * i][4] = 0.0;
        a[2 * i][5] = 0.0;
        a[2 * i][6] = x * xp;
        a[2 * i][7] = y * xp;
        b[2 * i] = -xp;

        // Row 2
        a[2 * i + 1][0] = 0.0;
        a[2 * i + 1][1] = 0.0;
        a[2 * i + 1][2] = 0.0;
        a[2 * i + 1][3] = -x;
        a[2 * i + 1][4] = -y;
        a[2 * i + 1][5] = -1.0;
        a[2 * i + 1][6] = x * yp;
        a[2 * i + 1][7] = y * yp;
        b[2 * i + 1] = -yp;
    }

    // Solve Ax = b using Gaussian elimination
    let h = solve_gaussian(a, b);

    [
        h[0], h[1], h[2],
        h[3], h[4], h[5],
        h[6], h[7], 1.0
    ]
}

fn solve_gaussian(mut a: [[f32; 8]; 8], mut b: [f32; 8]) -> [f32; 8] {
    let n = 8;

    for i in 0..n {
        // Pivot
        let mut max_el = a[i][i].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[k][i].abs() > max_el {
                max_el = a[k][i].abs();
                max_row = k;
            }
        }

        // Swap
        for k in i..n {
            let tmp = a[max_row][k];
            a[max_row][k] = a[i][k];
            a[i][k] = tmp;
        }
        let tmp = b[max_row];
        b[max_row] = b[i];
        b[i] = tmp;

        // Eliminate
        if a[i][i].abs() < 1e-6 {
            // Singular or near-singular
            continue;
        }

        for k in (i + 1)..n {
            let c = -a[k][i] / a[i][i];
            for j in i..n {
                if i == j {
                    a[k][j] = 0.0;
                } else {
                    a[k][j] += c * a[i][j];
                }
            }
            b[k] += c * b[i];
        }
    }

    // Back substitution
    let mut x = [0.0; 8];
    for i in (0..n).rev() {
        if a[i][i].abs() < 1e-6 {
            x[i] = 0.0;
            continue;
        }
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += a[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / a[i][i];
    }
    x
}

