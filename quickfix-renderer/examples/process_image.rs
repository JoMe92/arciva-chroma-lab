use quickfix_renderer::{operations, QuickFixAdjustments};
use std::env;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <input_image> <output_image> <adjustments_json>", args[0]);
        eprintln!(r#"Example: {} input.png output.png '{{"exposure": {{"exposure": 1.0}}}}'"#, args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let json_str = &args[3];

    println!("Loading image from {}", input_path);
    let img = image::open(input_path)?.to_rgba8();
    let (width, height) = img.dimensions();
    let mut data = img.into_raw();

    println!("Parsing adjustments...");
    let adjustments: QuickFixAdjustments = serde_json::from_str(json_str)?;
    println!("Adjustments: {:?}", adjustments);

    println!("Processing frame ({}x{})...", width, height);
    let (processed_data, new_w, new_h) = operations::process_frame_internal(&mut data, width, height, &adjustments)
        .map_err(|e| format!("Processing error: {}", e))?;

    println!("Saving to {} ({}x{})", output_path, new_w, new_h);
    let output_img = image::RgbaImage::from_raw(new_w, new_h, processed_data)
        .ok_or("Failed to create output buffer")?;
    
    // Note: If crop/rotate changed dimensions, process_frame_internal returns the raw buffer.
    // But process_frame_internal currently returns Vec<u8>.
    // Wait, if dimensions change (crop), the output buffer size won't match input width/height.
    // My implementation of `process_frame_internal` returns `img.into_raw()`.
    // But `img` inside `process_frame_internal` might have been resized by `apply_crop_rotate`.
    // The `process_frame_internal` signature returns `Result<Vec<u8>, String>`.
    // It does NOT return the new dimensions.
    // This is a flaw in my current implementation if I want to support resizing!
    // The prompt said: "process_frame(buffer, width, height, adjustments)".
    // And "returns a processed buffer".
    // If the size changes, the caller needs to know the new size.
    // For "Quick Fix" preview, maybe size doesn't change?
    // "Crop/rotate (bicubic-like resampling)"
    // "CropSettings" has "aspect_ratio".
    // If I crop, size changes.
    // I should probably update `process_frame_internal` to return `(Vec<u8>, u32, u32)` or similar.
    // But for this example, let's assume size might change and we need to handle it.
    // BUT, `process_frame_internal` signature is `-> Result<Vec<u8>, String>`.
    // I can't know the new dimensions from just `Vec<u8>` without metadata.
    
    // Let's check `process_frame_internal` implementation again.
    // It returns `img.into_raw()`. `img` is the final `RgbaImage`.
    // So the vector has the correct data.
    // But I don't know the width/height to reconstruct it!
    
    // I MUST fix `process_frame_internal` to return dimensions if I want this to work for crops.
    // However, for the immediate user request "how to test", I can fix this now.
    
    // Let's modify `process_frame_internal` to return `(Vec<u8>, u32, u32)`.
    // And update `process_frame` (WASM) to return... wait, WASM `process_frame` returns `Vec<u8>`.
    // In JS, `Vec<u8>` becomes a `Uint8Array`.
    // If the size changes, the JS side also needs to know.
    // Usually for a "process_frame" that modifies in place or returns buffer, if size changes, we need a struct return.
    
    // For now, to unblock the user, I will assume NO resize for the example, OR I will fix the implementation.
    // Given "Quick Fix" usually implies "adjustments", maybe crop is handled differently?
    // "Crop/rotate ... optional centered aspect-ratio crop".
    // If I crop, pixels are removed.
    
    // I will update `process_frame_internal` to return `(Vec<u8>, u32, u32)`.
    // This is a necessary fix.
    
    // But first, let's write the example assuming I'll fix it.
    
    output_img.save(output_path)?;
    println!("Done!");

    Ok(())
}
