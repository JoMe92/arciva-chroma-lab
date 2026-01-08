use std::str::FromStr;
use roxmltree::Document;

/// Parses a LUT file based on extension/format hint.
pub fn parse_lut(content: &str, ext: &str) -> Result<(Vec<f32>, u32), String> {
    match ext.to_lowercase().as_str() {
        "cube" => parse_cube_file(content),
        "3dl" => parse_3dl_file(content),
        "xmp" => parse_xmp_file(content),
        _ => Err(format!("Unsupported LUT format: {}", ext)),
    }
}

pub fn parse_cube_file(content: &str) -> Result<(Vec<f32>, u32), String> {
    let mut size = 0u32;
    let mut data = Vec::new();
    let mut data_started = false;

    for line in content.lines() {
        let line = line.trim();
        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if !data_started {
            if line.starts_with("LUT_3D_SIZE") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(s) = parts[1].parse::<u32>() {
                        size = s;
                        data.reserve((size * size * size * 3) as usize);
                    } else {
                        return Err("Invalid size format".to_string());
                    }
                }
            } else if line.starts_with("LUT_1D_SIZE") {
                return Err("1D LUTs are not supported".to_string());
            } else if line.starts_with("TITLE") || line.starts_with("DOMAIN_") {
                continue;
            } else {
                if size > 0 {
                    data_started = true;
                    parse_data_line(line, &mut data)?;
                } else if is_numeric_line(line) {
                    return Err("Found data before LUT_3D_SIZE".to_string());
                }
            }
        } else {
            parse_data_line(line, &mut data)?;
        }
    }

    if size == 0 {
        return Err("LUT_3D_SIZE not found".to_string());
    }
    validate_data_len(&data, size)
}

fn parse_3dl_file(content: &str) -> Result<(Vec<f32>, u32), String> {
    let mut size = 0u32;
    let mut data = Vec::new();
    let mut data_started = false;
    let mut grid_points_read = false;

    // 3DL usually starts with grid points. multiple lines or one line.
    // Example:
    // 0 64 128 ... 1023
    // followed by RGB data
    
    // We need to detect the grid definition to determine size.
    // Assuming simple format: valid grid definition line has N integers.
    
    let mut lines = content.lines().filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'));
    
    // Peek first line to check if it's metadata or grid
    while let Some(line) = lines.next() {
        let line = line.trim();
        
        if !grid_points_read {
            // Check if it's a grid line. Just a list of numbers.
            if is_numeric_line(line) {
                let nums: Vec<&str> = line.split_whitespace().collect();
                // If it's the grid definition, the number of items is the size.
                // But wait, 3DL might have 3 rows of grid points?
                // Or just one row if cubic?
                // Let's assume if we read a line of >3 integers, it's a grid def.
                if nums.len() > 3 {
                    size = nums.len() as u32;
                    // If we found one, we might find 2 more (separate channels), or just this one.
                    // We assume cubic grid, so we just take this size.
                    // However, we need to consume potential next 2 grid lines if they exist.
                    // Implementation detail: we could just read untill valid RGB triplets appear.
                    grid_points_read = true;
                    
                    // Pre-allocate
                    data.reserve((size * size * size * 3) as usize);
                    continue; 
                } else if nums.len() == 3 {
                    // This looks like data. If we haven't seen grid points yet, that's an issue unless inferred?
                    // Some formats might not list grid points if implicit?
                    // But 3DL usually lists them.
                    // If we encounter data first, error.
                    return Err("Found data before grid definition in .3dl".to_string());
                }
            }
            // Skip other metadata like "shaper" or weird headers?
        } else {
            // Grid read, now read data
            // But we might encounter 2 more grid lines if they listed R, G, B separately.
            // If the line has `size` items again, it's a grid line.
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == size as usize {
                // Another grid line, skip
                continue;
            } else if parts.len() == 3 {
                 // Data line
                 parse_data_line(line, &mut data)?;
            } else {
                // Unknown?
            }
        }
    }
    
    if size == 0 {
         return Err("Could not determine grid size from .3dl".to_string());
    }

    // 3DL data range is often 0..1023 (10bit) or 0..4095 or 0..65535?
    // We need to normalize to 0..1.
    // Check max value?
    let max_val = data.iter().fold(0.0f32, |a, &b| a.max(b));
    if max_val > 1.0 {
        // Assume 10-bit (1023) or 12-bit?
        // Usually 3DL is 10-bit integer based often.
        // Let's normalize by max_val or a standard?
        // If max is > 1.0, we normalize.
        // Common steps are 1023 (10-bit).
        // Let's normalize by 1023.0 if it looks like 10-bit?
        // Or if max > 255.0?
        // Safer: normalize by the nearest power of 2 minus 1? or just max_val found?
        // If we use max_val found, we might clip whites if the LUT doesn't reach 100% white?
        // 3DL "mesh" usually defines the input range (0..1023).
        // It is safest to assume standard 3DL is 10-bit output usually.
        // Let's try 1023.0 normalization factor if max > 1.0.
        // (Autodesk/Flame ususally use 10-bit Log or Lin).
        
        let scale = if max_val > 60000.0 {
            65535.0
        } else if max_val > 3000.0 {
            4095.0 // 12 bit
        } else if max_val > 1.0 {
            1023.0 // 10 bit
        } else {
            1.0
        };
        
        for v in &mut data {
            *v /= scale;
        }
    }

    validate_data_len(&data, size)
}

fn parse_xmp_file(content: &str) -> Result<(Vec<f32>, u32), String> {
    let doc = Document::parse(content).map_err(|e| format!("XML Parse error: {}", e))?;
    
    // Look for crs:LookTable or similar.
    // Namespaces are tricky in roxmltree lookup without explicit full names?
    // We can just iterate nodes and check logical names.
    
    // We search for a node name ending in "LookTable" (crs:LookTable usually)
    // There might be <crs:LookTable> ... hex data ... </crs:LookTable>
    // Or inside Description.
    
    let look_table_node = doc.descendants().find(|n| {
        n.tag_name().name().ends_with("LookTable") || n.tag_name().name().ends_with("Table") // broad check?
    });
    
    if let Some(node) = look_table_node {
        // Text content is the LUT data. usually hex blob.
        let text = node.text().unwrap_or("").trim();
        // Check parsing format.
        // 32 chars hex string = 16 bytes = 128 bit md5? no.
        // usually it's a huge blob.
        // If it's hex, 2 chars per byte?
        // Or base64?
        // Adobe XMP LookTable is often MD5 if it refers to external?
        // If it's "RGB Table", it's raw bytes?
        
        // Actually, often crs:LookTable is NOT embedded full data in all profiles.
        // But in "Enhanced Profiles" (xmp files that serve as LUTs), it IS the table.
        // And it's often binary-as-hex.
        
        // Let's assume hex string of 32-bit float or 8-bit int?
        // Usually RGB floats?
        // Wait, if it's hex, one float is 8 hex chars?
        
        // Let's assume we can't fully support all XMP variants blindly.
        // But let's support plain text numbers if it is list of numbers?
        
        if text.contains(|c: char| c.is_whitespace() && !c.is_control()) {
             // If it has spaces, maybe just numbers?
             // <rs:Table>0.000 0.100 ...</rs:Table> ? (Rare for XMP)
             let mut data = Vec::new();
             for part in text.split_whitespace() {
                 if let Ok(v) = f32::from_str(part) {
                     data.push(v);
                 }
             }
             if data.len() > 0 {
                 // Guess size
                 let c = data.len();
                 // size^3 * 3 = c => size = cbrt(c/3)
                 let size = ((c as f32 / 3.0).powf(1.0/3.0).round()) as u32;
                 if (size * size * size * 3) as usize == c {
                     return validate_data_len(&data, size);
                 }
             }
        }
        
        // If no spaces, maybe hex blob.
        // Try decoding hex.
        if text.len() > 100 && text.chars().all(|c| c.is_digit(16)) {
             // Hex string.
             // Assume packed binary f32? or u8?
             // Usually Adobe uses 32-bit floats.
             // 8 chars per float?
             
             let mut data = Vec::new();
             // Chunks of 8 chars?
             // Or maybe it is base64? (A-Za-z0-9+/=)
             // Hex is just 0-9A-F. 
             // If all hex, likely hex.
             
             // Implementation: Parse chunks of 2 chars as byte? Then assemble floats?
             // Or chunks of 8 chars as f32 hex representation? (unlikely standard).
             // Most likely it is a byte array represented as hex. 
             // And the bytes form f32s (Little Endian?).
             
             // Let's try: parse to bytes, then cast to f32 slice.
             // Need even length.
             if text.len() % 2 != 0 {
                 return Err("Hex string has odd length".to_string());
             }
             
             let bytes: Vec<u8> = (0..text.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap_or(0))
                .collect();
                
             // Now view as f32s. 
             // We need exactly bytes.len() / 4 floats.
             if bytes.len() % 4 == 0 {
                 data = bytes.chunks(4).map(|chunk| {
                     f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                 }).collect();
                 
                  // Guess size
                 let c = data.len();
                 let size = ((c as f32 / 3.0).powf(1.0/3.0).round()) as u32;
                 if size > 0 && (size * size * size * 3) as usize == c {
                     return validate_data_len(&data, size);
                 }
             }
        }
    }
    
    Err("Could not find valid LUT data in XMP".to_string())
}

fn validate_data_len(data: &Vec<f32>, size: u32) -> Result<(Vec<f32>, u32), String> {
    let expected = (size * size * size * 3) as usize;
    if data.len() == expected {
        Ok((data.clone(), size)) // Clone needed to return ownership
    } else {
        Err(format!("Data length mismatch. Expected {}, got {}", expected, data.len()))
    }
}

fn is_numeric_line(line: &str) -> bool {
    line.chars().next().map_or(false, |c| c.is_digit(10) || c == '-')
}

fn parse_data_line(line: &str, data: &mut Vec<f32>) -> Result<(), String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(format!("Invalid data line: '{}'", line));
    }
    for part in parts {
        match f32::from_str(part) {
            Ok(v) => data.push(v),
            Err(_) => return Err(format!("Invalid number: '{}'", part)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_cube() {
        let content = "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1";
        let (data, size) = parse_lut(content, "cube").unwrap();
        assert_eq!(size, 2);
        assert_eq!(data.len(), 24);
    }
    
    #[test]
    fn test_parse_3dl_implicit() {
        // Simplified 3DL test content
        let content = "0 1023\n0 0 0\n1023 0 0\n0 1023 0\n1023 1023 0\n0 0 1023\n1023 0 1023\n0 1023 1023\n1023 1023 1023";
        let (data, size) = parse_lut(content, "3dl").unwrap();
        assert_eq!(size, 2);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[3], 1.0); // 1023 normalized to 1.0 if parser uses 1023 scale
    }
}
