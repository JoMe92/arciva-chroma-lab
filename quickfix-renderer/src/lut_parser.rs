use std::str::FromStr;

/// Parses a standard Adobe .cube file content.
/// Returns a tuple of (LUT data as flat Vec<f32>, LUT size).
/// The data is ordered by R, then G, then B (standard .cube order).
/// 
/// Supports:
/// - Comments starting with #
/// - TITLE (ignored)
/// - DOMAIN_MIN / DOMAIN_MAX (ignored, assumed 0.0-1.0)
/// - LUT_3D_SIZE
/// 
/// Does NOT support:
/// - LUT_1D_SIZE (will error if found)
/// - Non-standard formats
pub fn parse_cube_file(content: &str) -> Result<(Vec<f32>, u32), String> {
    let mut size = 0u32;
    let mut data = Vec::new();
    let mut data_started = false;

    // Use lines() to iterate.
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
                // Ignore metadata
                continue;
            } else {
                // Assume data starts here if we have a size, OR checking for first number
                // Standard says keywords are uppercase. Use simple heuristic: 
                // if it parses as 3 floats, it's data.
                if size > 0 {
                    // Start parsing data
                    data_started = true;
                    parse_data_line(line, &mut data)?;
                } else {
                    // Maybe unexpected keyword or data before size
                    // If line looks like numbers, error because size missing
                    if is_numeric_line(line) {
                         return Err("Found data before LUT_3D_SIZE".to_string());
                    }
                }
            }
        } else {
            // Data mode
            parse_data_line(line, &mut data)?;
        }
    }

    if size == 0 {
        return Err("LUT_3D_SIZE not found".to_string());
    }

    let expected_count = (size * size * size * 3) as usize;
    if data.len() != expected_count {
        return Err(format!("Expected {} values, found {}", expected_count, data.len()));
    }

    Ok((data, size))
}

fn is_numeric_line(line: &str) -> bool {
    // Quick check if line starts with a digit or minus
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
        let content = r#"
        # A simple test LUT
        TITLE Test
        LUT_3D_SIZE 2
        0.0 0.0 0.0
        1.0 0.0 0.0
        0.0 1.0 0.0
        1.0 1.0 0.0
        0.0 0.0 1.0
        1.0 0.0 1.0
        0.0 1.0 1.0
        1.0 1.0 1.0
        "#;

        let (data, size) = parse_cube_file(content).unwrap();
        assert_eq!(size, 2);
        assert_eq!(data.len(), 2 * 2 * 2 * 3);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[3], 1.0); // Next R
        assert_eq!(data.last(), Some(&1.0));
    }

    #[test]
    fn test_parse_invalid_size() {
        let content = "LUT_3D_SIZE invalid";
        assert!(parse_cube_file(content).is_err());
    }

    #[test]
    fn test_parse_missing_size() {
        let content = "0.0 0.0 0.0";
        assert!(parse_cube_file(content).is_err());
    }
    
    #[test]
    fn test_parse_1d_error() {
        let content = "LUT_1D_SIZE 10";
        assert!(parse_cube_file(content).is_err());
    }
}
