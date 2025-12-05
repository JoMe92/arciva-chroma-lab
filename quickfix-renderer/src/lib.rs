use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct QuickFixAdjustments {
    pub exposure: f32,
    pub contrast: f32,
    // Add other fields as needed
}

#[wasm_bindgen]
impl QuickFixAdjustments {
    #[wasm_bindgen(constructor)]
    pub fn new() -> QuickFixAdjustments {
        QuickFixAdjustments {
            exposure: 0.0,
            contrast: 0.0,
        }
    }
}

impl Default for QuickFixAdjustments {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
pub fn process_frame(
    data: &mut [u8],
    width: u32,
    height: u32,
    adjustments: &QuickFixAdjustments,
) -> Result<(), JsValue> {
    // Placeholder: No-op for now, just logging
    web_sys::console::log_1(
        &format!(
            "Processing frame {}x{} with adjustments: {:?}",
            width, height, adjustments
        )
        .into(),
    );

    // In the future, we will modify 'data' in place here based on features.
    let _ = data; // Suppress unused variable warning

    Ok(())
}
