use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct RendererOptions {
    #[wasm_bindgen(skip)]
    pub preferred_backend: Option<String>,

    #[wasm_bindgen(skip)]
    pub max_preview_size: Option<u32>,
}

#[wasm_bindgen]
impl RendererOptions {
    #[wasm_bindgen(constructor)]
    pub fn new(
        preferred_backend: Option<String>,
        max_preview_size: Option<u32>,
    ) -> RendererOptions {
        RendererOptions {
            preferred_backend,
            max_preview_size,
        }
    }

    #[wasm_bindgen(getter = preferredBackend)]
    pub fn preferred_backend(&self) -> Option<String> {
        self.preferred_backend.clone()
    }

    #[wasm_bindgen(setter = preferredBackend)]
    pub fn set_preferred_backend(&mut self, backend: Option<String>) {
        self.preferred_backend = backend;
    }

    #[wasm_bindgen(getter = maxPreviewSize)]
    pub fn max_preview_size(&self) -> Option<u32> {
        self.max_preview_size
    }

    #[wasm_bindgen(setter = maxPreviewSize)]
    pub fn set_max_preview_size(&mut self, size: Option<u32>) {
        self.max_preview_size = size;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct ProcessOptions {
    #[wasm_bindgen(skip)]
    pub return_image_bitmap: Option<bool>,
}

#[wasm_bindgen]
impl ProcessOptions {
    #[wasm_bindgen(constructor)]
    pub fn new(return_image_bitmap: Option<bool>) -> ProcessOptions {
        ProcessOptions {
            return_image_bitmap,
        }
    }

    #[wasm_bindgen(getter = returnImageBitmap)]
    pub fn return_image_bitmap(&self) -> Option<bool> {
        self.return_image_bitmap
    }

    #[wasm_bindgen(setter = returnImageBitmap)]
    pub fn set_return_image_bitmap(&mut self, val: Option<bool>) {
        self.return_image_bitmap = val;
    }
}
