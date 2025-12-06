use crate::renderer::{Renderer, RendererError};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

pub mod api_types;
pub mod operations;
pub mod renderer;
pub mod shaders;
#[cfg(target_arch = "wasm32")]
pub mod webgl;
#[cfg(target_arch = "wasm32")]
pub mod webgpu;

use api_types::{ProcessOptions, RendererOptions};

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CropSettings {
    pub rotation: Option<f32>,
    pub aspect_ratio: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExposureSettings {
    pub exposure: Option<f32>,
    pub contrast: Option<f32>,
    pub highlights: Option<f32>,
    pub shadows: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ColorSettings {
    pub temperature: Option<f32>,
    pub tint: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GrainSettings {
    pub amount: f32,
    pub size: String, // "fine", "medium", "coarse"
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeometrySettings {
    pub vertical: Option<f32>,
    pub horizontal: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuickFixAdjustments {
    pub crop: Option<CropSettings>,
    pub exposure: Option<ExposureSettings>,
    pub color: Option<ColorSettings>,
    pub grain: Option<GrainSettings>,
    pub geometry: Option<GeometrySettings>,
}

#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = r#"
export interface CropSettings {
    rotation?: number;
    aspect_ratio?: number;
}

export interface ExposureSettings {
    exposure?: number;
    contrast?: number;
    highlights?: number;
    highlight_saturation?: number;
    shadows?: number;
    shadow_saturation?: number;
}

export interface ColorSettings {
    temperature?: number;
    tint?: number;
}

export interface GrainSettings {
    amount: number;
    size: "fine" | "medium" | "coarse";
    seed?: number;
}

export interface GeometrySettings {
    vertical?: number;
    horizontal?: number;
}

export interface QuickFixAdjustments {
    crop?: CropSettings;
    exposure?: ExposureSettings;
    color?: ColorSettings;
    grain?: GrainSettings;
    geometry?: GeometrySettings;
}
"#;

impl QuickFixAdjustments {
    pub fn new() -> QuickFixAdjustments {
        QuickFixAdjustments::default()
    }
}

#[wasm_bindgen]
pub struct FrameResult {
    data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[wasm_bindgen]
impl FrameResult {
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }
}

// Legacy CPU synchronous entry point - kept for backward compat or direct usage
#[wasm_bindgen]
pub fn process_frame_sync(
    data: &mut [u8],
    width: u32,
    height: u32,
    adjustments: JsValue,
) -> Result<FrameResult, JsValue> {
    let adjustments: QuickFixAdjustments = serde_wasm_bindgen::from_value(adjustments)?;
    let (data, w, h) = operations::process_frame_internal(data, width, height, &adjustments)
        .map_err(|e| JsValue::from_str(&e))?;

    Ok(FrameResult {
        data,
        width: w,
        height: h,
    })
}

// CPU Renderer wrapper to fit Renderer trait
struct CpuRenderer;
#[async_trait::async_trait(?Send)]
impl Renderer for CpuRenderer {
    async fn init(&mut self) -> Result<(), RendererError> {
        Ok(())
    }
    async fn render(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
    ) -> Result<Vec<u8>, RendererError> {
        // We need to copy data because process_frame_internal takes &mut [u8]
        let mut data_vec = data.to_vec();
        let (res, _, _) =
            operations::process_frame_internal(&mut data_vec, width, height, settings)
                .map_err(RendererError::RenderFailed)?;
        Ok(res)
    }
    async fn render_to_canvas(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
        canvas: &web_sys::HtmlCanvasElement,
    ) -> Result<(), RendererError> {
        let mut data_vec = data.to_vec();
        let (res, w, h) =
            operations::process_frame_internal(&mut data_vec, width, height, settings)
                .map_err(RendererError::RenderFailed)?;

        if canvas.width() != w || canvas.height() != h {
            canvas.set_width(w);
            canvas.set_height(h);
        }

        // Draw to canvas using 2D context
        let ctx = canvas
            .get_context("2d")
            .map_err(|_| RendererError::RenderFailed("Failed to get 2d context".into()))?
            .ok_or(RendererError::RenderFailed(
                "Failed to get 2d context".into(),
            ))?
            .dyn_into::<web_sys::CanvasRenderingContext2d>()
            .map_err(|_| RendererError::RenderFailed("Failed to cast to 2d context".into()))?;

        let clamped = wasm_bindgen::Clamped(&res[..]);
        let image_data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(clamped, w, h)
            .map_err(|e| RendererError::RenderFailed(format!("{:?}", e)))?;

        ctx.put_image_data(&image_data, 0.0, 0.0)
            .map_err(|e| RendererError::RenderFailed(format!("{:?}", e)))?;

        Ok(())
    }
}

#[wasm_bindgen]
pub struct QuickFixRenderer {
    renderer: Box<dyn Renderer>,
    backend_name: String,
}

#[wasm_bindgen]
impl QuickFixRenderer {
    // Replaces the old create method with init, matching the requirement
    pub async fn init(options: Option<RendererOptions>) -> Result<QuickFixRenderer, JsValue> {
        let options = options.unwrap_or(RendererOptions {
            preferred_backend: None,
            max_preview_size: None,
        });

        let force = options.preferred_backend.as_deref();

        // 1. Try WebGPU (WASM only)
        #[cfg(target_arch = "wasm32")]
        if force.is_none() || force == Some("webgpu") {
            let mut renderer = webgpu::WebGpuRenderer::new();
            if renderer.init().await.is_ok() {
                return Ok(QuickFixRenderer {
                    renderer: Box::new(renderer),
                    backend_name: "webgpu".to_string(),
                });
            } else if force == Some("webgpu") {
                return Err(JsValue::from_str("WebGPU forced but failed to initialize"));
            }
        }

        // 2. Try WebGL2 (WASM only)
        #[cfg(target_arch = "wasm32")]
        if force.is_none() || force == Some("webgl2") {
            let mut renderer = webgl::WebGlRenderer::new();
            if renderer.init().await.is_ok() {
                return Ok(QuickFixRenderer {
                    renderer: Box::new(renderer),
                    backend_name: "webgl2".to_string(),
                });
            } else if force == Some("webgl2") {
                return Err(JsValue::from_str("WebGL2 forced but failed to initialize"));
            }
        }

        // 3. Fallback to CPU
        if force.is_none() || force == Some("cpu") {
            return Ok(QuickFixRenderer {
                renderer: Box::new(CpuRenderer),
                backend_name: "cpu".to_string(),
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        if force == Some("webgpu") || force == Some("webgl2") {
            return Err(JsValue::from_str(
                "WebGPU/WebGL2 not supported on this platform",
            ));
        }

        Err(JsValue::from_str("No suitable backend found"))
    }

    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        self.backend_name.clone()
    }

    pub async fn process_frame(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        adjustments: JsValue,
        _options: Option<ProcessOptions>,
    ) -> Result<FrameResult, JsValue> {
        let adjustments: QuickFixAdjustments = serde_wasm_bindgen::from_value(adjustments)?;
        let result = self
            .renderer
            .render(data, width, height, &adjustments)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(FrameResult {
            data: result,
            width,
            height,
        })
    }

    // Keep render_to_canvas for convenience if passing a canvas directly
    pub async fn render_to_canvas(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        adjustments: JsValue,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<(), JsValue> {
        let adjustments: QuickFixAdjustments = serde_wasm_bindgen::from_value(adjustments)?;
        self.renderer
            .render_to_canvas(data, width, height, &adjustments, &canvas)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let json = r#"{
            "crop": {"rotation": 90.0},
            "exposure": {"exposure": 1.0},
            "grain": {"amount": 0.5, "size": "fine", "seed": 123}
        }"#;
        let adj: QuickFixAdjustments = serde_json::from_str(json).unwrap();
        assert_eq!(adj.crop.unwrap().rotation, Some(90.0));
        assert_eq!(adj.exposure.unwrap().exposure, Some(1.0));
        assert_eq!(adj.grain.as_ref().unwrap().amount, 0.5);
        assert_eq!(adj.grain.as_ref().unwrap().seed, Some(123));
    }

    #[test]
    fn test_deterministic_grain() {
        let width = 100;
        let height = 100;
        let mut data1 = vec![128u8; (width * height * 4) as usize];
        let mut data2 = vec![128u8; (width * height * 4) as usize];

        let mut adj = QuickFixAdjustments::default();
        adj.grain = Some(GrainSettings {
            amount: 0.5,
            size: "medium".to_string(),
            seed: Some(555),
        });

        let res1 = operations::process_frame_internal(&mut data1, width, height, &adj).unwrap();
        let res2 = operations::process_frame_internal(&mut data2, width, height, &adj).unwrap();

        assert_eq!(
            res1.0, res2.0,
            "Grain output should be identical for same seed"
        );
    }

    #[test]
    fn test_pipeline_no_panic() {
        let width = 10;
        let height = 10;
        let mut data = vec![0u8; (width * height * 4) as usize];
        let adj = QuickFixAdjustments::default();
        let res = operations::process_frame_internal(&mut data, width, height, &adj);
        assert!(res.is_ok());
    }
}
 
