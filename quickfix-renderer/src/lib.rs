use crate::renderer::{Renderer, RendererError};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

pub mod api_types;
pub mod geometry;
pub mod lut_parser;
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
#[serde(rename_all = "camelCase")]
pub struct CropRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[wasm_bindgen]
pub fn get_opaque_crop(rotation_deg: f32, width: f32, height: f32) -> JsValue {
    let rect = calculate_opaque_crop_rect(rotation_deg, width, height);
    serde_wasm_bindgen::to_value(&rect).unwrap()
}

use crate::api_types::Lut3DSettings;

#[derive(Serialize, Deserialize)]
struct LutResult {
    data: Vec<f32>,
    size: u32,
}

#[wasm_bindgen]
pub fn parse_cube_lut(content: &str) -> Result<JsValue, JsValue> {
    // Legacy support or default to .cube
    parse_lut(content, "cube")
}

#[wasm_bindgen]
pub fn parse_lut(content: &str, ext: &str) -> Result<JsValue, JsValue> {
    let (data, size) = lut_parser::parse_lut(content, ext).map_err(|e| JsValue::from_str(&e))?;
    let res = LutResult { data, size };
    Ok(serde_wasm_bindgen::to_value(&res)?)
}

pub fn calculate_opaque_crop_rect(rotation_deg: f32, width: f32, height: f32) -> CropRect {
    let rotation_rad = rotation_deg.to_radians();
    let (new_w, new_h) = geometry::calculate_largest_interior_rect(width, height, rotation_rad);

    let nw = new_w / width;
    let nh = new_h / height;

    CropRect {
        x: (1.0 - nw) / 2.0,
        y: (1.0 - nh) / 2.0,
        width: nw,
        height: nh,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CropSettings {
    pub rotation: Option<f32>,
    pub aspect_ratio: Option<f32>,
    pub rect: Option<CropRect>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ExposureSettings {
    pub exposure: Option<f32>,
    pub contrast: Option<f32>,
    pub highlights: Option<f32>,
    pub shadows: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ColorSettings {
    pub temperature: Option<f32>,
    pub tint: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GrainSettings {
    pub amount: f32,
    pub size: String, // "fine", "medium", "coarse"
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GeometrySettings {
    pub vertical: Option<f32>,
    pub horizontal: Option<f32>,
    pub flip_vertical: Option<bool>,
    pub flip_horizontal: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DenoiseSettings {
    pub luminance: f32,
    pub color: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CurvePoint {
    pub x: f32, // 0.0 to 1.0
    pub y: f32, // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ChannelCurve {
    pub points: Vec<CurvePoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CurvesSettings {
    pub intensity: f32,
    pub master: Option<ChannelCurve>,
    pub red: Option<ChannelCurve>,
    pub green: Option<ChannelCurve>,
    pub blue: Option<ChannelCurve>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct HslRangeSettings {
    pub hue: f32, // Offset degrees or normalized? Let's use -1.0 to 1.0 (mapped to e.g. -30 to 30 deg)
    pub saturation: f32, // -1.0 to 1.0 (offset)
    pub luminance: f32, // -1.0 to 1.0 (offset)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SplitToningSettings {
    pub shadow_hue: f32,    // 0..360
    pub shadow_sat: f32,    // 0..1
    pub highlight_hue: f32, // 0..360
    pub highlight_sat: f32, // 0..1
    pub balance: f32,       // -1..1
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct HslSettings {
    pub red: HslRangeSettings,
    pub orange: HslRangeSettings,
    pub yellow: HslRangeSettings,
    pub green: HslRangeSettings,
    pub aqua: HslRangeSettings,
    pub blue: HslRangeSettings,
    pub purple: HslRangeSettings,
    pub magenta: HslRangeSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct QuickFixAdjustments {
    pub crop: Option<CropSettings>,
    pub exposure: Option<ExposureSettings>,
    pub color: Option<ColorSettings>,
    pub curves: Option<CurvesSettings>,
    pub hsl: Option<HslSettings>,
    pub split_toning: Option<SplitToningSettings>,
    pub grain: Option<GrainSettings>,
    pub geometry: Option<GeometrySettings>,
    pub denoise: Option<DenoiseSettings>,

    pub lut: Option<Lut3DSettings>,
    pub vignette: Option<VignetteSettings>,
    pub sharpen: Option<SharpenSettings>,
    pub clarity: Option<ClaritySettings>,
    pub dehaze: Option<DehazeSettings>,
    pub distortion: Option<LensDistortionSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct VignetteSettings {
    pub amount: f32, // 0.0 to 1.0 (or -1.0 to 1.0 if supporting white vignette?) Users usually want dark. Let's assume -1.0 to 1.0 where < 0 is dark.
    pub midpoint: f32, // 0.0 to 1.0 (size of clear area)
    pub roundness: f32, // -1.0 to 1.0 (shape, square vs circle)
    pub feather: f32, // 0.0 to 1.0 (smoothness)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SharpenSettings {
    pub amount: f32,    // 0.0 to 1.0 (or higher)
    pub radius: f32,    // 0.0 to ... px
    pub threshold: f32, // 0.0 to 255.0
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ClaritySettings {
    pub amount: f32, // -1.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DehazeSettings {
    pub amount: f32, // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LensDistortionSettings {
    pub k1: f32, // Main radial distortion
    pub k2: f32, // Secondary radial distortion
                 // We could add centers or scale, but let's stick to k1/k2 for now
}

#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = r#"
export interface CropRect {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface LutResult {
    data: Float32Array;
    size: number;
}

export function parse_cube_lut(content: string): LutResult;

export class QuickFixRenderer {
  free(): void;
  static init(options?: RendererOptions): Promise<QuickFixRenderer>;
  backend: string;
  process_frame(data: Uint8Array, width: number, height: number, adjustments: QuickFixAdjustments, options?: ProcessOptions): Promise<FrameResult>;
  render_to_canvas(data: Uint8Array, width: number, height: number, adjustments: QuickFixAdjustments, canvas: HTMLCanvasElement): Promise<void>;
  final_render(data: Uint8Array, width: number, height: number, adjustments: QuickFixAdjustments): Promise<FrameResult>;
  set_lut(lut: LutResult, id?: string): Promise<void>;
}

export function get_opaque_crop(rotation_deg: number, width: number, height: number): CropRect;

export interface CropSettings {
    rotation?: number;
    aspect_ratio?: number;
    rect?: CropRect;
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
    flipVertical?: boolean;
    flipHorizontal?: boolean;
}

export interface DenoiseSettings {
    luminance: number;
    color: number;
}

export interface CurvesSettings {
    intensity: number;
    master?: ChannelCurve;
    red?: ChannelCurve;
    green?: ChannelCurve;
    blue?: ChannelCurve;
}

export interface HslRangeSettings {
    hue: number;
    saturation: number;
    luminance: number;
}

export interface HslSettings {
    red: HslRangeSettings;
    orange: HslRangeSettings;
    yellow: HslRangeSettings;
    green: HslRangeSettings;
    aqua: HslRangeSettings;
    blue: HslRangeSettings;
    purple: HslRangeSettings;
    magenta: HslRangeSettings;
}

export interface SplitToningSettings {
    shadowHue: number;
    shadowSat: number;
    highlightHue: number;
    highlightSat: number;
    balance: number;
}

export interface VignetteSettings {
    amount: number;
    midpoint: number;
    roundness: number;
    feather: number;
}

export interface SharpenSettings {
    amount: number;
    radius: number;
    threshold: number;
}

export interface ClaritySettings {
    amount: number;
}

export interface DehazeSettings {
    amount: number;
}

export interface LensDistortionSettings {
    k1: number;
    k2: number;
}

export interface ChannelCurve {
    points: CurvePoint[];
}

export interface CurvePoint {
    x: number;
    y: number;
}

export interface QuickFixAdjustments {
    crop?: CropSettings;
    exposure?: ExposureSettings;
    color?: ColorSettings;
    curves?: CurvesSettings;
    hsl?: HslSettings;
    splitToning?: SplitToningSettings;
    grain?: GrainSettings;
    geometry?: GeometrySettings;
    denoise?: DenoiseSettings;
    lut?: Lut3DSettings; // includes externalId
    vignette?: VignetteSettings;
    sharpen?: SharpenSettings;
    clarity?: ClaritySettings;
    dehaze?: DehazeSettings;
    distortion?: LensDistortionSettings;
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
    histogram: Vec<u32>,
}

#[wasm_bindgen]
impl FrameResult {
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn histogram(&self) -> Vec<u32> {
        self.histogram.clone()
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
    // Sync version does not support LUT for now
    let (data, w, h, histogram) =
        operations::process_frame_internal(data, width, height, &adjustments, None)
            .map_err(JsValue::from)?;

    Ok(FrameResult {
        data,
        width: w,
        height: h,
        histogram,
    })
}

// CPU Renderer wrapper to fit Renderer trait
struct CpuRenderer {
    luts: std::collections::HashMap<String, (Vec<f32>, u32)>,
    active_lut: Option<String>,
}

#[async_trait::async_trait(?Send)]
impl Renderer for CpuRenderer {
    async fn init(&mut self) -> Result<(), RendererError> {
        Ok(())
    }

    async fn set_lut(&mut self, id: Option<&str>, data: &[f32], size: u32) -> Result<(), RendererError> {
        let id = id.map(|s| s.to_string()).unwrap_or_else(|| "default".to_string());
        self.luts.insert(id.clone(), (data.to_vec(), size));
        self.active_lut = Some(id);
        Ok(())
    }

    async fn render(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
        _source_id: Option<&str>,
    ) -> Result<(Vec<u8>, Vec<u32>), RendererError> {
        // We need to copy data because process_frame_internal takes &mut [u8]
        let mut data_vec = data.to_vec();

        // Pass LUT data if available
        // First check settings for ID
        let lut_id = settings.lut.as_ref()
            .and_then(|l| l.external_id.as_deref())
            .or(self.active_lut.as_deref());
            
        let lut_ref = lut_id.and_then(|id| self.luts.get(id)).map(|(d, s)| (d.as_slice(), *s));

        let (res, _, _, histogram) =
            operations::process_frame_internal(&mut data_vec, width, height, settings, lut_ref)
                .map_err(RendererError::RenderFailed)?;
        Ok((res, histogram))
    }
    async fn render_to_canvas(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
        canvas: &web_sys::HtmlCanvasElement,
        _source_id: Option<&str>,
    ) -> Result<(), RendererError> {
        let mut data_vec = data.to_vec();
        
         if let Some(lut_settings) = &settings.lut {
            if let Some(id) = &lut_settings.external_id {
                let lut_ref = self.luts.get(id).map(|(d, s)| (d.as_slice(), *s));
                
                let (res, w, h, _) =
                     operations::process_frame_internal(&mut data_vec, width, height, settings, lut_ref)
                        .map_err(RendererError::RenderFailed)?;
                 // Copy canvas stuff...
                 if canvas.width() != w || canvas.height() != h {
                    canvas.set_width(w);
                    canvas.set_height(h);
                }

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
                 return Ok(());
            }
        }
        
        // Fallback to active logic if no ID or other cases (legacy)
        let lut_ref = self.active_lut.as_ref().and_then(|id| self.luts.get(id)).map(|(d,s)| (d.as_slice(), *s));

        let (res, w, h, _) =
            operations::process_frame_internal(&mut data_vec, width, height, settings, lut_ref)
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
    lut_cache: Option<(Vec<f32>, u32)>,
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
                    lut_cache: None,
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
                    lut_cache: None,
                });
            } else if force == Some("webgl2") {
                return Err(JsValue::from_str("WebGL2 forced but failed to initialize"));
            }
        }

        // 3. Fallback to CPU
        // 3. Fallback to CPU
        if force.is_none() || force == Some("cpu") {
            return Ok(QuickFixRenderer {
                renderer: Box::new(CpuRenderer { 
                    luts: std::collections::HashMap::new(),
                    active_lut: None,
                }),
                backend_name: "cpu".to_string(),
                lut_cache: None,
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
        options: Option<ProcessOptions>,
    ) -> Result<FrameResult, JsValue> {
        let adjustments: QuickFixAdjustments = serde_wasm_bindgen::from_value(adjustments)?;

        // Debug Log
        use web_sys::console;
        console::log_1(
            &format!(
                "WASM: Processing frame. Crop rect: {:?}",
                adjustments.crop.as_ref().and_then(|c| c.rect.as_ref())
            )
            .into(),
        );

        let source_id = options.as_ref().and_then(|o| o.source_id.as_deref());

        let (result_data, histogram) = self
            .renderer
            .render(data, width, height, &adjustments, source_id)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(FrameResult {
            data: result_data,
            width,
            height,
            histogram,
        })
    }

    pub async fn set_lut(&mut self, lut: JsValue, id: Option<String>) -> Result<(), JsValue> {
        // lut is expected to be { data: Float32Array, size: number }
        // We can deserialize it into LutResult struct
        let lut: LutResult = serde_wasm_bindgen::from_value(lut)?;
        self.renderer
            .set_lut(id.as_deref(), &lut.data, lut.size)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            
        // Cache for CPU final export
        self.lut_cache = Some((lut.data, lut.size));
        Ok(())
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
        // source_id not supported in simple render_to_canvas yet or need options? 
        // For now pass None.
        self.renderer
            .render_to_canvas(data, width, height, &adjustments, &canvas, None)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }

    pub async fn final_render(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        adjustments: JsValue,
    ) -> Result<FrameResult, JsValue> {
        let adjustments: QuickFixAdjustments = serde_wasm_bindgen::from_value(adjustments)?;
        
         // Debug Log
        use web_sys::console;
        console::log_1(&"WASM: Starting final_render (tiled)".into());

        // Extract LUT if available in CPU renderer or passed separately? 
        // Currently final_render doesn't support WebGPU/WebGL backends easily because they are coupled to context.
        // It makes sense to run final_render on CPU to avoid GPU limits for huge exports, or implementing tiled GPU usage.
        // Given the requirement "process full-resolution... without crashing", CPU tiling is safest.
        
        // We'll borrow LUT from the renderer if it's a CpuRenderer, or we might need to properly pass it.
        // But `self.renderer` is a trait object.
        // Actually, we can just strictly use the CPU implementation for final export for now.
        // Or we should update the Renderer trait to support `final_render`.
        // Simplest: Call operations::final_render_tiled directly here.
        // But what about LUT?
        // We can expose LUT getter in Renderer trait or just ignore LUT for now if not critical, OR better:
        // Assume user passed LUT in `adjustments.lut` but the DATA is separate.
        // The `QuickFixRenderer` has a `set_lut` method.
        // But `operations::final_render_tiled` takes `lut_buffer`.
        // We don't have easy access to the LUT stored inside `Box<dyn Renderer>`.
        // But wait, `CpuRenderer` has `lut` field.
        // If we cast `self.renderer` to `CpuRenderer`, we can get it.
        // But if `self.renderer` is `WebGpuRenderer`, the LUT is in a texture.
        // So for `final_render`, we might need the user to re-supply the LUT or cache it in `QuickFixRenderer`.
        
        // Strategy: Add `lut_cache` to `QuickFixRenderer` struct in `lib.rs`!
        // When `set_lut` is called, store it there too.
        
        // For now, let's implement the method and use a workaround or fix the struct.
        // I will just modify the struct fields in a moment.
        
        // Use cached LUT
        let lut_ref = self.lut_cache.as_ref().map(|(d, s)| (d.as_slice(), *s));
        
        let (res, w, h) = operations::final_render_tiled(data, width, height, &adjustments, lut_ref)
             .map_err(|e| JsValue::from_str(&e))?;
             
        let histogram = operations::compute_histogram(&res);

        Ok(FrameResult {
            data: res,
            width: w,
            height: h,
            histogram,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_types::Lut3DSettings; // Import specific type
    use crate::operations::apply_lut_in_place; // Import helper
    use image::{Rgba, RgbaImage};

    fn create_test_image(width: u32, height: u32, color: [u8; 4]) -> RgbaImage {
        let mut img = RgbaImage::new(width, height);
        for p in img.pixels_mut() {
            *p = Rgba(color);
        }
        img
    }

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
    fn test_denoise_deserialization() {
        let json = r#"{
            "denoise": {"luminance": 0.5, "color": 0.7}
        }"#;
        let adj: QuickFixAdjustments = serde_json::from_str(json).unwrap();
        assert_eq!(adj.denoise.as_ref().unwrap().luminance, 0.5);
        assert_eq!(adj.denoise.as_ref().unwrap().color, 0.7);
    }

    #[test]
    fn test_deterministic_grain() {
        let width = 100;
        let height = 100;
        let mut data1 = vec![128u8; (width * height * 4) as usize];
        let mut data2 = vec![128u8; (width * height * 4) as usize];

        let adj = QuickFixAdjustments {
            grain: Some(GrainSettings {
                amount: 0.5,
                size: "medium".to_string(),
                seed: Some(555),
            }),
            ..Default::default()
        };

        let res1 =
            operations::process_frame_internal(&mut data1, width, height, &adj, None).unwrap();
        let res2 =
            operations::process_frame_internal(&mut data2, width, height, &adj, None).unwrap();

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
        let res = operations::process_frame_internal(&mut data, width, height, &adj, None);
        assert!(res.is_ok());
    }

    #[test]
    fn test_apply_lut_identity() {
        let width = 2;
        let height = 2;
        let mut img = create_test_image(width, height, [100, 100, 100, 255]);

        // Identity LUT 2x2x2
        // Order: R(0.0), R(1.0). For each R, G(0), G(1). For each G, B(0), B(1).
        // size = 2.
        // B moves fastest? Code: idx = ((b * size + g) * size + r).
        // Wait, loop order in code:
        // for r.. for g.. for b.. -> sample(root..).
        // sample function: idx = ((b * size + g) * size + r)
        // This implies r is safest (inner-most)? No.
        // ((b * size + g) * size + r) implies:
        // r changes 0..size.
        // g changes 0..size.
        // b changes 0..size.
        // So memory layout: r changes fastest (contiguous).
        // 0,0,0 -> 1,0,0 -> 0,1,0 -> 1,1,0 ...

        // Identity LUT: value at (r,g,b) is (r,g,b).
        let mut lut_data = Vec::new();
        for b in 0..2 {
            for g in 0..2 {
                for r in 0..2 {
                    lut_data.push(r as f32);
                    lut_data.push(g as f32);
                    lut_data.push(b as f32);
                }
            }
        }

        let settings = Lut3DSettings {
            intensity: 1.0,
            ..Default::default()
        };

        apply_lut_in_place(&mut img, &lut_data, 2, &settings);

        let px = img.get_pixel(0, 0);
        // Input 100/255 ~= 0.392.
        // LUT should map 0.392 -> 0.392.
        // Allow simplified rounding error
        assert!((px[0] as i32 - 100).abs() <= 1);
        assert!((px[1] as i32 - 100).abs() <= 1);
        assert!((px[2] as i32 - 100).abs() <= 1);
    }

    #[test]
    fn test_apply_lut_red() {
        let width = 2;
        let height = 2;
        // Input gray
        let mut img = create_test_image(width, height, [100, 100, 100, 255]);

        // Red LUT: Always returns (1.0, 0.0, 0.0)
        let mut lut_data = Vec::new(); // 2x2x2
        for _ in 0..8 {
            lut_data.push(1.0);
            lut_data.push(0.0);
            lut_data.push(0.0);
        }

        let settings = Lut3DSettings {
            intensity: 0.5, // 50% blend
            ..Default::default()
        };

        apply_lut_in_place(&mut img, &lut_data, 2, &settings);

        let px = img.get_pixel(0, 0);
        // Original: 100. Target: 255, 0, 0.
        // Result: 100 * 0.5 + 255 * 0.5 = 50 + 127.5 = 177.5
        // G/B: 100 * 0.5 + 0 = 50.

        assert!(px[0] >= 177 && px[0] <= 178);
        assert!(px[1] >= 49 && px[1] <= 51);
        assert!(px[2] >= 49 && px[2] <= 51);
    }

    #[test]
    fn test_get_opaque_crop_api() {
        let crop = calculate_opaque_crop_rect(90.0, 100.0, 50.0);

        // 90 deg rotation of 100x50 fits into 50x100.
        // Largest interior with ratio 2:1 inside 50x100.
        // w = 50. h = 25.
        // norm_w = 50/100 = 0.5.
        // norm_h = 25/50 = 0.5.
        // x = (1 - 0.5)/2 = 0.25.
        // y = (1 - 0.5)/2 = 0.25.

        assert!((crop.width - 0.5).abs() < 1e-4);
        assert!((crop.height - 0.5).abs() < 1e-4);
        assert!((crop.x - 0.25).abs() < 1e-4);
        assert!((crop.y - 0.25).abs() < 1e-4);
    }

    #[test]
    fn test_cpu_renderer_lut_caching() {
        // Instantiate CpuRenderer directly
        let mut renderer = CpuRenderer {
            luts: std::collections::HashMap::new(),
            active_lut: None,
        };

        // LUT A: Red tint (R=1.0, G=0.0, B=0.0)
        let mut lut_a = Vec::with_capacity(8 * 3);
        for _ in 0..8 { lut_a.extend_from_slice(&[1.0, 0.0, 0.0]); }
        
        // LUT B: Blue tint (R=0.0, G=0.0, B=1.0)
        let mut lut_b = Vec::with_capacity(8 * 3);
        for _ in 0..8 { lut_b.extend_from_slice(&[0.0, 0.0, 1.0]); }

        // 1. Upload LUTs using block_on since we don't have tokio runtime
        futures::executor::block_on(async {
            renderer.set_lut(Some("lut_a"), &lut_a, 2).await.unwrap();
            renderer.set_lut(Some("lut_b"), &lut_b, 2).await.unwrap();
        });

        assert_eq!(renderer.luts.len(), 2);
        assert_eq!(renderer.active_lut, Some("lut_b".to_string()));

        // 2. Prepare test image (Medium Gray)
        let input_data = vec![128, 128, 128, 255]; 
        let width = 1;

        // 3. Render with LUT A (Red) using id
        let settings_a = QuickFixAdjustments {
            lut: Some(Lut3DSettings {
                intensity: 1.0,
                tint: None,
                external_id: Some("lut_a".to_string()),
            }),
            ..Default::default()
        };

        let (res_a, _) = futures::executor::block_on(renderer.render(&input_data, width, 1, &settings_a, None)).unwrap();
        
        assert!(res_a[0] > 200); // Red
        assert!(res_a[1] < 50);  // Green
        assert!(res_a[2] < 50);  // Blue

        // 4. Render with LUT B (Blue) using id
        let settings_b = QuickFixAdjustments {
            lut: Some(Lut3DSettings {
                intensity: 1.0,
                tint: None,
                external_id: Some("lut_b".to_string()),
            }),
            ..Default::default()
        };

        let (res_b, _) = futures::executor::block_on(renderer.render(&input_data, width, 1, &settings_b, None)).unwrap();
        
        assert!(res_b[0] < 50);  // Red
        assert!(res_b[2] > 200); // Blue
        
        // 5. Render with Missing ID
        let settings_c = QuickFixAdjustments {
            lut: Some(Lut3DSettings {
                intensity: 1.0,
                tint: None,
                external_id: Some("missing".to_string()),
            }),
            ..Default::default()
        };
        
        let (res_c, _) = futures::executor::block_on(renderer.render(&input_data, width, 1, &settings_c, None)).unwrap();
        // Should be original gray
        assert!((res_c[0] as i32 - 128).abs() < 5);
        assert!((res_c[1] as i32 - 128).abs() < 5);
        assert!((res_c[2] as i32 - 128).abs() < 5);
    }
}
