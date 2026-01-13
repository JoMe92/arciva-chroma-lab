use crate::QuickFixAdjustments;
use async_trait::async_trait;

#[derive(Debug, thiserror::Error)]
pub enum RendererError {
    #[error("WebGPU not supported")]
    WebGpuNotSupported,
    #[error("WebGL2 not supported")]
    WebGl2NotSupported,
    #[error("Initialization failed: {0}")]
    InitFailed(String),
    #[error("Render failed: {0}")]
    RenderFailed(String),
}

#[async_trait(?Send)]
pub trait Renderer {
    /// Initialize the renderer with the given canvas (optional)
    async fn init(&mut self) -> Result<(), RendererError>;

    /// Render a frame with the given settings and return the result as a buffer
    async fn render(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
        source_id: Option<&str>,
    ) -> Result<(Vec<u8>, Vec<u32>), RendererError>;

    /// Render directly to a canvas (if applicable)
    /// This is useful for the preview path to avoid readback
    async fn render_to_canvas(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
        canvas: &web_sys::HtmlCanvasElement,
        source_id: Option<&str>,
    ) -> Result<(), RendererError>;

    /// Set a 3D LUT for color grading with an optional ID for caching.
    /// Data is flat RGB float array. Size is dimension size (e.g. 33 for 33x33x33).
    async fn set_lut(
        &mut self,
        _id: Option<&str>,
        _data: &[f32],
        _size: u32,
    ) -> Result<(), RendererError> {
        Ok(()) // Default implementation for backends that don't support it yet
    }
}
