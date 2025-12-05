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
    ) -> Result<Vec<u8>, RendererError>;

    /// Render directly to a canvas (if applicable)
    /// This is useful for the preview path to avoid readback
    async fn render_to_canvas(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
        canvas: &web_sys::HtmlCanvasElement,
    ) -> Result<(), RendererError>;
}
