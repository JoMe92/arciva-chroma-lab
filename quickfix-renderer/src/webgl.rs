use crate::{renderer::{Renderer, RendererError}, QuickFixAdjustments, shaders::{GLSL_VERTEX, GLSL_FRAGMENT}};
use async_trait::async_trait;
use glow::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

pub struct WebGlRenderer {
    context: Option<glow::Context>,
    program: Option<glow::Program>,
    vao: Option<glow::VertexArray>,
    texture: Option<glow::Texture>,
    grain_texture: Option<glow::Texture>,
    width: u32,
    height: u32,
    canvas: Option<HtmlCanvasElement>, // Keep reference if we created context from it
}

impl WebGlRenderer {
    pub fn new() -> Self {
        Self {
            context: None,
            program: None,
            vao: None,
            texture: None,
            grain_texture: None,
            width: 0,
            height: 0,
            canvas: None,
        }
    }

    fn ensure_initialized(&mut self, canvas: Option<&HtmlCanvasElement>) -> Result<(), RendererError> {
        if self.context.is_some() {
            return Ok(());
        }

        // If canvas is provided, use it. Otherwise create an offscreen canvas?
        // WebGL requires a canvas.
        // If we are just processing frames (readback), we need an offscreen canvas.
        let gl = if let Some(c) = canvas {
            let context = c.get_context("webgl2")
                .map_err(|_| RendererError::WebGl2NotSupported)?
                .ok_or(RendererError::WebGl2NotSupported)?
                .dyn_into::<web_sys::WebGl2RenderingContext>()
                .map_err(|_| RendererError::WebGl2NotSupported)?;
            glow::Context::from_webgl2_context(context)
        } else {
            // Create offscreen canvas
            let doc = web_sys::window().unwrap().document().unwrap();
            let c = doc.create_element("canvas").unwrap().dyn_into::<HtmlCanvasElement>().unwrap();
            // Set some size?
            c.set_width(1);
            c.set_height(1);
            
            let context = c.get_context("webgl2")
                .map_err(|_| RendererError::WebGl2NotSupported)?
                .ok_or(RendererError::WebGl2NotSupported)?
                .dyn_into::<web_sys::WebGl2RenderingContext>()
                .map_err(|_| RendererError::WebGl2NotSupported)?;
            
            self.canvas = Some(c);
            glow::Context::from_webgl2_context(context)
        };

        unsafe {
            let program = gl.create_program().map_err(RendererError::InitFailed)?;
            
            let vs = gl.create_shader(glow::VERTEX_SHADER).map_err(RendererError::InitFailed)?;
            gl.shader_source(vs, GLSL_VERTEX);
            gl.compile_shader(vs);
            if !gl.get_shader_compile_status(vs) {
                return Err(RendererError::InitFailed(gl.get_shader_info_log(vs)));
            }
            
            let fs = gl.create_shader(glow::FRAGMENT_SHADER).map_err(RendererError::InitFailed)?;
            gl.shader_source(fs, GLSL_FRAGMENT);
            gl.compile_shader(fs);
            if !gl.get_shader_compile_status(fs) {
                return Err(RendererError::InitFailed(gl.get_shader_info_log(fs)));
            }
            
            gl.attach_shader(program, vs);
            gl.attach_shader(program, fs);
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                return Err(RendererError::InitFailed(gl.get_program_info_log(program)));
            }
            
            // VAO for full screen quad
            let vao = gl.create_vertex_array().map_err(RendererError::InitFailed)?;
            gl.bind_vertex_array(Some(vao));
            
            let vbo = gl.create_buffer().map_err(RendererError::InitFailed)?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            // Quad: -1,-1 to 1,1
            let vertices: [f32; 12] = [
                -1.0, -1.0,
                 1.0, -1.0,
                -1.0,  1.0,
                -1.0,  1.0,
                 1.0, -1.0,
                 1.0,  1.0,
            ];
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&vertices), glow::STATIC_DRAW);
            
            let pos_loc = gl.get_attrib_location(program, "position").unwrap();
            gl.enable_vertex_attrib_array(pos_loc);
            gl.vertex_attrib_pointer_f32(pos_loc, 2, glow::FLOAT, false, 0, 0);
            
            // Textures
            let texture = gl.create_texture().map_err(RendererError::InitFailed)?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            
            let grain_texture = gl.create_texture().map_err(RendererError::InitFailed)?;
            gl.bind_texture(glow::TEXTURE_2D, Some(grain_texture));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            
            // Upload grain data
            let grain_size = 256;
            use rand_distr::{Distribution, Normal};
            let normal = Normal::new(0.5, 0.15).unwrap();
            let mut rng = rand::thread_rng();
            let mut grain_data = Vec::with_capacity(grain_size * grain_size * 4);
            for _ in 0..(grain_size * grain_size) {
                let v: f32 = normal.sample(&mut rng);
                let b = (v.clamp(0.0, 1.0) * 255.0) as u8;
                grain_data.extend_from_slice(&[b, b, b, 255]); // RGBA
            }
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                grain_size as i32,
                grain_size as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                Some(&grain_data),
            );
            
            self.context = Some(gl);
            self.program = Some(program);
            self.vao = Some(vao);
            self.texture = Some(texture);
            self.grain_texture = Some(grain_texture);
        }
        
        Ok(())
    }
    
    unsafe fn render_internal(&mut self, data: &[u8], width: u32, height: u32, settings: &QuickFixAdjustments) -> Result<(), RendererError> {
        let gl = self.context.as_ref().unwrap();
        let program = self.program.unwrap();
        
        gl.use_program(Some(program));
        gl.bind_vertex_array(self.vao);
        
        // Upload source texture
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, self.texture);
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA as i32,
            width as i32,
            height as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            Some(data),
        );
        
        // Bind grain texture
        gl.active_texture(glow::TEXTURE1);
        gl.bind_texture(glow::TEXTURE_2D, self.grain_texture);
        
        // Uniforms
        let loc = |name| gl.get_uniform_location(program, name);
        
        gl.uniform_1_i32(loc("u_texture").as_ref(), 0);
        gl.uniform_1_i32(loc("u_grain").as_ref(), 1);
        
        gl.uniform_1_f32(loc("u_geo_vertical").as_ref(), settings.geometry.as_ref().and_then(|g| g.vertical).unwrap_or(0.0));
        gl.uniform_1_f32(loc("u_geo_horizontal").as_ref(), settings.geometry.as_ref().and_then(|g| g.horizontal).unwrap_or(0.0));
        gl.uniform_1_f32(loc("u_crop_rotation").as_ref(), settings.crop.as_ref().and_then(|c| c.rotation).unwrap_or(0.0).to_radians());
        gl.uniform_1_f32(loc("u_crop_aspect").as_ref(), settings.crop.as_ref().and_then(|c| c.aspect_ratio).unwrap_or(0.0));
        gl.uniform_1_f32(loc("u_exposure").as_ref(), settings.exposure.as_ref().and_then(|e| e.exposure).unwrap_or(0.0));
        gl.uniform_1_f32(loc("u_contrast").as_ref(), settings.exposure.as_ref().and_then(|e| e.contrast).unwrap_or(1.0));
        gl.uniform_1_f32(loc("u_highlights").as_ref(), settings.exposure.as_ref().and_then(|e| e.highlights).unwrap_or(0.0));
        gl.uniform_1_f32(loc("u_shadows").as_ref(), settings.exposure.as_ref().and_then(|e| e.shadows).unwrap_or(0.0));
        gl.uniform_1_f32(loc("u_temp").as_ref(), settings.color.as_ref().and_then(|c| c.temperature).unwrap_or(0.0));
        gl.uniform_1_f32(loc("u_tint").as_ref(), settings.color.as_ref().and_then(|c| c.tint).unwrap_or(0.0));
        gl.uniform_1_f32(loc("u_grain_amount").as_ref(), settings.grain.as_ref().map(|g| g.amount).unwrap_or(0.0));
        
        let grain_size = match settings.grain.as_ref().map(|g| g.size.as_str()) {
            Some("medium") => 2.0,
            Some("coarse") => 4.0,
            _ => 1.0,
        };
        gl.uniform_1_f32(loc("u_grain_size").as_ref(), grain_size);
        gl.uniform_2_f32(loc("u_src_size").as_ref(), width as f32, height as f32);
        
        gl.viewport(0, 0, width as i32, height as i32);
        gl.draw_arrays(glow::TRIANGLES, 0, 6);
        
        Ok(())
    }
}

#[async_trait(?Send)]
impl Renderer for WebGlRenderer {
    async fn init(&mut self) -> Result<(), RendererError> {
        self.ensure_initialized(None)
    }

    async fn render(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
    ) -> Result<Vec<u8>, RendererError> {
        self.ensure_initialized(None)?;
        
        // Resize canvas if needed
        if let Some(canvas) = &self.canvas {
            if canvas.width() != width || canvas.height() != height {
                canvas.set_width(width);
                canvas.set_height(height);
            }
        }
        
        unsafe {
            self.render_internal(data, width, height, settings)?;
            
            let gl = self.context.as_ref().unwrap();
            let mut pixels = vec![0u8; (width * height * 4) as usize];
            gl.read_pixels(
                0, 0,
                width as i32, height as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(&mut pixels),
            );
            
            Ok(pixels)
        }
    }

    async fn render_to_canvas(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
        canvas: &HtmlCanvasElement,
    ) -> Result<(), RendererError> {
        // If we are rendering to a specific canvas, we need to initialize with THAT canvas context.
        // But our struct holds one context.
        // If we want to support rendering to arbitrary canvases, we need to re-initialize or share resources?
        // WebGL contexts are tied to canvases.
        // If we re-init, we lose compiled shaders/textures unless we share.
        // For "Quick Fix", we likely have one preview canvas.
        // So we should initialize with THAT canvas.
        
        // If we already have a context and it matches this canvas, good.
        // If not, we re-init.
        
        // Check if context's canvas is the same?
        // Hard to check equality of canvas objects easily without storing reference.
        // Let's assume if we call render_to_canvas, we want to use that canvas.
        
        // Optimization: If we already have a context, check if it's for this canvas.
        // If not, drop old context and create new one?
        // Or just create a new renderer instance for each canvas.
        
        // For now, let's just re-init if needed.
        // Note: This is expensive if done every frame.
        // The consumer should keep the renderer instance alive for the canvas.
        
        self.ensure_initialized(Some(canvas))?;
        
        if canvas.width() != width || canvas.height() != height {
            canvas.set_width(width);
            canvas.set_height(height);
        }
        
        unsafe {
            self.render_internal(data, width, height, settings)?;
        }
        
        Ok(())
    }
}
