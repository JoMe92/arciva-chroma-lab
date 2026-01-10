use crate::{
    renderer::{Renderer, RendererError},
    shaders::{GLSL_FRAGMENT, GLSL_VERTEX},
    QuickFixAdjustments,
};
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
    lut_texture: Option<glow::Texture>,
    curves_texture: Option<glow::Texture>,

    lut_cache: Option<(Vec<f32>, u32)>,
    lut_dirty: bool,
    curves_dirty: bool,

    canvas: Option<CanvasBackend>, // Keep reference if we created context from it
}

#[derive(Clone)]
enum CanvasBackend {
    Html(HtmlCanvasElement),
    Offscreen(web_sys::OffscreenCanvas),
}

impl CanvasBackend {
    fn width(&self) -> u32 {
        match self {
            CanvasBackend::Html(c) => c.width(),
            CanvasBackend::Offscreen(c) => c.width(),
        }
    }

    fn height(&self) -> u32 {
        match self {
            CanvasBackend::Html(c) => c.height(),
            CanvasBackend::Offscreen(c) => c.height(),
        }
    }

    fn set_width(&self, width: u32) {
        match self {
            CanvasBackend::Html(c) => c.set_width(width),
            CanvasBackend::Offscreen(c) => c.set_width(width),
        }
    }

    fn set_height(&self, height: u32) {
        match self {
            CanvasBackend::Html(c) => c.set_height(height),
            CanvasBackend::Offscreen(c) => c.set_height(height),
        }
    }
}

impl PartialEq<HtmlCanvasElement> for CanvasBackend {
    fn eq(&self, other: &HtmlCanvasElement) -> bool {
        match self {
            CanvasBackend::Html(c) => c == other,
            _ => false,
        }
    }
}

impl Default for WebGlRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl WebGlRenderer {
    pub fn new() -> Self {
        Self {
            context: None,
            program: None,
            vao: None,
            texture: None,
            grain_texture: None,
            lut_texture: None,
            curves_texture: None,
            lut_cache: None,
            lut_dirty: false,
            curves_dirty: false,
            canvas: None,
        }
    }

    fn ensure_initialized(
        &mut self,
        canvas: Option<&HtmlCanvasElement>,
    ) -> Result<(), RendererError> {
        // Check if we need to re-initialize
        // If we have a context, check if it matches the requested canvas
        if let Some(current_ctx_canvas) = &self.canvas {
            if let Some(requested_canvas) = canvas {
                if current_ctx_canvas == requested_canvas {
                    // Use PartialEq for comparison
                    // Canvas matches, no need to re-initialize
                    return Ok(());
                } else {
                    // Canvas changed, we must re-initialize
                    self.context = None;
                    self.program = None;
                    self.vao = None;
                    self.texture = None;
                    self.grain_texture = None;
                    self.lut_texture = None;
                    // Keep cache and mark dirty to re-upload
                    self.lut_dirty = true;
                    self.curves_dirty = true;
                    self.canvas = None;
                }
            } else {
                // No canvas requested, but we have one. Re-initialize to offscreen or new HtmlCanvas.
                self.context = None;
                self.program = None;
                self.vao = None;
                self.texture = None;
                self.grain_texture = None;
                self.lut_texture = None;
                self.lut_dirty = true;
                self.curves_dirty = true;
                self.canvas = None;
            }
        }

        if self.context.is_some() {
            return Ok(());
        }

        // If canvas is provided, use it. Otherwise create an offscreen canvas
        let (gl, used_canvas) = if let Some(c) = canvas {
            let context = c
                .get_context("webgl2")
                .map_err(|_| RendererError::WebGl2NotSupported)?
                .ok_or(RendererError::WebGl2NotSupported)?
                .dyn_into::<web_sys::WebGl2RenderingContext>()
                .map_err(|_| RendererError::WebGl2NotSupported)?;
            (
                glow::Context::from_webgl2_context(context),
                CanvasBackend::Html(c.clone()),
            )
        } else {
            // Try creating a canvas via document (Main Thread)
            if let Some(window) = web_sys::window() {
                if let Some(doc) = window.document() {
                    if let Ok(el) = doc.create_element("canvas") {
                        if let Ok(c) = el.dyn_into::<HtmlCanvasElement>() {
                            c.set_width(1);
                            c.set_height(1);
                            let context = c
                                .get_context("webgl2")
                                .map_err(|_| RendererError::WebGl2NotSupported)?
                                .ok_or(RendererError::WebGl2NotSupported)?
                                .dyn_into::<web_sys::WebGl2RenderingContext>()
                                .map_err(|_| RendererError::WebGl2NotSupported)?;

                            (
                                glow::Context::from_webgl2_context(context),
                                CanvasBackend::Html(c),
                            )
                        } else {
                            return Err(RendererError::WebGl2NotSupported);
                        }
                    } else {
                        return Err(RendererError::WebGl2NotSupported);
                    }
                } else {
                    // Worker fallback (no document)
                    if let Ok(c) = web_sys::OffscreenCanvas::new(1, 1) {
                        let context = c
                            .get_context("webgl2")
                            .map_err(|_| RendererError::WebGl2NotSupported)?
                            .ok_or(RendererError::WebGl2NotSupported)?
                            .dyn_into::<web_sys::WebGl2RenderingContext>()
                            .map_err(|_| RendererError::WebGl2NotSupported)?;
                        (
                            glow::Context::from_webgl2_context(context),
                            CanvasBackend::Offscreen(c),
                        )
                    } else {
                        return Err(RendererError::WebGl2NotSupported);
                    }
                }
            } else {
                // Worker fallback (no window)
                if let Ok(c) = web_sys::OffscreenCanvas::new(1, 1) {
                    let context = c
                        .get_context("webgl2")
                        .map_err(|_| RendererError::WebGl2NotSupported)?
                        .ok_or(RendererError::WebGl2NotSupported)?
                        .dyn_into::<web_sys::WebGl2RenderingContext>()
                        .map_err(|_| RendererError::WebGl2NotSupported)?;
                    (
                        glow::Context::from_webgl2_context(context),
                        CanvasBackend::Offscreen(c),
                    )
                } else {
                    return Err(RendererError::WebGl2NotSupported);
                }
            }
        };

        self.canvas = Some(used_canvas);

        unsafe {
            let program = gl.create_program().map_err(RendererError::InitFailed)?;

            let vs = gl
                .create_shader(glow::VERTEX_SHADER)
                .map_err(RendererError::InitFailed)?;
            gl.shader_source(vs, GLSL_VERTEX);
            gl.compile_shader(vs);
            if !gl.get_shader_compile_status(vs) {
                return Err(RendererError::InitFailed(gl.get_shader_info_log(vs)));
            }

            let fs = gl
                .create_shader(glow::FRAGMENT_SHADER)
                .map_err(RendererError::InitFailed)?;
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
            let vao = gl
                .create_vertex_array()
                .map_err(RendererError::InitFailed)?;
            gl.bind_vertex_array(Some(vao));

            let vbo = gl.create_buffer().map_err(RendererError::InitFailed)?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            // Quad: -1,-1 to 1,1
            let vertices: [f32; 12] = [
                -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
            ];
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&vertices),
                glow::STATIC_DRAW,
            );

            let pos_loc = gl.get_attrib_location(program, "position").unwrap();
            gl.enable_vertex_attrib_array(pos_loc);
            gl.vertex_attrib_pointer_f32(pos_loc, 2, glow::FLOAT, false, 0, 0);

            // Textures
            let texture = gl.create_texture().map_err(RendererError::InitFailed)?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );

            let grain_texture = gl.create_texture().map_err(RendererError::InitFailed)?;
            gl.bind_texture(glow::TEXTURE_2D, Some(grain_texture));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::REPEAT as i32);
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );

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

            // LUT Texture
            let lut_texture = gl.create_texture().map_err(RendererError::InitFailed)?;
            gl.bind_texture(glow::TEXTURE_3D, Some(lut_texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_3D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_3D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_3D,
                glow::TEXTURE_WRAP_R,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_3D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_3D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            // Don't upload data here, will do in render if dirty
            self.lut_dirty = true;

            // Curves Texture (256x1, 3 channels)
            let curves_texture = gl.create_texture().map_err(RendererError::InitFailed)?;
            gl.bind_texture(glow::TEXTURE_2D, Some(curves_texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            self.curves_dirty = true;

            self.context = Some(gl);
            self.program = Some(program);
            self.vao = Some(vao);
            self.texture = Some(texture);
            self.grain_texture = Some(grain_texture);
            self.lut_texture = Some(lut_texture);
            self.curves_texture = Some(curves_texture);
        }

        Ok(())
    }

    unsafe fn render_internal(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
    ) -> Result<(), RendererError> {
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

        // Bind LUT texture and upload if dirty
        gl.active_texture(glow::TEXTURE2);
        gl.bind_texture(glow::TEXTURE_3D, self.lut_texture);

        if self.lut_dirty {
            if let Some((data, size)) = &self.lut_cache {
                // Upload data
                // tex_image_3d(target, level, internalformat, width, height, depth, border, format, type, pixels)
                gl.tex_image_3d(
                    glow::TEXTURE_3D,
                    0,
                    glow::RGB32F as i32, // Float texture
                    *size as i32,
                    *size as i32,
                    *size as i32,
                    0,
                    glow::RGB,
                    glow::FLOAT,
                    Some(bytemuck::cast_slice(data)),
                );
            } else {
                // Upload placeholder 1x1x1 white
                let placeholder = [1.0f32, 1.0, 1.0];
                gl.tex_image_3d(
                    glow::TEXTURE_3D,
                    0,
                    glow::RGB32F as i32,
                    1,
                    1,
                    1,
                    0,
                    glow::RGB,
                    glow::FLOAT,
                    Some(bytemuck::cast_slice(&placeholder)),
                );
            }
            self.lut_dirty = false;
        }

        // Curves LUT
        gl.active_texture(glow::TEXTURE3);
        gl.bind_texture(glow::TEXTURE_2D, self.curves_texture);
        if let Some(curves) = &settings.curves {
            // Generate combined curves LUT
            let master = crate::operations::generate_curve_lut(
                &curves
                    .master
                    .as_ref()
                    .map(|c| c.points.clone())
                    .unwrap_or_default(),
            );
            let red = crate::operations::generate_curve_lut(
                &curves
                    .red
                    .as_ref()
                    .map(|c| c.points.clone())
                    .unwrap_or_default(),
            );
            let green = crate::operations::generate_curve_lut(
                &curves
                    .green
                    .as_ref()
                    .map(|c| c.points.clone())
                    .unwrap_or_default(),
            );
            let blue = crate::operations::generate_curve_lut(
                &curves
                    .blue
                    .as_ref()
                    .map(|c| c.points.clone())
                    .unwrap_or_default(),
            );

            let mut combined_data = Vec::with_capacity(256 * 3);
            for &m in &master {
                let rx = (m * 255.0).clamp(0.0, 255.0);
                let ri = rx.floor() as usize;
                let rf = rx - ri as f32;

                let r = if ri >= 255 {
                    red[255]
                } else {
                    red[ri] * (1.0 - rf) + red[ri + 1] * rf
                };
                let g = if ri >= 255 {
                    green[255]
                } else {
                    green[ri] * (1.0 - rf) + green[ri + 1] * rf
                };
                let b = if ri >= 255 {
                    blue[255]
                } else {
                    blue[ri] * (1.0 - rf) + blue[ri + 1] * rf
                };

                combined_data.push(r);
                combined_data.push(g);
                combined_data.push(b);
            }

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB32F as i32,
                256,
                1,
                0,
                glow::RGB,
                glow::FLOAT,
                Some(bytemuck::cast_slice(&combined_data)),
            );
        } else {
            // Identity LUT
            let mut identity = Vec::with_capacity(256 * 3);
            for i in 0..256 {
                let v = i as f32 / 255.0;
                identity.push(v);
                identity.push(v);
                identity.push(v);
            }
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB32F as i32,
                256,
                1,
                0,
                glow::RGB,
                glow::FLOAT,
                Some(bytemuck::cast_slice(&identity)),
            );
        }

        // Uniforms
        let loc = |name| gl.get_uniform_location(program, name);

        gl.uniform_1_i32(loc("u_texture").as_ref(), 0);
        gl.uniform_1_i32(loc("u_grain").as_ref(), 1);
        gl.uniform_1_i32(loc("u_lut").as_ref(), 2);
        gl.uniform_1_i32(loc("u_curves").as_ref(), 3);

        let vertical = settings
            .geometry
            .as_ref()
            .and_then(|g| g.vertical)
            .unwrap_or(0.0);
        let horizontal = settings
            .geometry
            .as_ref()
            .and_then(|g| g.horizontal)
            .unwrap_or(0.0);

        let corners = crate::geometry::calculate_distortion_state(vertical, horizontal);
        let matrix = crate::geometry::calculate_homography_from_unit_square(&corners);

        gl.uniform_matrix_3_f32_slice(loc("u_homography_matrix").as_ref(), false, &matrix);
        gl.uniform_1_f32(
            loc("u_flip_vertical").as_ref(),
            if settings
                .geometry
                .as_ref()
                .and_then(|g| g.flip_vertical)
                .unwrap_or(false)
            {
                1.0
            } else {
                0.0
            },
        );
        gl.uniform_1_f32(
            loc("u_flip_horizontal").as_ref(),
            if settings
                .geometry
                .as_ref()
                .and_then(|g| g.flip_horizontal)
                .unwrap_or(false)
            {
                1.0
            } else {
                0.0
            },
        );
        gl.uniform_1_f32(
            loc("u_crop_rotation").as_ref(),
            settings
                .crop
                .as_ref()
                .and_then(|c| c.rotation)
                .unwrap_or(0.0)
                .to_radians(),
        );
        gl.uniform_1_f32(
            loc("u_crop_aspect").as_ref(),
            settings
                .crop
                .as_ref()
                .and_then(|c| c.aspect_ratio)
                .unwrap_or(0.0),
        );
        gl.uniform_1_f32(
            loc("u_exposure").as_ref(),
            settings
                .exposure
                .as_ref()
                .and_then(|e| e.exposure)
                .unwrap_or(0.0),
        );
        gl.uniform_1_f32(
            loc("u_contrast").as_ref(),
            settings
                .exposure
                .as_ref()
                .and_then(|e| e.contrast)
                .unwrap_or(1.0),
        );
        gl.uniform_1_f32(
            loc("u_highlights").as_ref(),
            settings
                .exposure
                .as_ref()
                .and_then(|e| e.highlights)
                .unwrap_or(0.0),
        );
        gl.uniform_1_f32(
            loc("u_shadows").as_ref(),
            settings
                .exposure
                .as_ref()
                .and_then(|e| e.shadows)
                .unwrap_or(0.0),
        );
        gl.uniform_1_f32(
            loc("u_temp").as_ref(),
            settings
                .color
                .as_ref()
                .and_then(|c| c.temperature)
                .unwrap_or(0.0),
        );
        gl.uniform_1_f32(
            loc("u_tint").as_ref(),
            settings.color.as_ref().and_then(|c| c.tint).unwrap_or(0.0),
        );
        gl.uniform_1_f32(
            loc("u_grain_amount").as_ref(),
            settings.grain.as_ref().map(|g| g.amount).unwrap_or(0.0),
        );
        gl.uniform_1_f32(
            loc("u_lut_intensity").as_ref(),
            if self.lut_cache.is_some() {
                settings.lut.as_ref().map(|l| l.intensity).unwrap_or(0.0)
            } else {
                0.0
            },
        );

        let grain_size = match settings.grain.as_ref().map(|g| g.size.as_str()) {
            Some("medium") => 2.0,
            Some("coarse") => 4.0,
            _ => 1.0,
        };
        gl.uniform_1_f32(loc("u_grain_size").as_ref(), grain_size);
        gl.uniform_1_f32(
            loc("u_denoise_luminance").as_ref(),
            settings
                .denoise
                .as_ref()
                .map(|d| d.luminance)
                .unwrap_or(0.0),
        );
        gl.uniform_1_f32(
            loc("u_denoise_color").as_ref(),
            settings.denoise.as_ref().map(|d| d.color).unwrap_or(0.0),
        );
        gl.uniform_1_f32(
            loc("u_curves_intensity").as_ref(),
            settings.curves.as_ref().map(|c| c.intensity).unwrap_or(1.0),
        );
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

    async fn set_lut(&mut self, data: &[f32], size: u32) -> Result<(), RendererError> {
        // We clone valid data into our cache
        self.lut_cache = Some((data.to_vec(), size));
        self.lut_dirty = true;
        Ok(())
    }

    async fn render(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
    ) -> Result<(Vec<u8>, Vec<u32>), RendererError> {
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
                0,
                0,
                width as i32,
                height as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(&mut pixels),
            );

            let histogram = crate::operations::compute_histogram(&pixels);
            Ok((pixels, histogram))
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
