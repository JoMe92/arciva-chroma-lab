use crate::{
    renderer::{Renderer, RendererError},
    shaders::WGSL_SHADER,
    QuickFixAdjustments,
};
use async_trait::async_trait;
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SettingsUniform {
    homography_matrix: [[f32; 4]; 3], // 48 bytes (3 vec3s aligned to vec4)
    geo_padding: f32,                 // 4 bytes
    flip_vertical: f32,
    flip_horizontal: f32,
    crop_rotation: f32,
    crop_aspect: f32,
    exposure: f32,
    contrast: f32,
    highlights: f32,
    shadows: f32,
    temp: f32,
    tint: f32,
    grain_amount: f32,
    grain_size: f32,
    src_width: f32,
    src_height: f32,
    lut_intensity: f32,
    _padding: f32, // Pad to 16 bytes alignment if needed.
                   // Previous: 16 floats exactly?
                   // geo (4), flip (4), crop_rot/asp/exp/cont (4), high/shad/temp/tint (4), grain/size/w/h (4).
                   // Total 20 floats?
                   // Let's recount.
                   // Matrix (12 floats, 4 padding in aligned vec3 cols?) -> 48 bytes.
                   // geo_padding (1) = 49th float? No matrix is separate.
                   // Struct:
                   // mat3x3 (48 bytes)
                   // geo_padding (4 bytes) -> offset 52
                   // flip_v (4) -> 56
                   // flip_h (4) -> 60
                   // crop_rot (4) -> 64
                   // crop_asp (4) -> 68
                   // exp (4) -> 72
                   // cont (4) -> 76
                   // high (4) -> 80
                   // shad (4) -> 84
                   // temp (4) -> 88
                   // tint (4) -> 92
                   // grain_amt (4) -> 96
                   // grain_sz (4) -> 100
                   // src_w (4) -> 104
                   // src_h (4) -> 108

                   // Now adding lut_intensity.
                   // lut_int (4) -> 112
                   // Struct alignment usually 16 bytes for uniform buffers.
                   // 112 is divisible by 16 (112 = 16 * 7).
                   // So we are good?
                   // Let's add padding just in case to match WGSL explicitly if needed, but WGSL is packed?
                   // WGSL `Settings` struct:
                   // ... src_width, src_height;
                   // lut_intensity;
                   // };
                   // WGSL struct size is implicitly padded to 16 bytes at end.
}

pub struct WebGpuRenderer {
    instance: wgpu::Instance,
    adapter: Option<wgpu::Adapter>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    grain_texture: Option<wgpu::Texture>,
    grain_sampler: Option<wgpu::Sampler>,
    lut_texture: Option<wgpu::Texture>,
    lut_sampler: Option<wgpu::Sampler>,
    default_lut_texture: Option<wgpu::Texture>,
}

impl Default for WebGpuRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl WebGpuRenderer {
    pub fn new() -> Self {
        Self {
            instance: wgpu::Instance::default(),
            adapter: None,
            device: None,
            queue: None,
            pipeline: None,
            bind_group_layout: None,
            grain_texture: None,
            grain_sampler: None,
            lut_texture: None,
            lut_sampler: None,
            default_lut_texture: None,
        }
    }

    async fn ensure_initialized(&mut self) -> Result<(), RendererError> {
        if self.device.is_some() {
            return Ok(());
        }

        let adapter = self
            .instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(RendererError::WebGpuNotSupported)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("QuickFix Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                },
                None,
            )
            .await
            .map_err(|e| RendererError::InitFailed(e.to_string()))?;

        // Create Bind Group Layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("QuickFix Bind Group Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Diffuse Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Diffuse Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Grain Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Grain Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // LUT Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // LUT Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Load Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("QuickFix Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL_SHADER)),
        });

        // Create Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("QuickFix Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("QuickFix Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create Grain Texture (Pre-seeded noise)
        // 256x256 noise texture
        let grain_size = 256;
        let mut rng = rand::thread_rng();
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.5, 0.15).unwrap();

        let mut grain_data = Vec::with_capacity(grain_size * grain_size * 4);
        for _ in 0..(grain_size * grain_size) {
            let v: f32 = normal.sample(&mut rng);
            let b = (v.clamp(0.0, 1.0) * 255.0) as u8;
            grain_data.extend_from_slice(&[b, b, b, 255]);
        }

        let grain_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("Grain Texture"),
                size: wgpu::Extent3d {
                    width: grain_size as u32,
                    height: grain_size as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &grain_data,
        );

        let grain_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Grain Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Default LUT (Identity 2x2x2 or just 1x1x1?)
        // 1x1x1 is fine if we don't sample it (intensity 0)
        let default_lut_size = 1;
        let default_lut_data = [0, 0, 0, 255]; // Black
        let default_lut_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("Default LUT"),
                size: wgpu::Extent3d {
                    width: default_lut_size,
                    height: default_lut_size,
                    depth_or_array_layers: default_lut_size,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &default_lut_data,
        );

        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("LUT Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        self.adapter = Some(adapter);
        self.device = Some(device);
        self.queue = Some(queue);
        self.pipeline = Some(pipeline);
        self.bind_group_layout = Some(bind_group_layout);
        self.grain_texture = Some(grain_texture);
        self.grain_sampler = Some(grain_sampler);
        self.default_lut_texture = Some(default_lut_texture);
        self.lut_sampler = Some(lut_sampler);

        Ok(())
    }
}

#[async_trait(?Send)]
impl Renderer for WebGpuRenderer {
    async fn init(&mut self) -> Result<(), RendererError> {
        self.ensure_initialized().await
    }

    async fn set_lut(&mut self, data: &[f32], size: u32) -> Result<(), RendererError> {
        self.ensure_initialized().await?;
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        // Convert f32 RGB to u8 RGBA
        // Data is size*size*size * 3 floats.
        // We need size*size*size * 4 bytes.
        let num_pixels = (size * size * size) as usize;
        let mut texture_data = Vec::with_capacity(num_pixels * 4);

        for chunk in data.chunks(3) {
            let r = (chunk[0].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (chunk[1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (chunk[2].clamp(0.0, 1.0) * 255.0) as u8;
            texture_data.extend_from_slice(&[r, g, b, 255]);
        }

        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("LUT Texture"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: size,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &texture_data,
        );

        self.lut_texture = Some(texture);
        Ok(())
    }

    async fn render(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
    ) -> Result<(Vec<u8>, Vec<u32>), RendererError> {
        self.ensure_initialized().await?;
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let bind_group_layout = self.bind_group_layout.as_ref().unwrap();

        // Upload Source Texture
        let src_texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Source Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        );

        let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let src_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Source Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create Output Texture for Readback
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Uniforms
        // Calculate Homography
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
        let h = crate::geometry::calculate_homography_from_unit_square(&corners); // [f32; 9]

        // Pack into 3 vec4s for std140 layout
        let mat = [
            [h[0], h[1], h[2], 0.0],
            [h[3], h[4], h[5], 0.0],
            [h[6], h[7], h[8], 0.0],
        ];

        let uniforms = SettingsUniform {
            homography_matrix: mat,
            geo_padding: 0.0,
            flip_vertical: if settings
                .geometry
                .as_ref()
                .and_then(|g| g.flip_vertical)
                .unwrap_or(false)
            {
                1.0
            } else {
                0.0
            },
            flip_horizontal: if settings
                .geometry
                .as_ref()
                .and_then(|g| g.flip_horizontal)
                .unwrap_or(false)
            {
                1.0
            } else {
                0.0
            },
            crop_rotation: settings
                .crop
                .as_ref()
                .and_then(|c| c.rotation)
                .unwrap_or(0.0)
                .to_radians(),
            crop_aspect: settings
                .crop
                .as_ref()
                .and_then(|c| c.aspect_ratio)
                .unwrap_or(0.0),
            exposure: settings
                .exposure
                .as_ref()
                .and_then(|e| e.exposure)
                .unwrap_or(0.0),
            contrast: settings
                .exposure
                .as_ref()
                .and_then(|e| e.contrast)
                .unwrap_or(1.0),
            highlights: settings
                .exposure
                .as_ref()
                .and_then(|e| e.highlights)
                .unwrap_or(0.0),
            shadows: settings
                .exposure
                .as_ref()
                .and_then(|e| e.shadows)
                .unwrap_or(0.0),
            temp: settings
                .color
                .as_ref()
                .and_then(|c| c.temperature)
                .unwrap_or(0.0),
            tint: settings.color.as_ref().and_then(|c| c.tint).unwrap_or(0.0),
            grain_amount: settings.grain.as_ref().map(|g| g.amount).unwrap_or(0.0),
            grain_size: match settings.grain.as_ref().map(|g| g.size.as_str()) {
                Some("medium") => 2.0,
                Some("coarse") => 4.0,
                _ => 1.0,
            },
            src_width: width as f32,
            src_height: height as f32,
            lut_intensity: if self.lut_texture.is_some() {
                settings.lut.as_ref().map(|l| l.intensity).unwrap_or(0.0)
            } else {
                0.0
            },
            _padding: 0.0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Bind Group
        let lut_view = self
            .lut_texture
            .as_ref()
            .or(self.default_lut_texture.as_ref())
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&src_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &self
                            .grain_texture
                            .as_ref()
                            .unwrap()
                            .create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(self.grain_sampler.as_ref().unwrap()),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(self.lut_sampler.as_ref().unwrap()),
                },
            ],
        });

        // Encode Commands
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        // Readback Buffer
        let u32_size = std::mem::size_of::<u32>() as u32;
        let output_buffer_size = (u32_size * width * height) as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            label: Some("Output Buffer"),
            mapped_at_creation: false,
        };
        let output_buffer = device.create_buffer(&output_buffer_desc);

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(u32_size * width),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(Some(encoder.finish()));

        // Map buffer
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.await
            .unwrap()
            .map_err(|e| RendererError::RenderFailed(e.to_string()))?;

        let data = buffer_slice.get_mapped_range().to_vec();
        output_buffer.unmap();

        let histogram = crate::operations::compute_histogram(&data);
        Ok((data, histogram))
    }

    async fn render_to_canvas(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        settings: &QuickFixAdjustments,
        canvas: &web_sys::HtmlCanvasElement,
    ) -> Result<(), RendererError> {
        self.ensure_initialized().await?;

        // For render_to_canvas, we need a surface.
        // But creating a surface requires the instance and the canvas.
        // And the surface must be compatible with the adapter.
        // If we created the adapter without a surface, it might not be compatible.
        // However, we passed `compatible_surface: None` in init.
        // In WebGPU, usually any adapter works for any canvas?
        // Let's try to create a surface.

        let instance = &self.instance;
        let target = wgpu::SurfaceTarget::Canvas(canvas.clone());
        let surface = instance
            .create_surface(target)
            .map_err(|e| RendererError::InitFailed(e.to_string()))?;
        let adapter = self.adapter.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        // Configure surface
        let caps = surface.get_capabilities(adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(device, &config);

        // We need to rebuild the pipeline if the format is different?
        // Our pipeline is built for Rgba8Unorm.
        // If surface format is different (e.g. Bgra8Unorm), we need a new pipeline or a compatible one.
        // For simplicity, let's assume we can just use a new pipeline or the same one if formats match.
        // But to be robust, we should probably create the pipeline on the fly or cache it by format.
        // For this task, let's just create a new pipeline for the surface format if it differs.

        let target_format = config.format;

        // Re-create pipeline for this format
        // (Copy-paste pipeline creation logic, or refactor. Refactoring is better but for now let's inline)
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("QuickFix Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL_SHADER)),
        });

        let bind_group_layout = self.bind_group_layout.as_ref().unwrap();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("QuickFix Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("QuickFix Pipeline Surface"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Upload Source Texture (same as render)
        let src_texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Source Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        );

        let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let src_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Source Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Uniforms (same as render)
        // Calculate Homography
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
        let h = crate::geometry::calculate_homography_from_unit_square(&corners); // [f32; 9]

        // Pack into 3 vec4s for std140 layout
        // Col 0: h[0], h[1], h[2], pad
        // Col 1: h[3], h[4], h[5], pad
        // Col 2: h[6], h[7], h[8], pad
        let mat = [
            [h[0], h[1], h[2], 0.0],
            [h[3], h[4], h[5], 0.0],
            [h[6], h[7], h[8], 0.0],
        ];

        let uniforms = SettingsUniform {
            homography_matrix: mat,
            geo_padding: 0.0,
            flip_vertical: if settings
                .geometry
                .as_ref()
                .and_then(|g| g.flip_vertical)
                .unwrap_or(false)
            {
                1.0
            } else {
                0.0
            },
            flip_horizontal: if settings
                .geometry
                .as_ref()
                .and_then(|g| g.flip_horizontal)
                .unwrap_or(false)
            {
                1.0
            } else {
                0.0
            },
            crop_rotation: settings
                .crop
                .as_ref()
                .and_then(|c| c.rotation)
                .unwrap_or(0.0)
                .to_radians(),
            crop_aspect: settings
                .crop
                .as_ref()
                .and_then(|c| c.aspect_ratio)
                .unwrap_or(0.0),
            exposure: settings
                .exposure
                .as_ref()
                .and_then(|e| e.exposure)
                .unwrap_or(0.0),
            contrast: settings
                .exposure
                .as_ref()
                .and_then(|e| e.contrast)
                .unwrap_or(1.0),
            highlights: settings
                .exposure
                .as_ref()
                .and_then(|e| e.highlights)
                .unwrap_or(0.0),
            shadows: settings
                .exposure
                .as_ref()
                .and_then(|e| e.shadows)
                .unwrap_or(0.0),
            temp: settings
                .color
                .as_ref()
                .and_then(|c| c.temperature)
                .unwrap_or(0.0),
            tint: settings.color.as_ref().and_then(|c| c.tint).unwrap_or(0.0),
            grain_amount: settings.grain.as_ref().map(|g| g.amount).unwrap_or(0.0),
            grain_size: match settings.grain.as_ref().map(|g| g.size.as_str()) {
                Some("medium") => 2.0,
                Some("coarse") => 4.0,
                _ => 1.0,
            },
            src_width: width as f32,
            src_height: height as f32,
            lut_intensity: settings.lut.as_ref().map(|l| l.intensity).unwrap_or(0.0),
            _padding: 0.0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Bind Group
        let lut_view = self
            .lut_texture
            .as_ref()
            .or(self.default_lut_texture.as_ref())
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&src_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &self
                            .grain_texture
                            .as_ref()
                            .unwrap()
                            .create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(self.grain_sampler.as_ref().unwrap()),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(self.lut_sampler.as_ref().unwrap()),
                },
            ],
        });

        // Render to Surface
        let frame = surface
            .get_current_texture()
            .map_err(|e| RendererError::RenderFailed(e.to_string()))?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
        frame.present();

        Ok(())
    }
}
