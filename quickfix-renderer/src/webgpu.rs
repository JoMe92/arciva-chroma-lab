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
    denoise_luminance: f32,
    denoise_color: f32,
    curves_intensity: f32,
    st_shadow_hue: f32,
    st_shadow_sat: f32,
    st_highlight_hue: f32,
    st_highlight_sat: f32,
    st_balance: f32,
    /// Padding to match WGSL alignment (144 bytes used + 1 padding + 2 distortion + 1 align = 160)
    _padding: f32,
    distortion_k1: f32,
    distortion_k2: f32,
    hsl_enabled: f32,
    hsl: [[f32; 4]; 8],
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
    curves_sampler: Option<wgpu::Sampler>,
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
            curves_sampler: None,
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
                // Curves Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Curves Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
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
        self.queue = Some(queue);
        self.pipeline = Some(pipeline);
        self.bind_group_layout = Some(bind_group_layout);
        self.grain_texture = Some(grain_texture);
        self.grain_sampler = Some(grain_sampler);
        self.default_lut_texture = Some(default_lut_texture);
        self.lut_sampler = Some(lut_sampler);

        // Curves Sampler
        let curves_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Curves Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.curves_sampler = Some(curves_sampler);
        self.device = Some(device);

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
            denoise_luminance: settings
                .denoise
                .as_ref()
                .map(|d| d.luminance)
                .unwrap_or(0.0),
            denoise_color: settings.denoise.as_ref().map(|d| d.color).unwrap_or(0.0),
            curves_intensity: settings.curves.as_ref().map(|c| c.intensity).unwrap_or(1.0),
            // _padding: 0.0, // Removed duplicate
            st_shadow_hue: settings
                .split_toning
                .as_ref()
                .map(|s| s.shadow_hue / 360.0)
                .unwrap_or(0.0),
            st_shadow_sat: settings
                .split_toning
                .as_ref()
                .map(|s| s.shadow_sat)
                .unwrap_or(0.0),
            st_highlight_hue: settings
                .split_toning
                .as_ref()
                .map(|s| s.highlight_hue / 360.0)
                .unwrap_or(0.0),
            st_highlight_sat: settings
                .split_toning
                .as_ref()
                .map(|s| s.highlight_sat)
                .unwrap_or(0.0),
            st_balance: settings
                .split_toning
                .as_ref()
                .map(|s| s.balance)
                .unwrap_or(0.0),
            _padding: 0.0,
            distortion_k1: settings.distortion.as_ref().map(|d| d.k1).unwrap_or(0.0),
            distortion_k2: settings.distortion.as_ref().map(|d| d.k2).unwrap_or(0.0),
            hsl_enabled: if settings.hsl.is_some() { 1.0 } else { 0.0 },
            hsl: {
                let mut hsl_data = [[0.0f32; 4]; 8];
                let centers = [
                    0.0 / 360.0,
                    30.0 / 360.0,
                    60.0 / 360.0,
                    120.0 / 360.0,
                    180.0 / 360.0,
                    240.0 / 360.0,
                    270.0 / 360.0,
                    300.0 / 360.0,
                ];
                if let Some(hsl) = &settings.hsl {
                    let ranges = [
                        &hsl.red,
                        &hsl.orange,
                        &hsl.yellow,
                        &hsl.green,
                        &hsl.aqua,
                        &hsl.blue,
                        &hsl.purple,
                        &hsl.magenta,
                    ];
                    for i in 0..8 {
                        hsl_data[i][0] = ranges[i].hue;
                        hsl_data[i][1] = ranges[i].saturation;
                        hsl_data[i][2] = ranges[i].luminance;
                        hsl_data[i][3] = centers[i];
                    }
                } else {
                    for i in 0..8 {
                        hsl_data[i][3] = centers[i];
                    }
                }
                hsl_data
            },
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

        // Curves LUT Update
        let combined_data = if let Some(curves) = &settings.curves {
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

            let mut combined = Vec::with_capacity(256 * 4);
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

                combined.push(r);
                combined.push(g);
                combined.push(b);
                combined.push(1.0); // Alpha
            }
            combined
        } else {
            let mut identity = Vec::with_capacity(256 * 4);
            for i in 0..256 {
                let v = i as f32 / 255.0;
                identity.push(v);
                identity.push(v);
                identity.push(v);
                identity.push(1.0);
            }
            identity
        };

        let curves_texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Curves Texture"),
                size: wgpu::Extent3d {
                    width: 256,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&combined_data),
        );
        let curves_view = curves_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&curves_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(self.curves_sampler.as_ref().unwrap()),
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
            denoise_luminance: settings
                .denoise
                .as_ref()
                .map(|d| d.luminance)
                .unwrap_or(0.0),
            denoise_color: settings.denoise.as_ref().map(|d| d.color).unwrap_or(0.0),
            curves_intensity: settings.curves.as_ref().map(|c| c.intensity).unwrap_or(1.0),
            // _padding: 0.0, // Removed duplicate
            st_shadow_hue: settings
                .split_toning
                .as_ref()
                .map(|s| s.shadow_hue / 360.0)
                .unwrap_or(0.0),
            st_shadow_sat: settings
                .split_toning
                .as_ref()
                .map(|s| s.shadow_sat)
                .unwrap_or(0.0),
            st_highlight_hue: settings
                .split_toning
                .as_ref()
                .map(|s| s.highlight_hue / 360.0)
                .unwrap_or(0.0),
            st_highlight_sat: settings
                .split_toning
                .as_ref()
                .map(|s| s.highlight_sat)
                .unwrap_or(0.0),
            st_balance: settings
                .split_toning
                .as_ref()
                .map(|s| s.balance)
                .unwrap_or(0.0),
            _padding: 0.0,
            distortion_k1: settings.distortion.as_ref().map(|d| d.k1).unwrap_or(0.0),
            distortion_k2: settings.distortion.as_ref().map(|d| d.k2).unwrap_or(0.0),
            _padding_align: 0.0,
            hsl: {
                let mut hsl_data = [[0.0f32; 4]; 8];
                let centers = [
                    0.0 / 360.0,
                    30.0 / 360.0,
                    60.0 / 360.0,
                    120.0 / 360.0,
                    180.0 / 360.0,
                    240.0 / 360.0,
                    270.0 / 360.0,
                    300.0 / 360.0,
                ];
                if let Some(hsl) = &settings.hsl {
                    let ranges = [
                        &hsl.red,
                        &hsl.orange,
                        &hsl.yellow,
                        &hsl.green,
                        &hsl.aqua,
                        &hsl.blue,
                        &hsl.purple,
                        &hsl.magenta,
                    ];
                    for i in 0..8 {
                        hsl_data[i][0] = ranges[i].hue;
                        hsl_data[i][1] = ranges[i].saturation;
                        hsl_data[i][2] = ranges[i].luminance;
                        hsl_data[i][3] = centers[i];
                    }
                } else {
                    for i in 0..8 {
                        hsl_data[i][3] = centers[i];
                    }
                }
                hsl_data
            },
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

        // Curves LUT Update
        let combined_data = if let Some(curves) = &settings.curves {
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

            let mut combined = Vec::with_capacity(256 * 4);
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

                combined.push(r);
                combined.push(g);
                combined.push(b);
                combined.push(1.0); // Alpha
            }
            combined
        } else {
            let mut identity = Vec::with_capacity(256 * 4);
            for i in 0..256 {
                let v = i as f32 / 255.0;
                identity.push(v);
                identity.push(v);
                identity.push(v);
                identity.push(1.0);
            }
            identity
        };

        let curves_texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Curves Texture"),
                size: wgpu::Extent3d {
                    width: 256,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&combined_data),
        );
        let curves_view = curves_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&curves_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(self.curves_sampler.as_ref().unwrap()),
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
