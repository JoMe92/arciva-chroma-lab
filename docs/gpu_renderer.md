# GPU-Accelerated Quick Fix Renderer

This document describes the implementation of the GPU-accelerated rendering pipeline for Quick Fix image adjustments.

## Overview

The goal was to implement a high-performance, client-side rendering pipeline that matches the visual output of the existing Python/Pillow backend. The system prioritizes **WebGPU** for maximum performance, falls back to **WebGL2** for broad compatibility, and includes a **CPU** fallback for reference and legacy support.

## Architecture

The project is built using **Rust** and compiled to **WebAssembly (WASM)**.

### Core Components

1. **`QuickFixRenderer` (WASM API)**:
    * The main entry point exposed to JavaScript.
    * Handles runtime feature detection to select the best available backend (`WebGPU` -> `WebGL2` -> `CPU`).
    * Provides an asynchronous `init()` method and `render_to_canvas()` for efficient display.

2. **`Renderer` Trait**:
    * An internal Rust trait that abstracts the rendering logic.
    * Ensures all backends implement a consistent interface for initialization and rendering.

3. **Backends**:
    * **WebGPU (`src/webgpu.rs`)**: Uses `wgpu` to interface with modern GPU APIs. Implements a single-pass compute/render pipeline using WGSL shaders.
    * **WebGL2 (`src/webgl.rs`)**: Uses `glow` for OpenGL ES 3.0 bindings. Implements an equivalent pipeline using GLSL shaders.
    * **CPU (`src/lib.rs` & `src/operations.rs`)**: A pure Rust implementation using the `image` crate. It serves as the "ground truth" for visual correctness.

### Shaders & Operations

The rendering pipeline implements the following operations in a specific order to match the Python reference:

1. **Geometry**: Bilinear warping for perspective correction (approximated).
2. **Crop & Rotate**: Bicubic sampling for high-quality rotation and cropping.
3. **Exposure**: Exposure, contrast, highlights, and shadows adjustments.
4. **Color**: Temperature and Tint adjustments.
5. **Grain**: Film grain simulation using a pre-generated noise texture.

**Key Implementation Details:**

* **Bicubic Sampling**: Implemented manually in both WGSL and GLSL to ensure smooth transformations, matching Pillow's bicubic filter.
* **Grain Consistency**: Grain noise is pre-generated on the CPU using a seeded RNG (`ChaCha8Rng`) and uploaded as a texture. This ensures deterministic results across all backends and avoids expensive/complex random number generation in shaders.

## Usage

### Building

To build the WASM package:

```bash
pixi run build:wasm
# or
wasm-pack build --target web --out-dir pkg
```

### Example Application

A minimal React example is provided in `examples/minimal-viewer`.

To run it:

```bash
pixi run dev-example
```

This launches a Vite dev server where you can load an image, adjust sliders, and switch between rendering backends to compare performance and visual output.

### Integration

```typescript
import init, { QuickFixRenderer } from 'quickfix-renderer';

// Initialize WASM module
await init();

// Create renderer (automatically selects best backend)
const renderer = await new QuickFixRenderer();
console.log(`Using backend: ${renderer.backend}`);

// Render
const settings = {
  exposure: { exposure: 0.5, contrast: 1.1 },
  // ... other settings
};

await renderer.render_to_canvas(
  imageData, // Uint8Array of RGBA pixels
  width,
  height,
  settings,
  canvasElement
);
```

## Troubleshooting

### Linux WebGPU Support

WebGPU is still experimental on some Linux configurations. If `renderer.backend` defaults to `webgl2` or `cpu` despite having a capable GPU:

1. Ensure you are using a recent version of Chrome/Chromium.
2. Enable flags in `chrome://flags`:
    * `Vulkan`: Enabled
    * `WebGPU Developer Features`: Enabled
3. Launch Chrome with flags:

    ```bash
    google-chrome --enable-unsafe-webgpu --enable-features=Vulkan,UseSkiaRenderer
    ```

If WebGPU fails, the system automatically falls back to WebGL2, which offers excellent performance and is widely supported.
