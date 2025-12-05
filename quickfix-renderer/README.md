# Quick Fix Renderer

A high-performance, Rust-based image processing library for the web, powered by WebAssembly.

## Features

- **Multi-Backend**: Automatically selects the best available backend: **WebGPU**, **WebGL2**, or **CPU**.
- **Type-Safe**: Full TypeScript support with typed configuration objects.
- **Efficient**: Designed for use in Web Workers with minimal overhead.
- **Non-Destructive**: Applies a pipeline of adjustments (Crop, Exposure, Color, Grain, Geometry) to raw pixel data.

## Installation

```bash
npm install quickfix-renderer
```

## Quick Start

```typescript
import init, { QuickFixRenderer, RendererOptions } from 'quickfix-renderer';

// Initialize WASM
await init();

// Create Renderer
const renderer = await QuickFixRenderer.init(new RendererOptions("webgpu"));

// Process Image
const result = await renderer.process_frame(
    imageData, 
    width, 
    height, 
    {
        exposure: { exposure: 0.5 },
        color: { temperature: 0.2 }
    }
);

// Use result.data (Uint8Array)
```

## Documentation

For detailed API documentation, including all available adjustment settings and configuration options, please refer to the [API Reference](../docs/api-reference.md).

## License

Proprietary / Internal Use Only.
