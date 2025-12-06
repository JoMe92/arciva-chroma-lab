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

## Build & Test

### Prerequisites

- Rust (latest stable)
- `wasm-pack`

### Building

To build the package for web (ESM):

```bash
wasm-pack build --target web
```

This generates the `pkg/` directory containing the WASM binary and JS bindings.

### Testing

Run unit tests (headless):

```bash
cargo test
```

Run WASM tests (requires browser):

```bash
wasm-pack test --headless --firefox
```

## Feature Flags

The renderer supports multiple backends, controlled by Cargo features and runtime options.

| Feature | Description | Default |
|---------|-------------|---------|
| `webgpu` | Enables WebGPU backend (requires browser support) | No |
| `webgl2` | Enables WebGL2 backend | No |
| `cpu` | Enables CPU fallback backend | Yes |

**Fallback Order:**

1. **WebGPU** (if enabled & supported)
2. **WebGL2** (if enabled & supported)
3. **CPU** (always available if enabled)

To build with specific features:

```bash
# Enable all backends
wasm-pack build --target web --features "webgpu webgl2 cpu"
```

## Debugging

### Force Backend

You can force a specific backend during initialization to isolate issues:

```typescript
// Force CPU backend
const renderer = await QuickFixRenderer.init(new RendererOptions("cpu"));
```

### Common Issues

- **"Context lost"**: The GPU context was lost. The renderer does not currently auto-recover; re-initialization is required.
- **WASM Panic**: Check the browser console. `console_error_panic_hook` is enabled by default in debug builds, providing stack traces.
- **Missing WebGPU**: Ensure you are in a secure context (HTTPS or localhost) and the browser supports WebGPU.
