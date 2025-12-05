# Quick Fix Renderer API Reference

This document describes the TypeScript API for the `quickfix-renderer` package. This package provides a high-performance image processing pipeline using Rust and WebAssembly, capable of running on WebGPU, WebGL2, or CPU.

## Installation

```bash
pnpm add quickfix-renderer
# or
npm install quickfix-renderer
```

> **Note**: This package requires a bundler that supports WebAssembly (like Vite with `vite-plugin-wasm` or Webpack 5).

## Usage Overview

```typescript
import init, { QuickFixRenderer, RendererOptions } from 'quickfix-renderer';

// 1. Initialize the WASM module
await init();

// 2. Create the renderer (auto-selects best backend)
const renderer = await QuickFixRenderer.init(new RendererOptions());

console.log(`Using backend: ${renderer.backend}`);

// 3. Process a frame
const result = await renderer.process_frame(
  pixelData, // Uint8Array (RGBA)
  width,
  height,
  {
    exposure: { exposure: 1.0 },
    grain: { amount: 0.5, size: "fine" }
  }
);

// 4. Use the result
const processedData = result.data; // Uint8Array
```

## API Reference

### `QuickFixRenderer`

The main class for handling image processing.

#### `static init(options?: RendererOptions): Promise<QuickFixRenderer>`

Initializes a new renderer instance. This method is asynchronous because it may need to initialize GPU contexts.

- **Parameters**:
  - `options` (optional): Configuration for the renderer.
- **Returns**: A promise that resolves to a `QuickFixRenderer` instance.
- **Throws**: Error if no suitable backend (WebGPU, WebGL2, or CPU) can be initialized.

#### `process_frame(data: Uint8Array, width: number, height: number, adjustments: QuickFixAdjustments, options?: ProcessOptions): Promise<FrameResult>`

Processes a single image frame with the given adjustments.

- **Parameters**:
  - `data`: A flat `Uint8Array` containing RGBA pixel data. Length must be `width * height * 4`.
  - `width`: Width of the image in pixels.
  - `height`: Height of the image in pixels.
  - `adjustments`: An object defining the edits to apply (see `QuickFixAdjustments`).
  - `options` (optional): Per-frame processing options.
- **Returns**: A promise resolving to a `FrameResult`.

#### `render_to_canvas(data: Uint8Array, width: number, height: number, adjustments: QuickFixAdjustments, canvas: HTMLCanvasElement): Promise<void>`

Helper method to render directly to an HTML Canvas element.

- **Parameters**:
  - Same as `process_frame`, but takes a `canvas` element instead of returning data.
- **Returns**: Promise resolving when rendering is complete.

#### `get backend(): string`

Returns the name of the currently active backend: `"webgpu"`, `"webgl2"`, or `"cpu"`.

---

### `RendererOptions`

Configuration object for initializing the renderer.

```typescript
class RendererOptions {
  constructor(preferredBackend?: string | null, maxPreviewSize?: number | null);
  
  preferredBackend?: string; // "webgpu", "webgl2", or "cpu"
  maxPreviewSize?: number;   // Optional limit for internal buffers
}
```

- **preferredBackend**: Hints which backend to try first. If the preferred backend is not available, it will fall back to others unless forced (implementation detail: currently falls back gracefully).
  - Example: `new RendererOptions("webgpu")`

---

### `QuickFixAdjustments`

The core data structure for defining image edits. All fields are optional; omitted fields are treated as "no change".

```typescript
interface QuickFixAdjustments {
  crop?: CropSettings;
  exposure?: ExposureSettings;
  color?: ColorSettings;
  grain?: GrainSettings;
  geometry?: GeometrySettings;
}
```

#### `CropSettings`

```typescript
interface CropSettings {
  rotation?: number;     // Degrees (e.g., 90, -90)
  aspect_ratio?: number; // Target aspect ratio (width / height)
}
```

#### `ExposureSettings`

```typescript
interface ExposureSettings {
  exposure?: number;             // -1.0 to 1.0
  contrast?: number;             // -1.0 to 1.0
  highlights?: number;           // -1.0 to 1.0
  highlight_saturation?: number; // -1.0 to 1.0
  shadows?: number;              // -1.0 to 1.0
  shadow_saturation?: number;    // -1.0 to 1.0
}
```

#### `ColorSettings`

```typescript
interface ColorSettings {
  temperature?: number; // -1.0 (Cool) to 1.0 (Warm)
  tint?: number;        // -1.0 (Green) to 1.0 (Magenta)
}
```

#### `GrainSettings`

```typescript
interface GrainSettings {
  amount: number;                     // 0.0 to 1.0
  size: "fine" | "medium" | "coarse"; // Grain texture size
  seed?: number;                      // Optional seed for deterministic noise
}
```

#### `GeometrySettings`

```typescript
interface GeometrySettings {
  vertical?: number;   // Perspective correction
  horizontal?: number; // Perspective correction
}
```

---

### `FrameResult`

The result of a `process_frame` call.

```typescript
class FrameResult {
  readonly data: Uint8Array; // The processed RGBA pixel data
  readonly width: number;    // Output width
  readonly height: number;   // Output height
  free(): void;              // Explicitly release WASM memory (optional but recommended for long-running apps)
}
```

## Best Practices

1. **Reuse the Renderer**: Initialize `QuickFixRenderer` once and reuse it. Re-initializing creates new GPU contexts and compiles shaders, which is expensive.
2. **Web Workers**: It is highly recommended to run this renderer in a Web Worker to avoid blocking the main UI thread, especially when processing large images or using the CPU fallback.
3. **Memory Management**: While the JS garbage collector handles most things, calling `.free()` on `FrameResult` objects when you are done with them can help reduce WASM memory pressure in tight loops.
