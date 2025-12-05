# Quick Fix Renderer Usage Examples

This document provides practical examples of how to use the `quickfix-renderer` in various scenarios.

## 1. Basic Usage (Main Thread)

This is the simplest way to use the renderer, suitable for small images or testing.

```typescript
import init, { QuickFixRenderer, RendererOptions } from 'quickfix-renderer';

async function processImage(imageData: Uint8Array, width: number, height: number) {
    // 1. Initialize WASM
    await init();

    // 2. Create Renderer (auto-detects backend)
    const renderer = await QuickFixRenderer.init();
    console.log(`Initialized with backend: ${renderer.backend}`);

    // 3. Define Adjustments
    const adjustments = {
        exposure: { exposure: 0.5, contrast: 0.2 },
        color: { temperature: 0.1 }
    };

    // 4. Process Frame
    const result = await renderer.process_frame(imageData, width, height, adjustments);

    // 5. Use Result
    console.log(`Processed ${result.width}x${result.height} image`);
    return result.data; // Uint8Array
}
```

## 2. Web Worker Usage (Recommended)

For best UI performance, run the renderer in a Web Worker.

**worker.ts**

```typescript
import init, { QuickFixRenderer, RendererOptions } from 'quickfix-renderer';

let renderer: QuickFixRenderer | null = null;

self.onmessage = async (e) => {
    const { type, payload } = e.data;

    if (type === 'INIT') {
        await init();
        // Force WebGPU if desired, or let it auto-select
        renderer = await QuickFixRenderer.init(new RendererOptions("webgpu"));
        self.postMessage({ type: 'INIT_DONE', backend: renderer.backend });
    } 
    else if (type === 'PROCESS') {
        if (!renderer) return;
        
        const { buffer, width, height, adjustments } = payload;
        
        try {
            const result = await renderer.process_frame(buffer, width, height, adjustments);
            
            // Transfer buffer back to main thread to avoid copying
            const data = result.data;
            self.postMessage(
                { type: 'FRAME_DONE', data, width: result.width, height: result.height }, 
                [data.buffer] // Transferable
            );
            
            // Optional: Free WASM memory if you copied the data out
            result.free();
        } catch (err) {
            self.postMessage({ type: 'ERROR', error: err });
        }
    }
};
```

**main.ts**

```typescript
const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

worker.postMessage({ type: 'INIT' });

worker.onmessage = (e) => {
    if (e.data.type === 'FRAME_DONE') {
        const { data, width, height } = e.data;
        // Display data on canvas...
    }
};
```

## 3. Applying Specific Effects

### Grain (Film Grain)

Grain requires a seed for deterministic results (same grain pattern every time).

```typescript
const adjustments = {
    grain: {
        amount: 0.5,      // 50% strength
        size: "medium",   // "fine", "medium", or "coarse"
        seed: 12345       // Fixed seed for consistent animation/rendering
    }
};
```

### Crop & Rotate

Cropping and rotation change the output dimensions.

```typescript
const adjustments = {
    crop: {
        rotation: 90,      // Rotate 90 degrees clockwise
        aspect_ratio: 1.0  // Crop to square
    }
};

const result = await renderer.process_frame(data, width, height, adjustments);
// result.width and result.height will be different from input!
```

## 4. Rendering Directly to Canvas

If you have an `OffscreenCanvas` (in a Worker) or an `HTMLCanvasElement` (main thread), you can render directly to it.

```typescript
// Assuming 'canvas' is an HTMLCanvasElement or OffscreenCanvas
await renderer.render_to_canvas(
    data, 
    width, 
    height, 
    { exposure: { exposure: 1.0 } }, 
    canvas
);
```

## 5. Memory Management

When processing video or running in a loop, it's good practice to free the result objects to help the WASM memory allocator.

```typescript
for (const frame of videoFrames) {
    const result = await renderer.process_frame(frame.data, w, h, adjustments);
    
    // ... use result.data ...
    
    // Explicitly free the WASM object wrapper
    result.free(); 
}
```
