/**
 * Web Worker entry point for the Quick Fix Renderer.
 * Manages the WASM renderer instance and handles messages from the main thread.
 */

import init, { QuickFixRenderer, init_panic_hook, RendererOptions } from '../pkg/quickfix_renderer';
import { WorkerMessage, WorkerResponse } from './protocol';

const ctx: DedicatedWorkerGlobalScope = self as any;

let renderer: QuickFixRenderer | null = null;
let wasmInitPromise: Promise<void> | null = null;

// Track the latest request ID to implement cancellation/superseding
let latestRequestId = 0;

/**
 * Initializes the WASM module if not already initialized.
 */
async function initializeWasm() {
    if (!wasmInitPromise) {
        wasmInitPromise = init().then(() => {
            init_panic_hook();
        });
    }
    return wasmInitPromise;
}

ctx.onmessage = async (e: MessageEvent<WorkerMessage>) => {
    const msg = e.data;

    try {
        switch (msg.type) {
            case 'INIT':
                await initializeWasm();
                const { rendererOptions } = msg.payload;

                // Reconstruct options from payload
                // @ts-ignore
                const backend = msg.payload.rendererOptions.backend || 'auto';
                const options = new RendererOptions(backend === 'auto' ? undefined : backend);

                renderer = await QuickFixRenderer.init(options);
                const response: WorkerResponse = {
                    type: 'INIT_RESULT',
                    payload: { success: true, backend: renderer.backend }
                };
                ctx.postMessage(response);
                break;

            case 'RENDER':
                if (!renderer) throw new Error("Renderer not initialized");

                const { requestId, imageData, width, height, adjustments } = msg.payload;

                // Cancellation check: if a newer request has come in since we started processing (or queued),
                // we technically can't know easily without peeking the queue, but we can track "latest seen".
                if (requestId < latestRequestId) {
                    return;
                }
                latestRequestId = requestId;

                const startTime = performance.now();

                // Process frame
                // We need to handle ImageBitmap or ArrayBuffer.
                // The WASM signature likely expects Uint8Array (Clamped).
                let data: Uint8Array;
                if (imageData instanceof ImageBitmap) {
                    // If it's an ImageBitmap, we might need to draw it to an OffscreenCanvas to get bytes,
                    // OR the renderer supports ImageBitmap directly (unlikely for raw pixel manipulation unless webgl).
                    // The previous worker used `imageData` which was likely a Uint8ClampedArray from ImageData.
                    // Let's assume for now we get a buffer or we convert.
                    // If we get ImageBitmap, we need to extract pixels.
                    const osc = new OffscreenCanvas(width, height);
                    const osCtx = osc.getContext('2d');
                    if (!osCtx) throw new Error("Could not get OffscreenCanvas context");
                    osCtx.drawImage(imageData, 0, 0);
                    const id = osCtx.getImageData(0, 0, width, height);
                    data = new Uint8Array(id.data.buffer);
                } else {
                    data = new Uint8Array(imageData as ArrayBuffer);
                }

                const result = await renderer.process_frame(data, width, height, adjustments);
                const endTime = performance.now();

                // Check cancellation again before sending back
                if (requestId < latestRequestId) {
                    result.free();
                    return;
                }

                const resultData = result.data; // Uint8Array
                const resultBuffer = resultData.buffer as ArrayBuffer; // Ensure ArrayBuffer

                const frameResponse: WorkerResponse = {
                    type: 'FRAME_READY',
                    payload: {
                        requestId,
                        imageBitmap: resultBuffer, // Sending buffer back for now to be safe/simple
                        width: result.width,
                        height: result.height,
                        timing: endTime - startTime
                    }
                };

                // Transfer the buffer
                ctx.postMessage(frameResponse, [resultBuffer]);
                result.free();
                break;

            case 'CANCEL':
                // Update latest request ID to effectively cancel any in-flight work for older IDs
                if (msg.payload.requestId > latestRequestId) {
                    latestRequestId = msg.payload.requestId;
                }
                break;

            case 'DISPOSE':
                if (renderer) {
                    renderer.free();
                    renderer = null;
                }
                ctx.close();
                break;
        }
    } catch (err: any) {
        console.error("Worker Error:", err);
        const errorResponse: WorkerResponse = {
            type: 'ERROR',
            payload: {
                requestId: (msg as any).payload?.requestId,
                error: err.toString()
            }
        };
        ctx.postMessage(errorResponse);
    }
};
