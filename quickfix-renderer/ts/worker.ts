/**
 * Web Worker entry point for the Quick Fix Renderer.
 * Manages the WASM renderer instance and handles messages from the main thread.
 */

import init, { QuickFixRenderer, init_panic_hook, RendererOptions, parse_lut } from '../pkg/quickfix_renderer';
import { WorkerMessage, WorkerResponse } from './protocol';

const ctx: DedicatedWorkerGlobalScope = self as any;

let renderer: QuickFixRenderer | null = null;
let wasmInitPromise: Promise<void> | null = null;

// Track the latest request ID to implement cancellation/superseding
let latestRequestId = 0;

// Store source image data to avoid re-transferring on every render
let sourceImage: Uint8Array | null = null;

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

            case 'SET_IMAGE':
                const { imageData: newImage, width: w, height: h } = msg.payload;
                if (newImage instanceof ImageBitmap) {
                    const osc = new OffscreenCanvas(w, h);
                    const osCtx = osc.getContext('2d');
                    if (!osCtx) throw new Error("Could not get OffscreenCanvas context");
                    osCtx.drawImage(newImage, 0, 0);
                    const id = osCtx.getImageData(0, 0, w, h);
                    sourceImage = new Uint8Array(id.data.buffer);
                } else {
                    sourceImage = new Uint8Array(newImage as ArrayBuffer);
                }
                // console.log("Worker: Image set", w, h);
                break;

            case 'UPLOAD_LUT':
                if (!renderer) throw new Error("Renderer not initialized");
                try {
                    const { content, extension } = msg.payload;
                    // parse_lut is now the generic function
                    const lutResult = parse_lut(content, extension);
                    await renderer.set_lut(lutResult);
                    // console.log("Worker: LUT parsed and set");
                } catch (e) {
                    console.error("Worker: Failed to parse/set LUT", e);
                    throw e;
                }
                break;

            case 'FINAL_RENDER':
                if (!renderer) throw new Error("Renderer not initialized");

                const { requestId: fReqId, imageData: fImgData, width: fW, height: fH, adjustments: fAdj } = msg.payload;

                // Simple Cancellation check
                if (fReqId < latestRequestId) {
                    return;
                }
                // Don't update latestRequestId for final render? It's a parallel/long running task?
                // Actually, export acts as a render. If user changes sliders during export, we probably want to cancel export?
                // Or maybe export blocking?
                // Let's assume it respects same ID.
                latestRequestId = fReqId;

                // Determine which image data to use
                let fData: Uint8Array;
                if (fImgData) {
                    if (fImgData instanceof ImageBitmap) {
                        const osc = new OffscreenCanvas(fW, fH);
                        const osCtx = osc.getContext('2d');
                        if (!osCtx) throw new Error("Could not get OffscreenCanvas context");
                        osCtx.drawImage(fImgData, 0, 0);
                        const id = osCtx.getImageData(0, 0, fW, fH);
                        fData = new Uint8Array(id.data.buffer);
                    } else {
                        fData = new Uint8Array(fImgData as ArrayBuffer);
                    }
                } else if (sourceImage) {
                    fData = sourceImage;
                } else {
                    throw new Error("No image data provided and no source image set");
                }

                const fResult = await renderer.final_render(fData, fW, fH, fAdj);

                if (fReqId < latestRequestId) {
                    fResult.free();
                    return;
                }

                const fResData = fResult.data;
                const fResBuffer = fResData.buffer as ArrayBuffer;

                const fResponse: WorkerResponse = {
                    type: 'FINAL_RENDER_READY',
                    payload: {
                        requestId: fReqId,
                        data: fResBuffer,
                        width: fResult.width, // Should match fW usually, unless crop logic in WASM changed it (Wait, final_render IS crop aware)
                        height: fResult.height
                    }
                };
                ctx.postMessage(fResponse, [fResBuffer]);
                fResult.free();
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

                // Determine which image data to use
                let data: Uint8Array;
                if (imageData) {
                    // Use provided image data (stateless mode)
                    if (imageData instanceof ImageBitmap) {
                        const osc = new OffscreenCanvas(width, height);
                        const osCtx = osc.getContext('2d');
                        if (!osCtx) throw new Error("Could not get OffscreenCanvas context");
                        osCtx.drawImage(imageData, 0, 0);
                        const id = osCtx.getImageData(0, 0, width, height);
                        data = new Uint8Array(id.data.buffer);
                    } else {
                        data = new Uint8Array(imageData as ArrayBuffer);
                    }
                } else if (sourceImage) {
                    // Use stored image data (stateful mode)
                    data = sourceImage;
                } else {
                    throw new Error("No image data provided and no source image set");
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
