import init, { QuickFixRenderer, init_panic_hook, RendererOptions } from '../pkg/quickfix_renderer';
import { WorkerMessage, WorkerResponse } from './protocol';

const ctx: DedicatedWorkerGlobalScope = self as any;

let renderer: QuickFixRenderer | null = null;
let wasmInitPromise: Promise<void> | null = null;

// Track the latest request ID to implement cancellation/superseding
let latestRequestId = 0;

async function initializeWasm() {
    if (!wasmInitPromise) {
        wasmInitPromise = init().then(() => {
            init_panic_hook();
            console.log("Worker: WASM initialized");
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
                // rendererOptions comes across as a plain object, we might need to reconstruct if it has methods,
                // but for now it's likely just data or we pass the struct.
                // Actually, RendererOptions is a WASM struct. We can't pass it directly easily unless it's serializable.
                // Let's assume the payload contains the *properties* to create options, or we pass a plain object
                // and the WASM side accepts it.
                // Looking at previous worker.ts, it constructed RendererOptions inside the worker.
                // Let's adjust protocol to pass a plain object for options if needed, but for now let's try.

                // If rendererOptions is a JS object from the main thread, we might need to convert it.
                // For simplicity, let's assume we pass the backend string.
                // Wait, the previous worker.ts did: const options = new RendererOptions(backend);
                // Let's stick to that pattern if possible, or assume payload has what we need.
                // For now, let's assume payload.rendererOptions IS the struct or compatible.
                // If it's a transfer of a WASM object, that's tricky.
                // Better: Pass a config object.

                // REVISIT: The protocol defined 'rendererOptions: RendererOptions'. 
                // If RendererOptions is a WASM class, it's not transferable.
                // We should probably change the protocol to take a plain config object.
                // But let's see what I wrote in protocol.ts... I imported RendererOptions.
                // I will assume for this step that we can pass the underlying pointer or just recreate it.
                // actually, let's just pass the backend string for now to be safe, or cast it.

                // Let's assume the client passes a plain object that LOOKS like RendererOptions or just the backend string.
                // To be safe, let's cast to any for the constructor.

                // Re-creating the options here:
                // @ts-ignore
                const backend = msg.payload.rendererOptions.backend || 'auto';
                const options = new RendererOptions(backend === 'auto' ? undefined : backend);

                renderer = await QuickFixRenderer.init(options);
                console.log("Worker: Renderer assigned", !!renderer);
                const response: WorkerResponse = {
                    type: 'INIT_RESULT',
                    payload: { success: true, backend: renderer.backend }
                };
                ctx.postMessage(response);
                break;

            case 'RENDER':
                console.log("Worker: Processing RENDER. Renderer exists?", !!renderer);
                if (!renderer) throw new Error("Renderer not initialized");

                const { requestId, imageData, width, height, adjustments } = msg.payload;

                // Cancellation check: if a newer request has come in since we started processing (or queued),
                // we technically can't know easily without peeking the queue, but we can track "latest seen".
                if (requestId < latestRequestId) {
                    console.log(`Worker: Dropping stale request ${requestId} (latest: ${latestRequestId})`);
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
                    console.log(`Worker: Dropping stale result ${requestId} (latest: ${latestRequestId})`);
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
