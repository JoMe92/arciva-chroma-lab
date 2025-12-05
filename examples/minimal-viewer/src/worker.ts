import init, { QuickFixRenderer, init_panic_hook } from 'quickfix-renderer';

console.log("Worker script loaded");

let renderer: QuickFixRenderer | null = null;

// Global error handler
self.onerror = (e) => {
    console.error("Worker Global Error:", e);
};

let wasmInitPromise: Promise<void>;

// Initialize WASM non-blocking
try {
    console.log("Worker starting WASM init...");
    wasmInitPromise = init().then(() => {
        init_panic_hook();
        console.log("Worker WASM initialized");
    }).catch(e => {
        console.error("Worker WASM init failed:", e);
        throw e;
    });
} catch (e) {
    console.error("Worker script execution failed:", e);
}

const ctx: Worker = self as any;

ctx.onmessage = async (e) => {
    // Wait for WASM
    try {
        await wasmInitPromise;
    } catch (e) {
        console.error("Cannot process message, WASM failed to init");
        return;
    }

    const { type, payload, id } = e.data;
    console.log(`Worker received message: ${type}`, { id });

    try {
        if (type === 'init') {
            const { backend } = payload;
            console.log(`Worker initializing renderer with backend: ${backend}`);
            try {
                renderer = await QuickFixRenderer.create(backend === 'auto' ? undefined : backend);
                console.log(`Worker renderer created: ${renderer.backend}`);
                ctx.postMessage({ type: 'init_result', id, success: true, backend: renderer.backend });
            } catch (createErr) {
                console.error("Worker failed to create renderer:", createErr);
                throw createErr;
            }
        } else if (type === 'render') {
            if (!renderer) {
                throw new Error("Renderer not initialized");
            }

            const { imageData, width, height, adjustments } = payload;
            console.log("Worker received render request:", { width, height, dataSize: imageData.byteLength });
            if (imageData.length > 0) {
                console.log("Worker input pixel 0:", imageData[0], imageData[1], imageData[2], imageData[3]);
            }

            // Render
            // adjustments is already a plain object matching the struct structure
            const result = await renderer.render(imageData, width, height, adjustments);

            // Transfer buffer back
            const data = result.data;
            console.log("Worker render result:", { width: result.width, height: result.height, dataSize: data.byteLength });
            if (data.length > 0) {
                console.log("Worker output pixel 0:", data[0], data[1], data[2], data[3]);
            }

            ctx.postMessage({
                type: 'render_result',
                id,
                success: true,
                data: data,
                width: result.width,
                height: result.height
            }, [data.buffer]);

            result.free();
        }
    } catch (err: any) {
        console.error("Worker Error in message handler:", err);
        ctx.postMessage({ type: 'error', id, error: err.toString() });
    }
};
