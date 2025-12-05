import init, { process_frame, QuickFixAdjustments } from 'quickfix-renderer';

// Initialize WASM
await init();

self.onmessage = (e) => {
    const { imageData, adjustments } = e.data;
    const { width, height, data } = imageData;

    // Create adjustments struct
    const adj = new QuickFixAdjustments();
    adj.exposure = adjustments.exposure;
    adj.contrast = adjustments.contrast;
    // Set other fields...

    // Process frame
    // Note: process_frame expects a mutable slice. In JS, we pass a Uint8Array.
    // Ideally, we should use a shared buffer or transfer the buffer.
    // For now, we just pass the data.

    // Since process_frame takes &mut [u8], wasm-bindgen handles the view.
    // However, to modify it in place and return it, we might need to be careful about ownership.
    // The current signature is: process_frame(data: &mut [u8], ...)

    try {
        process_frame(data, width, height, adj);
        // Post back the modified data
        self.postMessage({ imageData }, [imageData.data.buffer]);
    } catch (err) {
        console.error(err);
    }
};
