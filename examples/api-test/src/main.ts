import init, { QuickFixRenderer, RendererOptions, QuickFixAdjustments } from 'quickfix-renderer';

async function run() {
    await init();

    const options = new RendererOptions("cpu", 1024);
    const renderer = await QuickFixRenderer.init(options);

    console.log("Renderer initialized with backend:", renderer.backend);

    const width = 100;
    const height = 100;
    const data = new Uint8Array(width * height * 4);
    // Fill with red
    for (let i = 0; i < data.length; i += 4) {
        data[i] = 255;     // R
        data[i + 1] = 0;   // G
        data[i + 2] = 0;   // B
        data[i + 3] = 255; // A
    }

    const adjustments: QuickFixAdjustments = {
        exposure: { exposure: 1.0 }
    };

    const result = await renderer.process_frame(data, width, height, adjustments);
    console.log("Frame processed. Result size:", result.data.length);

    document.getElementById("app")!.innerText = `Success! Backend: ${renderer.backend}, Result size: ${result.data.length}`;
}

run().catch((e) => {
    console.error(e);
    document.getElementById("app")!.innerText = `Error: ${e}`;
});
