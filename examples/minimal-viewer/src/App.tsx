import React, { useEffect, useRef, useState } from 'react';
import { QuickFixClient } from '../../../quickfix-renderer/pkg/client.js';
import { RendererOptions } from 'quickfix-renderer';

// Import worker URL using Vite's ?url suffix
// We need to point to the JS file in the package.
// Since 'quickfix-renderer' points to 'pkg', we can try importing from the package.
// Note: We might need to use a relative path if package resolution fails for ?url
import workerUrl from '../../../quickfix-renderer/pkg/worker.js?url';

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const clientRef = useRef<QuickFixClient | null>(null);

  const [backend, setBackend] = useState<string>('auto');
  const [currentBackend, setCurrentBackend] = useState<string>('initializing...');
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [imageData, setImageData] = useState<Uint8Array | null>(null);
  const [isRendering, setIsRendering] = useState(false);

  // Settings
  const [exposure, setExposure] = useState(0);
  const [contrast, setContrast] = useState(1);
  const [temp, setTemp] = useState(0);
  const [tint, setTint] = useState(0);
  const [grainAmount, setGrainAmount] = useState(0);
  const [grainSize, setGrainSize] = useState<'fine' | 'medium' | 'coarse'>('medium');
  const [rotation, setRotation] = useState(0);

  // Initialize Client
  useEffect(() => {
    console.log("App: Initializing QuickFixClient...");
    // Create client with the worker URL
    const client = new QuickFixClient(workerUrl);
    clientRef.current = client;

    return () => {
      console.log("App: Disposing QuickFixClient");
      client.dispose();
    };
  }, []);

  // Initialize Backend
  useEffect(() => {
    if (!clientRef.current) return;

    async function initBackend() {
      console.log("App: initBackend called with", backend);
      setCurrentBackend('initializing...');
      try {
        // We need to construct RendererOptions. 
        // Since we can't easily construct WASM struct here without init, 
        // and the client.init takes RendererOptions.
        // Wait, RendererOptions is exported from 'quickfix-renderer'.
        // But we need to have called `init()` (WASM init) in the main thread to use it?
        // NO. The worker does the WASM init.
        // The main thread just passes data.
        // My previous `worker.ts` implementation assumed `rendererOptions` payload.
        // And I used `@ts-ignore` to construct it in the worker.
        // So here I should pass a plain object that LOOKS like RendererOptions?
        // Or I should change `client.init` to take a plain object.

        // Let's assume for now I pass a plain object and cast it.
        // The worker will read `.backend` from it.
        const options = { backend } as any as RendererOptions;

        await clientRef.current!.init(options);
        // We don't get the backend back in init() promise currently (it returns void).
        // I should update client.init to return the result payload.
        // For now, let's assume success.
        setCurrentBackend(backend);
      } catch (e) {
        console.error("Failed to init backend:", e);
        setCurrentBackend("Error");
      }
    }
    initBackend();
  }, [backend]);

  // Load default image
  useEffect(() => {
    // Only load image if backend is ready (client initialized)
    // Actually, client is created in first useEffect, but initBackend runs async.
    // We should wait for 'currentBackend' to be valid (not 'initializing...' or 'Error').
    if (currentBackend === 'initializing...' || currentBackend === 'Error') return;

    const img = new Image();
    img.src = '/sample.jpg';
    img.onload = async () => {
      setImage(img);
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        const data = ctx.getImageData(0, 0, img.width, img.height).data;
        const buffer = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        setImageData(buffer);

        if (canvasRef.current) {
          canvasRef.current.width = img.width;
          canvasRef.current.height = img.height;
        }

        // Send image to worker immediately
        if (clientRef.current) {
          console.log("App: Sending image to worker...");
          // We need to copy because setImage transfers ownership
          const bufferCopy = buffer.slice();
          await clientRef.current.setImage(bufferCopy.buffer, img.width, img.height);
          console.log("App: Image sent to worker");
        }
      }
    };
  }, [currentBackend]); // Add currentBackend dependency

  // Render Loop
  useEffect(() => {
    if (!clientRef.current || !imageData || !image || !canvasRef.current || isRendering) {
      return;
    }

    const render = async () => {
      setIsRendering(true);
      const settings = {
        exposure: { exposure, contrast },
        color: { temperature: temp, tint },
        grain: { amount: grainAmount, size: grainSize },
        crop: { rotation },
        geometry: { vertical: 0, horizontal: 0 }
      };

      try {
        // Explicitly resize canvas
        if (canvasRef.current!.width !== image.width || canvasRef.current!.height !== image.height) {
          canvasRef.current!.width = image.width;
          canvasRef.current!.height = image.height;
        }

        // Stateful render: Pass null for imageData
        const res = await clientRef.current!.render(
          null,
          image.width,
          image.height,
          settings
        );

        const { imageBitmap, width, height } = res;

        const ctx = canvasRef.current!.getContext('2d');
        if (ctx) {
          // imageBitmap is ArrayBuffer (from my worker impl)
          const buf = imageBitmap as ArrayBuffer;
          const clamped = new Uint8ClampedArray(buf);
          const imgData = new ImageData(clamped, width, height);

          if (canvasRef.current!.width !== width || canvasRef.current!.height !== height) {
            canvasRef.current!.width = width;
            canvasRef.current!.height = height;
          }
          ctx.putImageData(imgData, 0, 0);
        }
      } catch (e) {
        console.error("Render failed:", e);
      } finally {
        setIsRendering(false);
      }
    };

    render();
  }, [imageData, image, exposure, contrast, temp, tint, grainAmount, grainSize, rotation, currentBackend]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', padding: '1rem' }}>
      <h1>Quick Fix GPU Renderer</h1>

      <div>
        <label>Backend: </label>
        <select value={backend} onChange={(e) => setBackend(e.target.value)}>
          <option value="auto">Auto</option>
          <option value="webgpu">WebGPU</option>
          <option value="webgl2">WebGL2</option>
          <option value="cpu">CPU</option>
        </select>
        <span> Current: {currentBackend}</span>
      </div>

      <div style={{ display: 'flex', gap: '1rem' }}>
        <canvas
          ref={canvasRef}
          key={backend}
          style={{
            border: '1px solid #ccc',
            maxWidth: '100%',
            maxHeight: '80vh',
            objectFit: 'contain',
            display: 'block'
          }}
        />

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', minWidth: '300px' }}>
          <h3>Exposure</h3>
          <label>Exposure: {exposure}</label>
          <input type="range" min="-2" max="2" step="0.1" value={exposure} onChange={e => setExposure(parseFloat(e.target.value))} />

          <label>Contrast: {contrast}</label>
          <input type="range" min="0.5" max="1.5" step="0.05" value={contrast} onChange={e => setContrast(parseFloat(e.target.value))} />

          <h3>Color</h3>
          <label>Temp: {temp}</label>
          <input type="range" min="-1" max="1" step="0.05" value={temp} onChange={e => setTemp(parseFloat(e.target.value))} />

          <label>Tint: {tint}</label>
          <input type="range" min="-1" max="1" step="0.05" value={tint} onChange={e => setTint(parseFloat(e.target.value))} />

          <h3>Grain</h3>
          <label>Amount: {grainAmount}</label>
          <input type="range" min="0" max="1" step="0.05" value={grainAmount} onChange={e => setGrainAmount(parseFloat(e.target.value))} />

          <label>Size: {grainSize}</label>
          <select value={grainSize} onChange={e => setGrainSize(e.target.value as 'fine' | 'medium' | 'coarse')}>
            <option value="fine">Fine</option>
            <option value="medium">Medium</option>
            <option value="coarse">Coarse</option>
          </select>

          <h3>Geometry</h3>
          <label>Rotation: {rotation}</label>
          <input type="range" min="-45" max="45" step="1" value={rotation} onChange={e => setRotation(parseFloat(e.target.value))} />
        </div>
      </div>
    </div>
  );
}

export default App;
