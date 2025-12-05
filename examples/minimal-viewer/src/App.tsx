import React, { useEffect, useRef, useState } from 'react';
import init, { QuickFixRenderer } from 'quickfix-renderer';

// Initialize WASM once
await init();

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [renderer, setRenderer] = useState<QuickFixRenderer | null>(null);
  const [backend, setBackend] = useState<string>('auto');
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [imageData, setImageData] = useState<Uint8Array | null>(null);

  // Settings
  const [exposure, setExposure] = useState(0);
  const [contrast, setContrast] = useState(1);
  const [temp, setTemp] = useState(0);
  const [tint, setTint] = useState(0);
  const [grainAmount, setGrainAmount] = useState(0);
  const [grainSize, setGrainSize] = useState('medium');
  const [rotation, setRotation] = useState(0);

  // Load default image
  useEffect(() => {
    const img = new Image();
    img.src = '/sample.jpg'; // Ensure this exists or use a placeholder
    img.onload = () => {
      setImage(img);

      // Get raw bytes
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        const data = ctx.getImageData(0, 0, img.width, img.height).data;
        setImageData(new Uint8Array(data.buffer));
      }
    };
  }, []);

  // Initialize Renderer
  useEffect(() => {
    async function initRenderer() {
      try {
        const r = await new QuickFixRenderer(backend === 'auto' ? undefined : backend);
        setRenderer(r);
        console.log(`Renderer initialized: ${r.backend}`);
      } catch (e) {
        console.error("Failed to init renderer:", e);
      }
    }
    initRenderer();
  }, [backend]);

  // Render Loop
  useEffect(() => {
    if (!renderer || !imageData || !image || !canvasRef.current) return;

    const render = async () => {
      const settings = {
        exposure: { exposure, contrast },
        color: { temperature: temp, tint },
        grain: { amount: grainAmount, size: grainSize },
        crop: { rotation },
        geometry: { vertical: 0, horizontal: 0 } // TODO: Add sliders
      };

      try {
        const start = performance.now();
        await renderer.render_to_canvas(
          imageData,
          image.width,
          image.height,
          settings,
          canvasRef.current
        );
        const end = performance.now();
        console.log(`Render time: ${(end - start).toFixed(2)}ms`);
      } catch (e) {
        console.error("Render failed:", e);
      }
    };

    render();
  }, [renderer, imageData, image, exposure, contrast, temp, tint, grainAmount, grainSize, rotation]);

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
        <span> Current: {renderer?.backend}</span>
      </div>

      <div style={{ display: 'flex', gap: '1rem' }}>
        <canvas
          ref={canvasRef}
          style={{ border: '1px solid #ccc', maxWidth: '100%', maxHeight: '600px' }}
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
          <select value={grainSize} onChange={e => setGrainSize(e.target.value)}>
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
