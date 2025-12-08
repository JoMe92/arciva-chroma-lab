import React, { useEffect, useRef, useState } from 'react';
import { QuickFixClient } from '../../../quickfix-renderer/pkg/client.js';
import { RendererOptions, get_opaque_crop } from 'quickfix-renderer';

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
  const [highlights, setHighlights] = useState(0); // New
  const [shadows, setShadows] = useState(0); // New
  const [temp, setTemp] = useState(0);
  const [tint, setTint] = useState(0);
  const [grainAmount, setGrainAmount] = useState(0);
  const [grainSize, setGrainSize] = useState<'fine' | 'medium' | 'coarse'>('medium');
  const [rotation, setRotation] = useState(0);
  // Geometry Settings
  const [geoVertical, setGeoVertical] = useState(0); // New
  const [geoHorizontal, setGeoHorizontal] = useState(0); // New
  const [flipVertical, setFlipVertical] = useState(false);
  const [flipHorizontal, setFlipHorizontal] = useState(false);

  // Crop State
  // Draft state (what the sliders control)
  const [cropX, setCropX] = useState(0.0);
  const [cropY, setCropY] = useState(0.0);
  const [cropW, setCropW] = useState(1.0);
  const [cropH, setCropH] = useState(1.0);
  // Applied state (what is sent to backend)
  const [appliedCrop, setAppliedCrop] = useState<{ x: number, y: number, width: number, height: number } | null>(null);
  const [autoCrop, setAutoCrop] = useState(false);

  const applyCrop = () => {
    setAppliedCrop({ x: cropX, y: cropY, width: cropW, height: cropH });
  };

  const resetCrop = () => {
    setAppliedCrop(null);
  };

  // Auto Crop Effect
  useEffect(() => {
    if (autoCrop && image && Math.abs(rotation) > 0) {
      // Calculate opaque crop
      const crop = get_opaque_crop(rotation, image.width, image.height);

      // Update sliders
      setCropX(crop.x);
      setCropY(crop.y);
      setCropW(crop.width);
      setCropH(crop.height);

      // Auto-apply ONLY if we want real-time update. 
      // The user usually expects "Auto Crop" to just SET the crop.
      setAppliedCrop({
        x: crop.x,
        y: crop.y,
        width: crop.width,
        height: crop.height
      });
    }
  }, [rotation, autoCrop, image]);


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
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const settings: any = { // Use any for now as TS types might not be updated in editor yet
        exposure: { exposure, contrast, highlights, shadows },
        color: { temperature: temp, tint },
        grain: { amount: grainAmount, size: grainSize },
        crop: {
          rotation,
          // Only send rect if it is APPLIED. Otherwise send undefined (full image).
          rect: appliedCrop ? appliedCrop : undefined
        },
        geometry: {
          vertical: geoVertical,
          horizontal: geoHorizontal,
          flipVertical: flipVertical,
          flipHorizontal: flipHorizontal
        }
      };

      try {
        // Explicitly resize canvas
        if (canvasRef.current!.width !== image.width || canvasRef.current!.height !== image.height) {
          canvasRef.current!.width = image.width;
          canvasRef.current!.height = image.height;
        }

        console.log("App: Rendering with settings:", JSON.stringify(settings, null, 2));

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
          // 1. Put the rendered image (Cropped or Full)
          const buf = imageBitmap as ArrayBuffer;
          const clamped = new Uint8ClampedArray(buf);
          const imgData = new ImageData(clamped, width, height);

          if (appliedCrop) {
            // CROP MODE: Draw only the selected region
            const cropX = Math.round(appliedCrop.x * width);
            const cropY = Math.round(appliedCrop.y * height);
            const cropW = Math.round(appliedCrop.width * width);
            const cropH = Math.round(appliedCrop.height * height);

            // Clamp bounds
            const safeX = Math.max(0, cropX);
            const safeY = Math.max(0, cropY);
            // Ensure width/height don't exceed image bounds
            const safeW = Math.min(width - safeX, cropW);
            const safeH = Math.min(height - safeY, cropH);

            if (safeW > 0 && safeH > 0) {
              // Resize canvas to CROP dimensions
              if (canvasRef.current!.width !== safeW || canvasRef.current!.height !== safeH) {
                canvasRef.current!.width = safeW;
                canvasRef.current!.height = safeH;
              }

              // Create cropped bitmap
              createImageBitmap(imgData, safeX, safeY, safeW, safeH).then(bitmap => {
                ctx.drawImage(bitmap, 0, 0);
                bitmap.close();
              }).catch(err => {
                console.error("Failed to create crop bitmap:", err);
              });
            }

          } else {
            // FULL MODE: Draw full image + Overlay
            if (canvasRef.current!.width !== width || canvasRef.current!.height !== height) {
              canvasRef.current!.width = width;
              canvasRef.current!.height = height;
            }
            ctx.putImageData(imgData, 0, 0);

            // Overlay on top
            const x = Math.round(cropX * width);
            const y = Math.round(cropY * height);
            const w = Math.round(cropW * width);
            const h = Math.round(cropH * height);

            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            // Semi-transparent fill outside
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            // Top
            ctx.fillRect(0, 0, width, y);
            // Bottom
            ctx.fillRect(0, y + h, width, height - (y + h));
            // Left
            ctx.fillRect(0, y, x, h);
            // Right
            ctx.fillRect(x + w, y, width - (x + w), h);
          }
        }
      } catch (e) {
        console.error("Render failed:", e);
      } finally {
        setIsRendering(false);
      }
    };

    render();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageData, image, exposure, contrast, highlights, shadows, temp, tint, grainAmount, grainSize, rotation, appliedCrop, cropX, cropY, cropW, cropH, geoVertical, geoHorizontal, flipVertical, flipHorizontal, currentBackend]);

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

          <label>Highlights: {highlights}</label>
          <input type="range" min="-1" max="1" step="0.1" value={highlights} onChange={e => setHighlights(parseFloat(e.target.value))} />

          <label>Shadows: {shadows}</label>
          <input type="range" min="-1" max="1" step="0.1" value={shadows} onChange={e => setShadows(parseFloat(e.target.value))} />

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

          <label>
            <input type="checkbox" checked={autoCrop} onChange={e => setAutoCrop(e.target.checked)} />
            Auto Crop (Straighten)
          </label>

          <label>Vertical Skew: {geoVertical}</label>
          <input type="range" min="-0.5" max="0.5" step="0.05" value={geoVertical} onChange={e => setGeoVertical(parseFloat(e.target.value))} />


          <label>Horizontal Skew: {geoHorizontal}</label>
          <input type="range" min="-0.5" max="0.5" step="0.05" value={geoHorizontal} onChange={e => setGeoHorizontal(parseFloat(e.target.value))} />

          <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input type="checkbox" checked={flipHorizontal} onChange={e => setFlipHorizontal(e.target.checked)} />
              Flip Horizontal
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input type="checkbox" checked={flipVertical} onChange={e => setFlipVertical(e.target.checked)} />
              Flip Vertical
            </label>
          </div>

          <h3>Crop</h3>
          <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
            <button onClick={applyCrop} disabled={!!appliedCrop}>Apply Crop</button>
            <button onClick={resetCrop} disabled={!appliedCrop}>Reset Crop</button>
          </div>

          {!appliedCrop ? (
            <>
              <p style={{ fontSize: '0.8rem', color: '#666' }}>Adjust sliders to position crop box (Red).</p>
              <label>X: {cropX.toFixed(2)}</label>
              <input type="range" min="0" max="1" step="0.01" value={cropX} onChange={e => setCropX(parseFloat(e.target.value))} />
              <label>Y: {cropY.toFixed(2)}</label>
              <input type="range" min="0" max="1" step="0.01" value={cropY} onChange={e => setCropY(parseFloat(e.target.value))} />
              <label>Width: {cropW.toFixed(2)}</label>
              <input type="range" min="0" max="1" step="0.01" value={cropW} onChange={e => setCropW(parseFloat(e.target.value))} />
              <label>Height: {cropH.toFixed(2)}</label>
              <input type="range" min="0" max="1" step="0.01" value={cropH} onChange={e => setCropH(parseFloat(e.target.value))} />
            </>
          ) : (
            <div>
              <p>Crop Applied. Click Reset to adjust.</p>
              <pre style={{ fontSize: '0.7em', background: '#eee', padding: '5px' }}>
                {JSON.stringify(appliedCrop, null, 2)}
              </pre>
            </div>
          )}

          <div style={{ marginTop: '1rem', borderTop: '1px solid #eee', paddingTop: '0.5rem' }}>
            <h4>Debug Info</h4>
            <div>Backend: {currentBackend}</div>
            <div>Canvas Size: {canvasRef.current ? `${canvasRef.current.width}x${canvasRef.current.height}` : 'N/A'}</div>
            <div>Image Size: {image ? `${image.width}x${image.height}` : 'N/A'}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
