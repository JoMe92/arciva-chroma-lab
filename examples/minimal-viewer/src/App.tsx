import React, { useEffect, useRef, useState } from 'react';


// Initialize WASM once


function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const pendingRequests = useRef<Map<number, { resolve: (data: any) => void, reject: (err: any) => void }>>(new Map());
  const requestCounter = useRef(0);
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
  const [grainSize, setGrainSize] = useState('medium');
  const [rotation, setRotation] = useState(0);

  // Helper to send message to worker and wait for response
  const sendWorkerMessage = (type: string, payload: any): Promise<any> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current) {
        reject(new Error("Worker not initialized"));
        return;
      }
      const id = requestCounter.current++;
      pendingRequests.current.set(id, { resolve, reject });
      workerRef.current.postMessage({ type, payload, id });
    });
  };

  // Initialize Worker
  useEffect(() => {
    console.log("App: Initializing worker...");
    const worker = new Worker(new URL('./worker.ts', import.meta.url), {
      type: 'module'
    });

    worker.onerror = (e) => {
      console.error("App: Worker Error Event:", e);
    };

    worker.onmessage = (e) => {
      const { id, success, error, ...rest } = e.data;
      const request = pendingRequests.current.get(id);
      if (request) {
        pendingRequests.current.delete(id);
        if (success) {
          request.resolve(rest);
        } else {
          console.error("App: Worker returned error:", error);
          request.reject(new Error(error));
        }
      } else {
        console.warn("App: Received message for unknown request", id);
      }
    };

    workerRef.current = worker;

    return () => {
      console.log("App: Terminating worker");
      worker.terminate();
    };
  }, []);

  // Initialize Renderer in Worker
  useEffect(() => {
    console.log("App: initBackend effect running", { backend, workerExists: !!workerRef.current });
    if (!workerRef.current) {
      console.warn("App: workerRef is null, skipping initBackend");
      return;
    }

    async function initBackend() {
      console.log("App: initBackend called");
      setCurrentBackend('initializing...');
      try {
        const res = await sendWorkerMessage('init', { backend });
        setCurrentBackend(res.backend);
        console.log(`Worker initialized backend: ${res.backend}`);
      } catch (e) {
        console.error("Failed to init backend:", e);
        setCurrentBackend("Error");
      }
    }
    initBackend();
  }, [backend]);

  // Load default image
  useEffect(() => {
    const img = new Image();
    img.src = '/sample.jpg';
    img.onload = () => {
      setImage(img);
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        const data = ctx.getImageData(0, 0, img.width, img.height).data;
        setImageData(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
        if (canvasRef.current) {
          canvasRef.current.width = img.width;
          canvasRef.current.height = img.height;
        }
      }
    };
  }, []);

  // Render Loop
  useEffect(() => {
    console.log("App: Render effect triggered", {
      hasWorker: !!workerRef.current,
      hasImageData: !!imageData,
      hasImage: !!image,
      hasCanvas: !!canvasRef.current,
      isRendering,
      currentBackend
    });

    if (!workerRef.current || !imageData || !image || !canvasRef.current || isRendering) {
      console.log("App: Render skipped due to missing dependencies or busy state");
      return;
    }
    if (currentBackend === 'initializing...' || currentBackend === 'Error') {
      console.log("App: Render skipped due to backend state:", currentBackend);
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

        console.log("Sending render request to worker", { width: image.width, height: image.height, dataSize: imageData.byteLength });
        const res = await sendWorkerMessage('render', {
          imageData,
          width: image.width,
          height: image.height,
          adjustments: settings
        });

        const { data, width, height } = res;
        console.log("Worker response:", { width, height, dataSize: data.byteLength });

        const ctx = canvasRef.current!.getContext('2d');
        if (ctx) {
          // Check if data has content
          if (data.length > 0 && data[0] === 0 && data[1] === 0 && data[2] === 0 && data[3] === 0) {
            console.warn("First pixel is transparent black!");
          }

          // Create ImageData from buffer
          const clamped = new Uint8ClampedArray(data.buffer);
          const imgData = new ImageData(clamped, width, height);

          // Handle resize if output size changed (e.g. crop)
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

    // Debounce or just run? 
    // Since we have isRendering flag, we skip if busy. 
    // But we should probably queue the latest request if one comes in while busy.
    // For now, simple skip is okay for "minimal-viewer".
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
