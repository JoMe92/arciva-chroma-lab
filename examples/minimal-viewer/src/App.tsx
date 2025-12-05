import { useState, useEffect, useRef } from 'react';
import './App.css';

// Define adjustments interface matching Rust struct
interface Adjustments {
  exposure: number;
  contrast: number;
}

function App() {
  const [adjustments, setAdjustments] = useState<Adjustments>({ exposure: 0, contrast: 0 });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    // Initialize worker
    workerRef.current = new Worker(new URL('./worker.ts', import.meta.url), {
      type: 'module'
    });

    workerRef.current.onmessage = (e) => {
      const { imageData } = e.data;
      const ctx = canvasRef.current?.getContext('2d');
      if (ctx) {
        ctx.putImageData(imageData, 0, 0);
      }
    };

    return () => {
      workerRef.current?.terminate();
    };
  }, []);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      canvas.width = img.width;
      canvas.height = img.height;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);

      // Send to worker
      workerRef.current?.postMessage({
        imageData,
        adjustments
      });
    };
    img.src = URL.createObjectURL(file);
  };

  const updateAdjustment = (key: keyof Adjustments, value: number) => {
    const newAdjustments = { ...adjustments, [key]: value };
    setAdjustments(newAdjustments);

    // We need to re-send the image data + new adjustments. 
    // For this simple example, we might need to store the original image data or read from canvas.
    // Reading from canvas is slow, but fine for a minimal example.
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Ideally we keep the original image data in memory to avoid accumulating errors
        // But for now, let's just assume the user re-uploads or we just send the current canvas (which is wrong for adjustments)
        // Better: Store original ImageData in a ref.
      }
    }
  };

  // Ref to store original image data
  const originalImageRef = useRef<ImageData | null>(null);

  // Update handleImageUpload to store original
  const handleImageUploadWithStore = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      canvas.width = img.width;
      canvas.height = img.height;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);
      originalImageRef.current = imageData;

      workerRef.current?.postMessage({
        imageData,
        adjustments
      });
    };
    img.src = URL.createObjectURL(file);
  };

  useEffect(() => {
    if (originalImageRef.current && workerRef.current) {
      workerRef.current.postMessage({
        imageData: originalImageRef.current,
        adjustments
      });
    }
  }, [adjustments]);

  return (
    <div className="App">
      <h1>Quick Fix Renderer</h1>
      <input type="file" accept="image/*" onChange={handleImageUploadWithStore} />

      <div className="controls">
        <label>
          Exposure: {adjustments.exposure}
          <input
            type="range"
            min="-2"
            max="2"
            step="0.1"
            value={adjustments.exposure}
            onChange={(e) => updateAdjustment('exposure', parseFloat(e.target.value))}
          />
        </label>
        <label>
          Contrast: {adjustments.contrast}
          <input
            type="range"
            min="-1"
            max="1"
            step="0.1"
            value={adjustments.contrast}
            onChange={(e) => updateAdjustment('contrast', parseFloat(e.target.value))}
          />
        </label>
      </div>

      <canvas ref={canvasRef} />
    </div>
  );
}

export default App;
