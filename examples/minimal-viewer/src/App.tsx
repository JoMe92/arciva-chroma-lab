import React, { useEffect, useRef, useState } from 'react';
import { QuickFixClient } from '../../../quickfix-renderer/pkg/client.js';
import { RendererOptions } from 'quickfix-renderer';
import { Histogram } from './Histogram';
import { CurveEditor } from './CurveEditor';
import type { CurvesSettings } from './CurveEditor';

// Ported from geometry.rs to avoid WASM init on main thread
function getOpaqueCrop(rotationDegrees: number, width: number, height: number) {
  if (width <= 0 || height <= 0) return { x: 0, y: 0, width: 0, height: 0 };

  const angleRad = Math.abs(rotationDegrees * Math.PI / 180.0);
  if (angleRad < 1e-5) return { x: 0, y: 0, width: 1, height: 1 };

  // Calculate largest interior rectangle
  const sinA = Math.sin(angleRad);
  const cosA = Math.cos(angleRad);

  const num1 = width;
  const den1 = width * cosA + height * sinA;

  const num2 = height;
  const den2 = width * sinA + height * cosA;

  const k1 = num1 / den1;
  const k2 = num2 / den2;

  const k = Math.min(k1, k2);

  const newW = width * k;
  const newH = height * k;

  const nw = newW / width;
  const nh = newH / height;

  return {
    x: (1.0 - nw) / 2.0,
    y: (1.0 - nh) / 2.0,
    width: nw,
    height: nh,
  };
}

// Import worker URL using Vite's ?url suffix
// We need to point to the JS file in the package.
// Since 'quickfix-renderer' points to 'pkg', we can try importing from the package.
// Note: We might need to use a relative path if package resolution fails for ?url
import workerUrl from '../../../quickfix-renderer/pkg/worker.js?url';

// Helper for White Balance Calculation
function calculateWhiteBalance(r: number, g: number, b: number) {
  // Avoid division by zero
  if (r + b === 0) return { temp: 0, tint: 0 };

  // 1. Calculate Temperature (T)
  // Formula derived from shader: T = (B - R) / (0.25 * (R + B))
  const temp = (b - r) / (0.25 * (r + b));

  // 2. Calculate Tint (t)
  // Let X = R * (1 + 0.25 * T)
  // Formula derived from shader: t = (G - X) / (0.1 * X + 0.2 * G)
  const X = r * (1.0 + 0.25 * temp);

  const denom = 0.1 * X + 0.2 * g;
  // If denom is too small, assume tint is 0
  if (Math.abs(denom) < 1e-5) return { temp: temp, tint: 0 };

  const tint = (g - X) / denom;

  return { temp, tint };
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const clientRef = useRef<QuickFixClient | null>(null);

  const [backend, setBackend] = useState<string>('auto');
  const [currentBackend, setCurrentBackend] = useState<string>('initializing...');
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [imageData, setImageData] = useState<Uint8Array | null>(null);
  const [isRendering, setIsRendering] = useState(false);
  const [histogramData, setHistogramData] = useState<number[]>([]);

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
  // LUT State
  const [lutIntensity, setLutIntensity] = useState(1.0);
  const [lutName, setLutName] = useState<string | null>(null);

  // Denoise
  const [denoiseLuminance, setDenoiseLuminance] = useState(0);
  const [denoiseColor, setDenoiseColor] = useState(0);

  // Interaction State
  const [isPickingWB, setIsPickingWB] = useState(false);

  // Split Toning
  const [stShadowHue, setStShadowHue] = useState(210); // Default Teal-ish
  const [stShadowSat, setStShadowSat] = useState(0);
  const [stHighlightHue, setStHighlightHue] = useState(30); // Default Orange-ish
  const [stHighlightSat, setStHighlightSat] = useState(0);

  const [stBalance, setStBalance] = useState(0);

  // Vignette
  const [vAmount, setVAmount] = useState(0);
  const [vMidpoint, setVMidpoint] = useState(0.5);
  const [vRoundness, setVRoundness] = useState(0);

  const [vFeather, setVFeather] = useState(0.5);

  // Sharpen / Clarity / Dehaze
  const [sharpenAmount, setSharpenAmount] = useState(0);
  const [sharpenRadius, setSharpenRadius] = useState(1.0);
  const [sharpenThreshold, setSharpenThreshold] = useState(0);

  const [clarityAmount, setClarityAmount] = useState(0);
  const [dehazeAmount, setDehazeAmount] = useState(0);


  const handleLutUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    const text = await file.text();

    // Using implicit import from module scope or dynamic import if needed.
    // The parse_cube_lut is exported from 'quickfix-renderer'.
    // We need to import it. 
    // Since this is inside App component, we can assume imports are available or use valid import.
    // We can assume imports are available or use valid import.

    const extension = file.name.split('.').pop() || '';

    try {
      if (clientRef.current) {
        console.log("App: Uploading LUT content string to worker...");
        await clientRef.current.uploadLut(text, extension);
        setLutName(file.name);
        console.log("App: LUT uploaded");
      }
    } catch (err) {
      console.error("Failed to upload LUT", err);
      alert("Failed to load LUT: " + err);
    }
  };

  // Geometry Settings
  const [geoVertical, setGeoVertical] = useState(0);
  const [geoHorizontal, setGeoHorizontal] = useState(0); // New
  const [flipVertical, setFlipVertical] = useState(false);
  const [flipHorizontal, setFlipHorizontal] = useState(false);

  const [curves, setCurves] = useState<CurvesSettings>({
    intensity: 1.0,
  });

  interface HslRange {
    hue: number;
    saturation: number;
    luminance: number;
  }

  interface HslSettings {
    red: HslRange;
    orange: HslRange;
    yellow: HslRange;
    green: HslRange;
    aqua: HslRange;
    blue: HslRange;
    purple: HslRange;
    magenta: HslRange;
  }

  const [hsl, setHsl] = useState<HslSettings>({
    red: { hue: 0, saturation: 0, luminance: 0 },
    orange: { hue: 0, saturation: 0, luminance: 0 },
    yellow: { hue: 0, saturation: 0, luminance: 0 },
    green: { hue: 0, saturation: 0, luminance: 0 },
    aqua: { hue: 0, saturation: 0, luminance: 0 },
    blue: { hue: 0, saturation: 0, luminance: 0 },
    purple: { hue: 0, saturation: 0, luminance: 0 },
    magenta: { hue: 0, saturation: 0, luminance: 0 },
  });

  const setCurvesIntensity = (intensity: number) => {
    setCurves(prev => ({ ...prev, intensity }));
  };

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
      const crop = getOpaqueCrop(rotation, image.width, image.height);

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

  // Auto-cancel WB Picker if Geometry/Rotation changes
  useEffect(() => {
    if (isPickingWB) {
      if (Math.abs(rotation) > 0 || Math.abs(geoVertical) > 0 || Math.abs(geoHorizontal) > 0) {
        setIsPickingWB(false);
      }
    }
  }, [rotation, geoVertical, geoHorizontal, isPickingWB]);


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
        lut: { intensity: lutIntensity }, // New
        denoise: { luminance: denoiseLuminance, color: denoiseColor },
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
        },
        curves: curves,
        hsl: hsl,
        splitToning: {
          shadowHue: stShadowHue,
          shadowSat: stShadowSat,
          highlightHue: stHighlightHue,
          highlightSat: stHighlightSat,
          balance: stBalance
        },
        vignette: {
          amount: vAmount,
          midpoint: vMidpoint,
          roundness: vRoundness,
          feather: vFeather
        },
        sharpen: {
          amount: sharpenAmount,
          radius: sharpenRadius,
          threshold: sharpenThreshold
        },
        clarity: {
          amount: clarityAmount
        },
        dehaze: {
          amount: dehazeAmount
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

        const { imageBitmap, width, height, histogram } = res;
        if (histogram) {
          setHistogramData(histogram);
        }

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
  }, [imageData, image, exposure, contrast, highlights, shadows, temp, tint, grainAmount, grainSize, rotation, appliedCrop, cropX, cropY, cropW, cropH, geoVertical, geoHorizontal, flipVertical, flipHorizontal, currentBackend, lutIntensity, denoiseLuminance, denoiseColor, curves, hsl, stShadowHue, stShadowSat, stHighlightHue, stHighlightSat, stBalance, vAmount, vMidpoint, vRoundness, vFeather, sharpenAmount, sharpenRadius, sharpenThreshold, clarityAmount, dehazeAmount]);

  // Handle Canvas Click for WB Picking
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Guard: Basic checks
    if (!isPickingWB || !imageData || !image || !canvasRef.current) return;

    // Guard: Geometry must be neutral (Rotation/Perspective makes mapping complex)
    if (Math.abs(rotation) > 0 || Math.abs(geoVertical) > 0 || Math.abs(geoHorizontal) > 0) {
      console.warn("WB Picker: Cannot pick with active geometry transforms.");
      setIsPickingWB(false);
      return;
    }

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Map screen coordinates to canvas internal resolution
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;

    const canvasX = Math.floor(x * scaleX);
    const canvasY = Math.floor(y * scaleY);

    let imgX = canvasX;
    let imgY = canvasY;

    if (appliedCrop) {
      // If cropped, map canvas coordinates back to original image coordinates
      // See render loop logic: safeX, safeY were used to crop existing image
      const width = image.width;
      const height = image.height;
      const cropX = Math.round(appliedCrop.x * width);
      const cropY = Math.round(appliedCrop.y * height);

      const safeX = Math.max(0, cropX);
      const safeY = Math.max(0, cropY);

      imgX = safeX + canvasX;
      imgY = safeY + canvasY;
    }

    // Boundary check
    if (imgX < 0 || imgX >= image.width || imgY < 0 || imgY >= image.height) {
      return;
    }

    // Handle Flips (Map visual coordinate to source coordinate)
    // If displayed image is flipped, the pixel at 'imgX' corresponds to 'width - imgX' in source
    if (flipHorizontal) {
      // Logic: If appliedCrop is active, the crop rect itself was flipped? 
      // Current pipeline: Flips happen LAST in shader (closest to resource)?
      // No, usually flips are applied to the whole image.
      // If we are in Crop mode, we are seeing a crop of the flipped image?
      // Let's assume Flip is applied to the WHOLE image space.
      // So Source(x) = Width - 1 - Display(x).
      // If Cropped: We have mapped Display(x) -> ImageSpace(x) via safeX offset.
      // Now invert the global flip.
      imgX = image.width - 1 - imgX;
    }
    if (flipVertical) {
      imgY = image.height - 1 - imgY;
    }

    // Sample from ORIGINAL image data
    const idx = (imgY * image.width + imgX) * 4;
    const r = imageData[idx];
    const g = imageData[idx + 1];
    const b = imageData[idx + 2];

    console.log(`WB Pick at (${imgX}, ${imgY}): R=${r}, G=${g}, B=${b}`);

    const res = calculateWhiteBalance(r, g, b);
    console.log("Calculated WB:", res);

    setTemp(res.temp);
    setTint(res.tint);
    setIsPickingWB(false); // Disable picker after selection
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden', background: '#111', color: '#eee', fontFamily: 'Inter, system-ui, sans-serif' }}>
      <header style={{ padding: '0.5rem 1rem', borderBottom: '1px solid #333', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h1 style={{ margin: 0, fontSize: '1.2rem' }}>Quick Fix GPU Renderer</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div>
            <label style={{ fontSize: '0.8rem', opacity: 0.7 }}>Backend: </label>
            <select
              value={backend}
              onChange={(e) => setBackend(e.target.value)}
              style={{ background: '#222', color: '#eee', border: '1px solid #444', borderRadius: '4px', fontSize: '0.8rem' }}
            >
              <option value="auto">Auto</option>
              <option value="webgpu">WebGPU</option>
              <option value="webgl2">WebGL2</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.2rem' }}>
            <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Current: {currentBackend}</span>
            <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Image Size: {image?.width}x{image?.height}</span>
          </div>
        </div>
      </header>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <main style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem', overflow: 'auto', background: '#000' }}>
          <canvas
            ref={canvasRef}
            key={backend}
            onClick={handleCanvasClick}
            style={{
              border: '1px solid #333',
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
              display: 'block',
              cursor: isPickingWB ? 'crosshair' : 'default',
              boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
            }}
          />
        </main>

        <aside aria-label="Sidebar controls" style={{
          width: '350px',
          borderLeft: '1px solid #333',
          overflowY: 'auto',
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem',
          background: '#111'
        }}>
          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Exposure</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Exposure</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{exposure}</span>
                </div>
                <input type="range" min="-2" max="2" step="0.1" value={exposure} onChange={e => setExposure(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Contrast</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{contrast}</span>
                </div>
                <input type="range" min="0.5" max="1.5" step="0.05" value={contrast} onChange={e => setContrast(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Highlights</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{highlights}</span>
                </div>
                <input type="range" min="-1" max="1" step="0.1" value={highlights} onChange={e => setHighlights(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Shadows</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{shadows}</span>
                </div>
                <input type="range" min="-1" max="1" step="0.1" value={shadows} onChange={e => setShadows(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Color</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <button
                onClick={() => setIsPickingWB(!isPickingWB)}
                disabled={Math.abs(rotation) > 0 || Math.abs(geoVertical) > 0 || Math.abs(geoHorizontal) > 0}
                title={Math.abs(rotation) > 0 || Math.abs(geoVertical) > 0 || Math.abs(geoHorizontal) > 0 ? "Reset Geometry/Rotation to pick WB" : "Click image to set White Balance"}
                style={{
                  background: isPickingWB ? '#444' : '#222',
                  color: '#eee',
                  border: '1px solid #444',
                  padding: '6px 12px',
                  borderRadius: '4px',
                  cursor: (Math.abs(rotation) > 0 || Math.abs(geoVertical) > 0 || Math.abs(geoHorizontal) > 0) ? 'not-allowed' : 'pointer',
                  fontSize: '0.8rem',
                  width: '100%',
                  opacity: (Math.abs(rotation) > 0 || Math.abs(geoVertical) > 0 || Math.abs(geoHorizontal) > 0) ? 0.5 : 1
                }}
              >
                {isPickingWB ? 'Cancel Picker' : 'Pick Neutral Gray'}
              </button>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Temp</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{temp.toFixed(2)}</span>
                </div>
                <input data-testid="temp-slider" type="range" min="-1" max="1" step="0.05" value={temp} onChange={e => setTemp(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Tint</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{tint.toFixed(2)}</span>
                </div>
                <input data-testid="tint-slider" type="range" min="-1" max="1" step="0.05" value={tint} onChange={e => setTint(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Split Toning</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 'bold', display: 'block', marginBottom: '0.5rem', color: '#ccc' }}>Highlights</label>
                <div style={{ marginBottom: '0.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                    <label style={{ fontSize: '0.85rem' }}>Hue</label>
                    <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{stHighlightHue}°</span>
                  </div>
                  <input type="range" min="0" max="360" step="1" value={stHighlightHue} onChange={e => setStHighlightHue(parseFloat(e.target.value))} style={{ width: '100%' }} />
                  <div style={{ height: '4px', background: `hsl(${stHighlightHue}, 100%, 50%)`, borderRadius: '2px', marginTop: '4px' }}></div>
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                    <label style={{ fontSize: '0.85rem' }}>Saturation</label>
                    <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{stHighlightSat}</span>
                  </div>
                  <input type="range" min="0" max="1" step="0.05" value={stHighlightSat} onChange={e => setStHighlightSat(parseFloat(e.target.value))} style={{ width: '100%' }} />
                </div>
              </div>

              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 'bold', display: 'block', marginBottom: '0.5rem', color: '#ccc' }}>Shadows</label>
                <div style={{ marginBottom: '0.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                    <label style={{ fontSize: '0.85rem' }}>Hue</label>
                    <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{stShadowHue}°</span>
                  </div>
                  <input type="range" min="0" max="360" step="1" value={stShadowHue} onChange={e => setStShadowHue(parseFloat(e.target.value))} style={{ width: '100%' }} />
                  <div style={{ height: '4px', background: `hsl(${stShadowHue}, 100%, 50%)`, borderRadius: '2px', marginTop: '4px' }}></div>
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                    <label style={{ fontSize: '0.85rem' }}>Saturation</label>
                    <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{stShadowSat}</span>
                  </div>
                  <input type="range" min="0" max="1" step="0.05" value={stShadowSat} onChange={e => setStShadowSat(parseFloat(e.target.value))} style={{ width: '100%' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Balance</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{stBalance}</span>
                </div>
                <input type="range" min="-1" max="1" step="0.1" value={stBalance} onChange={e => setStBalance(parseFloat(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', opacity: 0.5, marginTop: '2px' }}>
                  <span>Shadows</span>
                  <span>Highlights</span>
                </div>
              </div>
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>HSL Tuning</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {(['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta'] as const).map((col) => (
                <div key={col} style={{ background: '#1a1a1a', padding: '0.8rem', borderRadius: '4px' }}>
                  <div style={{ fontSize: '0.9rem', marginBottom: '0.6rem', color: '#fff', fontWeight: 'bold', textTransform: 'capitalize', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: col }} />
                    {col}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', opacity: 0.7 }}>
                        <label>Hue</label>
                        <span>{hsl[col].hue.toFixed(2)}</span>
                      </div>
                      <input
                        type="range" min="-1" max="1" step="0.1"
                        value={hsl[col].hue}
                        onChange={e => setHsl({ ...hsl, [col]: { ...hsl[col], hue: parseFloat(e.target.value) } })}
                        style={{ width: '100%' }}
                      />
                    </div>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', opacity: 0.7 }}>
                        <label>Sat</label>
                        <span>{hsl[col].saturation.toFixed(2)}</span>
                      </div>
                      <input
                        type="range" min="-1" max="1" step="0.1"
                        value={hsl[col].saturation}
                        onChange={e => setHsl({ ...hsl, [col]: { ...hsl[col], saturation: parseFloat(e.target.value) } })}
                        style={{ width: '100%' }}
                      />
                    </div>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', opacity: 0.7 }}>
                        <label>Lum</label>
                        <span>{hsl[col].luminance.toFixed(2)}</span>
                      </div>
                      <input
                        type="range" min="-1" max="1" step="0.1"
                        value={hsl[col].luminance}
                        onChange={e => setHsl({ ...hsl, [col]: { ...hsl[col], luminance: parseFloat(e.target.value) } })}
                        style={{ width: '100%' }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Details</h3>

            <div style={{ paddingBottom: '1rem', borderBottom: '1px solid #222', marginBottom: '1rem' }}>
              <label style={{ fontSize: '0.8rem', fontWeight: 'bold', display: 'block', marginBottom: '0.5rem', color: '#ccc' }}>Sharpen</label>

              <div style={{ marginBottom: '0.8rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Amount</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{sharpenAmount}</span>
                </div>
                <input type="range" min="0" max="5.0" step="0.1" value={sharpenAmount} onChange={e => setSharpenAmount(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div style={{ marginBottom: '0.8rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Radius</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{sharpenRadius}</span>
                </div>
                <input type="range" min="0.1" max="10.0" step="0.1" value={sharpenRadius} onChange={e => setSharpenRadius(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Threshold</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{sharpenThreshold}</span>
                </div>
                <input type="range" min="0" max="50" step="1" value={sharpenThreshold} onChange={e => setSharpenThreshold(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Clarity</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{clarityAmount}</span>
                </div>
                <input type="range" min="-1" max="1" step="0.05" value={clarityAmount} onChange={e => setClarityAmount(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Dehaze</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{dehazeAmount}</span>
                </div>
                <input type="range" min="0" max="1" step="0.05" value={dehazeAmount} onChange={e => setDehazeAmount(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Vignette</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Amount</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{vAmount}</span>
                </div>
                <input type="range" min="-1" max="1" step="0.05" value={vAmount} onChange={e => setVAmount(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Midpoint</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{vMidpoint}</span>
                </div>
                <input type="range" min="0" max="1" step="0.05" value={vMidpoint} onChange={e => setVMidpoint(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Roundness</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{vRoundness}</span>
                </div>
                <input type="range" min="-1" max="1" step="0.05" value={vRoundness} onChange={e => setVRoundness(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Feather</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{vFeather}</span>
                </div>
                <input type="range" min="0" max="1" step="0.05" value={vFeather} onChange={e => setVFeather(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Grain</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Amount</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{grainAmount}</span>
                </div>
                <input type="range" min="0" max="1" step="0.05" value={grainAmount} onChange={e => setGrainAmount(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <label style={{ fontSize: '0.85rem', display: 'block', marginBottom: '0.3rem' }}>Size</label>
                <select
                  value={grainSize}
                  onChange={e => setGrainSize(e.target.value as 'fine' | 'medium' | 'coarse')}
                  style={{ width: '100%', background: '#222', color: '#eee', border: '1px solid #444', padding: '4px', borderRadius: '4px' }}
                >
                  <option value="fine">Fine</option>
                  <option value="medium">Medium</option>
                  <option value="coarse">Coarse</option>
                </select>
              </div>
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Denoise</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Luminance</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{denoiseLuminance}</span>
                </div>
                <input type="range" min="0" max="1" step="0.05" value={denoiseLuminance} onChange={e => setDenoiseLuminance(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Color</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{denoiseColor}</span>
                </div>
                <input type="range" min="0" max="1" step="0.05" value={denoiseColor} onChange={e => setDenoiseColor(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>LUT</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <input type="file" accept=".cube,.3dl,.xmp,.xml" onChange={handleLutUpload} style={{ fontSize: '0.8rem' }} />
              {lutName && <span style={{ fontSize: '0.75rem', color: '#4caf50' }}>Loaded: {lutName}</span>}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Intensity</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{lutIntensity}</span>
                </div>
                <input type="range" min="0" max="1" step="0.05" value={lutIntensity} onChange={e => setLutIntensity(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Curves</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Curve Strength</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{curves.intensity}</span>
                </div>
                <input data-testid="curve-strength-slider" type="range" min="0" max="1" step="0.05" value={curves.intensity} onChange={e => setCurvesIntensity(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>
              <CurveEditor curves={curves} onChange={setCurves} />
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Geometry</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Rotation</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{rotation}°</span>
                </div>
                <input data-testid="rotation-slider" type="range" min="-45" max="45" step="1" value={rotation} onChange={e => setRotation(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                <input type="checkbox" checked={autoCrop} onChange={e => setAutoCrop(e.target.checked)} />
                Auto Crop (Straighten)
              </label>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Vertical Perspective</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{geoVertical}</span>
                </div>
                <input type="range" min="-1" max="1" step="0.05" value={geoVertical} onChange={e => setGeoVertical(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label style={{ fontSize: '0.85rem' }}>Horizontal Perspective</label>
                  <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>{geoHorizontal}</span>
                </div>
                <input type="range" min="-1" max="1" step="0.05" value={geoHorizontal} onChange={e => setGeoHorizontal(parseFloat(e.target.value))} style={{ width: '100%' }} />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                  <input type="checkbox" checked={flipHorizontal} onChange={e => setFlipHorizontal(e.target.checked)} />
                  Flip Horizontal
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                  <input type="checkbox" checked={flipVertical} onChange={e => setFlipVertical(e.target.checked)} />
                  Flip Vertical
                </label>
              </div>
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Crop</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button onClick={applyCrop} disabled={!!appliedCrop} style={{ flex: 1, padding: '6px', fontSize: '0.8rem' }}>Apply</button>
                <button onClick={resetCrop} disabled={!appliedCrop} style={{ flex: 1, padding: '6px', fontSize: '0.8rem' }}>Reset</button>
              </div>

              {!appliedCrop ? (
                <>
                  <p style={{ fontSize: '0.75rem', color: '#aaa', margin: 0 }}>Adjust red box on canvas.</p>
                  <div>
                    <label style={{ fontSize: '0.8rem' }}>X: {cropX.toFixed(2)}</label>
                    <input type="range" min="0" max="1" step="0.01" value={cropX} onChange={e => setCropX(parseFloat(e.target.value))} style={{ width: '100%' }} />
                  </div>
                  <div>
                    <label style={{ fontSize: '0.8rem' }}>Y: {cropY.toFixed(2)}</label>
                    <input type="range" min="0" max="1" step="0.01" value={cropY} onChange={e => setCropY(parseFloat(e.target.value))} style={{ width: '100%' }} />
                  </div>
                  <div>
                    <label style={{ fontSize: '0.8rem' }}>W: {cropW.toFixed(2)}</label>
                    <input type="range" min="0" max="1" step="0.01" value={cropW} onChange={e => setCropW(parseFloat(e.target.value))} style={{ width: '100%' }} />
                  </div>
                  <div>
                    <label style={{ fontSize: '0.8rem' }}>H: {cropH.toFixed(2)}</label>
                    <input type="range" min="0" max="1" step="0.01" value={cropH} onChange={e => setCropH(parseFloat(e.target.value))} style={{ width: '100%' }} />
                  </div>
                </>
              ) : (
                <div style={{ fontSize: '0.75rem', background: '#222', padding: '0.5rem', borderRadius: '4px' }}>
                  <p style={{ margin: '0 0 0.3rem 0' }}>Crop Applied.</p>
                  <pre style={{ margin: 0 }}>{JSON.stringify(appliedCrop, null, 2)}</pre>
                </div>
              )}
            </div>
          </section>

          <section>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Histogram</h3>
            <Histogram data={histogramData} />
          </section>

          <section style={{ opacity: 0.6, fontSize: '0.8rem', borderTop: '1px solid #333', paddingTop: '1rem' }}>
            <h4 style={{ margin: '0 0 0.5rem 0' }}>Debug Info</h4>
            <div>WB Mode: {isPickingWB ? 'ACTIVE' : 'Inactive'}</div>
            <div>Canvas: {canvasRef.current ? `${canvasRef.current.width}x${canvasRef.current.height}` : 'N/A'}</div>
            <div>Image: {image ? `${image.width}x${image.height}` : 'N/A'}</div>
          </section>
        </aside>
      </div>
    </div>
  );
}

export default App;
