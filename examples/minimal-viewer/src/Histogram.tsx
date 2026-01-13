import React, { useEffect, useRef } from 'react';

interface HistogramProps {
    data: number[]; // Flat array [R0..R255, G0..G255, B0..B255]
    width?: number;
    height?: number;
}

export function Histogram({ data, width = 256, height = 100 }: HistogramProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !data || data.length !== 256 * 3) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, width, height);

        // Find max value for normalization
        let max = 0;
        for (let i = 0; i < data.length; i++) {
            if (data[i] > max) max = data[i];
        }
        // Logarithmic scale often looks better for histograms, or just linear. 
        // Linear is requested effectively by "judge exposure and clipping objectively". 
        // Usually linear is better for clipping check.

        // Allow some headroom? clipping checks usually need to see if bin 0 or 255 is high.

        // Draw channels
        ctx.globalCompositeOperation = 'screen'; // Additive blending for RGB

        const drawChannel = (offset: number, color: string) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const count = data[offset + i];
                // Normalize height
                const h = (count / max) * height;
                // Draw bar -> 1px width
                const x = (i / 255) * width;
                const y = height - h;
                ctx.fillRect(x, y, width / 256, h);
            }
        };

        drawChannel(0, 'red');     // R
        drawChannel(256, 'green'); // G
        drawChannel(512, 'blue');  // B

        // Reset composite
        ctx.globalCompositeOperation = 'source-over';

    }, [data, width, height]);

    return (
        <canvas
            ref={canvasRef}
            width={width}
            height={height}
            style={{ border: '1px solid #444', background: '#000' }}
        />
    );
}
