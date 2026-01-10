import React, { useState, useMemo, useRef } from 'react';

export interface CurvePoint {
    x: number;
    y: number;
}

export interface ChannelCurve {
    points: CurvePoint[];
}

export interface CurvesSettings {
    intensity?: number;
    master?: ChannelCurve;
    red?: ChannelCurve;
    green?: ChannelCurve;
    blue?: ChannelCurve;
}

interface CurveEditorProps {
    curves: CurvesSettings;
    onChange: (curves: CurvesSettings) => void;
}

type ChannelType = 'master' | 'red' | 'green' | 'blue';

export function CurveEditor({ curves, onChange }: CurveEditorProps) {
    const [activeChannel, setActiveChannel] = useState<ChannelType>('master');
    const svgRef = useRef<SVGSVGElement>(null);
    const [draggingIdx, setDraggingIdx] = useState<number | null>(null);

    const activePoints = useMemo(() => {
        const curve = curves[activeChannel];
        if (!curve || !curve.points || curve.points.length === 0) {
            return [{ x: 0, y: 0 }, { x: 1, y: 1 }];
        }
        return [...curve.points].sort((a, b) => a.x - b.x);
    }, [curves, activeChannel]);

    // Simple Spline Implementation for UI
    const splinePoints = useMemo(() => {
        if (activePoints.length < 2) return [];

        const n = activePoints.length;
        const h = new Array(n - 1);
        for (let i = 0; i < n - 1; i++) {
            h[i] = activePoints[i + 1].x - activePoints[i].x;
            if (h[i] === 0) h[i] = 1e-6;
        }

        const alpha = new Array(n - 1).fill(0);
        for (let i = 1; i < n - 1; i++) {
            alpha[i] = (3 / h[i]) * (activePoints[i + 1].y - activePoints[i].y) - (3 / h[i - 1]) * (activePoints[i].y - activePoints[i - 1].y);
        }

        const l = new Array(n).fill(1);
        const mu = new Array(n).fill(0);
        const z = new Array(n).fill(0);

        for (let i = 1; i < n - 1; i++) {
            l[i] = 2 * (activePoints[i + 1].x - activePoints[i - 1].x) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        const b = new Array(n).fill(0);
        const c = new Array(n).fill(0);
        const d = new Array(n).fill(0);

        for (let j = n - 2; j >= 0; j--) {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (activePoints[j + 1].y - activePoints[j].y) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
            d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
        }

        const result = [];
        const steps = 100;
        for (let i = 0; i <= steps; i++) {
            const x = i / steps;
            let idx = n - 2;
            for (let j = 0; j < n - 1; j++) {
                if (x <= activePoints[j + 1].x) {
                    idx = j;
                    break;
                }
            }
            const dx = x - activePoints[idx].x;
            const y = activePoints[idx].y + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx;
            result.push({ x, y: Math.max(0, Math.min(1, y)) });
        }
        return result;
    }, [activePoints]);

    const normalizeCoords = (e: React.MouseEvent | React.TouchEvent) => {
        if (!svgRef.current) return { x: 0, y: 0 };
        const rect = svgRef.current.getBoundingClientRect();
        let clientX, clientY;
        if ('touches' in e) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = (e as React.MouseEvent).clientX;
            clientY = (e as React.MouseEvent).clientY;
        }
        return {
            x: Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)),
            y: Math.max(0, Math.min(1, 1 - (clientY - rect.top) / rect.height))
        };
    };

    const updatePoints = (newPoints: CurvePoint[]) => {
        const sorted = [...newPoints].sort((a, b) => a.x - b.x);
        onChange({
            ...curves,
            [activeChannel]: { points: sorted }
        });
    };

    const handleMouseDown = (e: React.MouseEvent, idx: number) => {
        e.stopPropagation();
        setDraggingIdx(idx);
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (draggingIdx === null) return;
        const { x, y } = normalizeCoords(e);
        const newPoints = [...activePoints];

        // Constrain X move to be between neighbors
        let finalX = x;
        if (draggingIdx === 0) finalX = 0;
        else if (draggingIdx === activePoints.length - 1) finalX = 1;
        else {
            const minX = activePoints[draggingIdx - 1].x + 0.01;
            const maxX = activePoints[draggingIdx + 1].x - 0.01;
            finalX = Math.max(minX, Math.min(maxX, x));
        }

        newPoints[draggingIdx] = { x: finalX, y };
        updatePoints(newPoints);
    };

    const handleMouseUp = () => {
        setDraggingIdx(null);
    };

    const handleSvgClick = (e: React.MouseEvent) => {
        if (draggingIdx !== null) return;
        const { x, y } = normalizeCoords(e);

        // Find if we clicked on a point (already handled by handleMouseDown but just in case)
        // Add new point if not clicking exactly on one
        const newPoints = [...activePoints, { x, y }];
        updatePoints(newPoints);
    };

    const deletePoint = (e: React.MouseEvent, idx: number) => {
        e.preventDefault();
        e.stopPropagation();
        if (idx === 0 || idx === activePoints.length - 1) return; // Can't delete ends
        const newPoints = activePoints.filter((_, i) => i !== idx);
        updatePoints(newPoints);
    };

    const pathData = useMemo(() => {
        if (splinePoints.length === 0) return "";
        return "M " + splinePoints.map(p => `${p.x * 256} ${(1 - p.y) * 256}`).join(" L ");
    }, [splinePoints]);

    return (
        <div style={{ background: '#222', padding: '10px', borderRadius: '8px', border: '1px solid #444', color: '#eee' }}>
            <div style={{ display: 'flex', gap: '5px', marginBottom: '10px' }}>
                {(['master', 'red', 'green', 'blue'] as const).map(ch => (
                    <button
                        key={ch}
                        onClick={() => setActiveChannel(ch)}
                        style={{
                            flex: 1,
                            background: activeChannel === ch ? (ch === 'master' ? '#666' : ch) : '#333',
                            color: activeChannel === ch ? '#fff' : '#aaa',
                            border: 'none',
                            padding: '5px',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '0.8rem',
                            textTransform: 'capitalize'
                        }}
                    >
                        {ch}
                    </button>
                ))}
            </div>

            <div style={{ position: 'relative', width: '256px', height: '256px', background: '#000', margin: '0 auto' }}>
                <svg
                    ref={svgRef}
                    width="256"
                    height="256"
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    onClick={handleSvgClick}
                    style={{ cursor: 'crosshair', display: 'block' }}
                >
                    {/* Grid */}
                    <line x1="0" y1="64" x2="256" y2="64" stroke="#333" strokeWidth="1" />
                    <line x1="0" y1="128" x2="256" y2="128" stroke="#333" strokeWidth="1" />
                    <line x1="0" y1="192" x2="256" y2="192" stroke="#333" strokeWidth="1" />
                    <line x1="64" y1="0" x2="64" y2="256" stroke="#333" strokeWidth="1" />
                    <line x1="128" y1="0" x2="128" y2="256" stroke="#333" strokeWidth="1" />
                    <line x1="192" y1="0" x2="192" y2="256" stroke="#333" strokeWidth="1" />

                    {/* Diagonal Identity Line */}
                    <line x1="0" y1="256" x2="256" y2="0" stroke="#444" strokeWidth="1" strokeDasharray="4" />

                    {/* Curve */}
                    <path
                        d={pathData}
                        fill="none"
                        stroke={activeChannel === 'master' ? '#fff' : activeChannel}
                        strokeWidth="2"
                    />

                    {/* Points */}
                    {activePoints.map((p, i) => (
                        <circle
                            key={i}
                            cx={p.x * 256}
                            cy={(1 - p.y) * 256}
                            r="5"
                            fill={draggingIdx === i ? '#fff' : (activeChannel === 'master' ? '#ccc' : activeChannel)}
                            stroke="#fff"
                            strokeWidth="1"
                            onMouseDown={(e) => handleMouseDown(e, i)}
                            onContextMenu={(e) => deletePoint(e, i)}
                            style={{ cursor: 'move' }}
                        />
                    ))}
                </svg>
            </div>

            <div style={{ marginTop: '10px', fontSize: '0.7rem', color: '#888', textAlign: 'center' }}>
                Click to add point. Drag to move. Right-click to delete.
            </div>

            <button
                onClick={() => updatePoints([{ x: 0, y: 0 }, { x: 1, y: 1 }])}
                style={{
                    width: '100%',
                    marginTop: '10px',
                    background: '#444',
                    color: '#eee',
                    border: 'none',
                    padding: '5px',
                    borderRadius: '4px',
                    cursor: 'pointer'
                }}
            >
                Reset Channel
            </button>
        </div>
    );
}
