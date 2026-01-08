import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import App from './App';

// Mock the CSS
vi.mock('./index.css', () => ({}));
vi.mock('./App.css', () => ({}));

// Mock the WASM module urls
vi.mock('../../../quickfix-renderer/pkg/worker.js?url', () => ({
    default: 'mock-worker-url'
}));

// Mock the Client
vi.mock('../../../quickfix-renderer/pkg/client.js', () => {
    return {
        QuickFixClient: class {
            constructor() { }
            async init() { }
            async setImage() { }
            async render() {
                return {
                    imageBitmap: new ArrayBuffer(0),
                    width: 100,
                    height: 100,
                    histogram: []
                };
            }
            async uploadLut() { }
            dispose() { }
        }
    };
});

// Mock dependent package
vi.mock('quickfix-renderer', () => ({
    RendererOptions: {}
}));

describe('App Integration', () => {
    let checkPixelData: Uint8ClampedArray;

    beforeEach(() => {
        // Mock Canvas Context
        // Create 100x100 buffer filled with (200, 150, 150, 255)
        const size = 100 * 100 * 4;
        checkPixelData = new Uint8ClampedArray(size);
        for (let i = 0; i < size; i += 4) {
            checkPixelData[i] = 200;
            checkPixelData[i + 1] = 150;
            checkPixelData[i + 2] = 150;
            checkPixelData[i + 3] = 255;
        }

        const mockContext = {
            drawImage: vi.fn(),
            getImageData: vi.fn(() => ({
                data: checkPixelData,
                width: 100,
                height: 100
            })),
            putImageData: vi.fn(),
            strokeRect: vi.fn(),
            fillRect: vi.fn(),
            beginPath: vi.fn(),
            moveTo: vi.fn(),
            lineTo: vi.fn(),
            stroke: vi.fn(),
        } as unknown as CanvasRenderingContext2D;

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(((contextId: string) => {
            if (contextId === '2d') return mockContext;
            return null;
        }) as any);

        // Mock Image
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (globalThis as any).Image = class {
            width = 100;
            height = 100;
            src = '';
            onload: (() => void) | null = null;
            constructor() {
                setTimeout(() => {
                    if (this.onload) this.onload();
                }, 10);
            }
        };

        // Mock ImageData
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (globalThis as any).ImageData = class {
            constructor(public data: Uint8ClampedArray, public width: number, public height: number) { }
        };

        // Mock createImageBitmap
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (globalThis as any).createImageBitmap = async (_image: any) => {
            return {
                close: () => { },
                width: 100,
                height: 100
            };
        };
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('renders and allows picking white balance', async () => {
        const { container } = render(<App />);

        // 1. Wait for canvas to be in document.
        await waitFor(() => {
            // Check for canvas in container
            expect(container.querySelector('canvas')).toBeInTheDocument();
            // Check if "Backend:" text is present, indicating app render
            const backendElements = screen.getAllByText(/Backend/i);
            expect(backendElements.length).toBeGreaterThan(0);

            // Wait for Image Size to be updated (implies setImageData has run)
            expect(screen.getByText(/Image Size: 100x100/i)).toBeInTheDocument();
        });

        const canvas = container.querySelector('canvas');
        expect(canvas).toBeInTheDocument();

        // 2. Find "Pick Neutral Gray" button
        const pickButton = screen.getByText(/Pick Neutral Gray/i);
        expect(pickButton).toBeInTheDocument();

        // 3. Click the button to enable picker mode
        fireEvent.click(pickButton);
        expect(pickButton).toHaveTextContent(/Cancel Picker/i);

        // 4. Verify canvas cursor changed
        expect(canvas).toHaveStyle({ cursor: 'crosshair' });

        // 5. Mock getBoundingClientRect for click coordinates
        vi.spyOn(canvas!, 'getBoundingClientRect').mockReturnValue({
            left: 0,
            top: 0,
            width: 100,
            height: 100,
            x: 0,
            y: 0,
            bottom: 100,
            right: 100,
            toJSON: () => { }
        });

        // 6. Click on the canvas (client coordinates)
        console.log("Simulating click at 50,50");
        fireEvent.click(canvas!, { clientX: 50, clientY: 50 });

        // 7. Verify sliders updated
        const tempInput = screen.getByTestId('temp-slider') as HTMLInputElement;
        const tintInput = screen.getByTestId('tint-slider') as HTMLInputElement;

        await waitFor(() => {
            const tempVal = parseFloat(tempInput.value);
            const tintVal = parseFloat(tintInput.value);
            console.log(`Temp: ${tempVal}, Tint: ${tintVal}`);

            // Expected: Temp ~ -0.57, Tint ~ -0.45
            expect(tempVal).toBeCloseTo(-0.57, 1);
            expect(tintVal).toBeCloseTo(-0.45, 1);
        });

        // 8. Verify mode reset
        expect(pickButton).toHaveTextContent(/Pick Neutral Gray/i);
        expect(canvas).toHaveStyle({ cursor: 'default' });
    });
});
