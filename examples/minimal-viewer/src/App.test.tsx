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
            async render(
                _imageData: ImageBitmap | ArrayBuffer | null,
                _width: number,
                _height: number,
                _adjustments: any,
                _sourceId?: string
            ) {
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

        vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(((contextId: string) => {
            if (contextId === '2d') return mockContext;
            return null;
        }) as unknown as (contextId: string) => CanvasRenderingContext2D | null);

        // Mock Image
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
        (globalThis as any).ImageData = class {
            data: Uint8ClampedArray;
            width: number;
            height: number;
            constructor(data: Uint8ClampedArray, width: number, height: number) {
                this.data = data;
                this.width = width;
                this.height = height;
            }
        };

        // Mock createImageBitmap
        (globalThis as any).createImageBitmap = async () => {
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

    it('passes sourceId to client render', async () => {
        // We need to spy on the QuickFixClient.prototype.render method
        // But we mocked the module return.
        // Let's rely on importing it and spying.

        // Wait, vi.mock happens before imports.
        // So import { QuickFixClient } from ... will get the Mocked Class.
        const { QuickFixClient } = await import('../../../quickfix-renderer/pkg/client.js');
        const spy = vi.spyOn(QuickFixClient.prototype, 'render');

        render(<App />);

        await waitFor(() => {
            expect(screen.getByText(/Image Size: 100x100/i)).toBeInTheDocument();
        });

        // The render should have been called
        await waitFor(() => {
            expect(spy).toHaveBeenCalled();
            // Check arguments. 
            // args: imageData, width, height, settings, sourceId
            const lastCall = spy.mock.lastCall as unknown as any[];
            const sourceId = lastCall[4];
            expect(typeof sourceId).toBe('string');
            expect(sourceId!.length).toBeGreaterThan(0);
        });
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

            // Expected: Temp ~ -0.57, Tint ~ -0.45
            expect(tempVal).toBeCloseTo(-0.57, 1);
            expect(tintVal).toBeCloseTo(-0.45, 1);
        });

        // 8. Verify mode reset
        expect(pickButton).toHaveTextContent(/Pick Neutral Gray/i);
        expect(canvas).toHaveStyle({ cursor: 'default' });
    });

    it('cancels picker when rotation changes', async () => {
        render(<App />);

        // Wait for load
        await waitFor(() => {
            const backendElements = screen.getAllByText(/Backend/i);
            expect(backendElements.length).toBeGreaterThan(0);
        });

        const pickButton = screen.getByText(/Pick Neutral Gray/i);

        // Enable picker
        fireEvent.click(pickButton);
        expect(pickButton).toHaveTextContent(/Cancel Picker/i);

        // Find Rotation slider
        const rotationSlider = screen.getByTestId('rotation-slider');

        // Change rotation
        fireEvent.change(rotationSlider, { target: { value: '5' } });

        // Verify picker is cancelled (text changes back)
        await waitFor(() => {
            expect(pickButton).toHaveTextContent(/Pick Neutral Gray/i);
        });

        // Also verify the button is now disabled
        expect(pickButton).toBeDisabled();
    });

    it('renders with a scrollable sidebar and curve strength slider', async () => {
        render(<App />);

        // Wait for load
        await waitFor(() => {
            const backendElements = screen.getAllByText(/Backend/i);
            expect(backendElements.length).toBeGreaterThan(0);
        });

        // 1. Verify sidebar exists and has correct styles for scrolling
        const sidebar = screen.getByRole('complementary');
        expect(sidebar).toBeInTheDocument();
        // The sidebar should have overflowY: auto
        expect(sidebar).toHaveStyle({ overflowY: 'auto' });

        // 2. Verify "Curve Strength" slider exists
        const curveStrengthSlider = screen.getByTestId('curve-strength-slider');
        expect(curveStrengthSlider).toBeInTheDocument();
        expect(curveStrengthSlider).toHaveValue('1');

        // 3. Verify it's labeled correctly
        expect(screen.getByText(/Curve Strength/i)).toBeInTheDocument();
    });
});
