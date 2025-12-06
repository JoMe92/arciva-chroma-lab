import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { QuickFixClient } from './client';
import { RendererOptions, QuickFixAdjustments } from '../pkg/quickfix_renderer';

// Mock the Worker API globally
class MockWorker implements Partial<Worker> {
    onmessage: ((this: Worker, ev: MessageEvent) => any) | null = null;
    postMessage(message: any, transferOrOptions?: any) {
        // Echo back messages or handle specific logical mocks
        this.handleMessage(message);
    }
    terminate() { }
    addEventListener(type: string, listener: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions): void { }
    removeEventListener(type: string, listener: EventListenerOrEventListenerObject, options?: boolean | EventListenerOptions): void { }
    dispatchEvent(event: Event): boolean { return true; }
    onerror: ((this: AbstractWorker, ev: ErrorEvent) => any) | null = null;
    onmessageerror: ((this: Worker, ev: MessageEvent) => any) | null = null;

    // Helper to simulate messages coming back from the worker
    emitMessage(data: any) {
        if (this.onmessage) {
            this.onmessage({ data } as MessageEvent);
        }
    }

    // Logic to simulate backend responses
    handleMessage(msg: any) {
        switch (msg.type) {
            case 'INIT':
                setTimeout(() => {
                    this.emitMessage({
                        type: 'INIT_RESULT',
                        payload: { success: true, backend: 'cpu-mock' }
                    });
                }, 10);
                break;
            case 'RENDER':
                const { requestId, width, height } = msg.payload;
                setTimeout(() => {
                    // Return a dummy buffer
                    const buffer = new Uint8Array(width * height * 4).buffer;
                    this.emitMessage({
                        type: 'FRAME_READY',
                        payload: {
                            requestId,
                            imageBitmap: buffer,
                            width,
                            height,
                            timing: 50
                        }
                    });
                }, 20);
                break;
            case 'SET_IMAGE':
                // No response needed for SET_IMAGE in protocol, or maybe just silent acknowledgment
                break;
            case 'DISPOSE':
                break;
        }
    }
}

// @ts-ignore
global.Worker = MockWorker;
// @ts-ignore
global.ImageBitmap = class ImageBitmap { };

// Mock the WASM module imports since they don't exist in unit test environment
vi.mock('../pkg/quickfix_renderer', () => {
    return {
        RendererOptions: class {
            preferred_backend?: string;
            constructor(backend?: string) {
                this.preferred_backend = backend;
            }
        },
        // QuickFixAdjustments is an interface, so no runtime mock needed
    };
});

describe('QuickFixClient', () => {
    let client: QuickFixClient;

    beforeEach(() => {
        client = new QuickFixClient('worker.js');
    });

    afterEach(() => {
        client.dispose();
    });

    it('initializes successfully', async () => {
        await expect(client.init(new RendererOptions())).resolves.toBeUndefined();
    });

    it('processes a render request', async () => {
        await client.init(new RendererOptions());
        const adjustments: QuickFixAdjustments = {};
        const result = await client.render(new ArrayBuffer(100), 10, 10, adjustments);

        expect(result).toBeDefined();
        expect(result.width).toBe(10);
        expect(result.height).toBe(10);
        expect(result.imageBitmap).toBeInstanceOf(ArrayBuffer);
    });

    it('handles multiple concurrent requests correctly', async () => {
        await client.init(new RendererOptions());
        const adjustments: QuickFixAdjustments = {};

        const p1 = client.render(null, 10, 10, adjustments);
        const p2 = client.render(null, 20, 20, adjustments);

        const [r1, r2] = await Promise.all([p1, p2]);

        expect(r1.width).toBe(10);
        expect(r2.width).toBe(20);
    });
});
