import { WorkerMessage, WorkerResponse } from './protocol';
import { RendererOptions, QuickFixAdjustments } from '../pkg/quickfix_renderer';

/**
 * Client-side wrapper for the Quick Fix Web Worker.
 * Handles worker lifecycle, message passing, and request/response correlation.
 */
export class QuickFixClient {
    private worker: Worker;
    private nextRequestId = 1;
    private pendingRequests = new Map<number, { resolve: (value: any) => void, reject: (reason: any) => void }>();

    private initResolver: ((value: void | PromiseLike<void>) => void) | null = null;
    private initRejecter: ((reason?: any) => void) | null = null;

    /**
     * Creates a new QuickFixClient.
     * @param workerOrUrl - The Worker instance or URL to the worker script.
     */
    constructor(workerOrUrl: string | URL | Worker) {
        if (workerOrUrl instanceof Worker) {
            this.worker = workerOrUrl;
        } else {
            this.worker = new Worker(workerOrUrl, { type: 'module' });
        }
        this.worker.onmessage = this.handleMessage.bind(this);
    }

    private handleMessage(event: MessageEvent<WorkerResponse>) {
        const msg = event.data;
        switch (msg.type) {
            case 'INIT_RESULT':
                if (this.initResolver) {
                    this.initResolver();
                    this.initResolver = null;
                    this.initRejecter = null;
                }
                break;
            case 'FRAME_READY':
                const { requestId, imageBitmap, width, height, timing } = msg.payload;
                const resolver = this.pendingRequests.get(requestId);
                if (resolver) {
                    resolver.resolve({ imageBitmap, width, height, timing });
                    this.pendingRequests.delete(requestId);
                }
                break;
            case 'FINAL_RENDER_READY':
                const { requestId: fReqId, data, width: fW, height: fH } = msg.payload;
                const fResolver = this.pendingRequests.get(fReqId);
                if (fResolver) {
                    fResolver.resolve({ data: new Uint8Array(data), width: fW, height: fH });
                    this.pendingRequests.delete(fReqId);
                }
                break;
            case 'ERROR':
                const { requestId: errReqId, error } = msg.payload;
                if (errReqId && this.pendingRequests.has(errReqId)) {
                    this.pendingRequests.get(errReqId)!.reject(new Error(error));
                    this.pendingRequests.delete(errReqId);
                } else if (this.initRejecter) {
                    this.initRejecter(new Error(error));
                    this.initResolver = null;
                    this.initRejecter = null;
                } else {
                    console.error('QuickFixClient Worker Error:', error);
                }
                break;
        }
    }

    /**
     * Initializes the WASM backend in the worker.
     * @param options - Configuration options for the renderer.
     */
    async init(options: RendererOptions): Promise<void> {
        return new Promise((resolve, reject) => {
            this.initResolver = resolve;
            this.initRejecter = reject;
            this.worker.postMessage({
                type: 'INIT',
                payload: { rendererOptions: options }
            } as WorkerMessage);
        });
    }

    /**
     * Sets the source image for subsequent render calls.
     * This allows for stateful rendering without re-transferring the image data.
     * @param imageData - The input image data.
     * @param width - Image width.
     * @param height - Image height.
     */
    async setImage(imageData: ImageBitmap | ArrayBuffer, width: number, height: number): Promise<void> {
        // Transfer buffer if it's an ArrayBuffer
        const transfer: Transferable[] = [];
        if (imageData instanceof ArrayBuffer) {
            transfer.push(imageData);
        }
        if (imageData instanceof ImageBitmap) {
            transfer.push(imageData);
        }

        this.worker.postMessage({
            type: 'SET_IMAGE',
            payload: { imageData, width, height }
        } as WorkerMessage, transfer);
    }

    /**
     * Uploads a raw .cube file content to the worker for parsing and application.
     * @param content - The text content of the .cube file.
     */
    async uploadLut(content: string, extension: string): Promise<void> {
        this.worker.postMessage({
            type: 'UPLOAD_LUT',
            payload: { content, extension }
        } as WorkerMessage);
    }

    /**
     * Sends a render request to the worker.
     * @param imageData - The input image data (optional if setImage was called).
     * @param width - Image width.
     * @param height - Image height.
     * @param adjustments - The adjustments to apply.
     * @returns A promise resolving to the processed image data.
     */
    render(
        imageData: ImageBitmap | ArrayBuffer | null,
        width: number,
        height: number,
        adjustments: QuickFixAdjustments,
        sourceId?: string
    ): Promise<{ imageBitmap: ImageBitmap | ArrayBuffer, width: number, height: number, timing: number }> {
        const requestId = this.nextRequestId++;

        return new Promise((resolve, reject) => {
            this.pendingRequests.set(requestId, { resolve, reject });

            // Transfer buffer if it's an ArrayBuffer
            const transfer: Transferable[] = [];
            if (imageData instanceof ArrayBuffer) {
                transfer.push(imageData);
            }
            // If ImageBitmap, it's also transferable
            if (imageData instanceof ImageBitmap) {
                transfer.push(imageData);
            }

            this.worker.postMessage({
                type: 'RENDER',
                payload: { requestId, imageData: imageData || undefined, width, height, adjustments, sourceId }
            } as WorkerMessage, transfer);
        });
    }

    /**
     * Sends a final high-res render request to the worker using tiling.
     * @param imageData - The input image data.
     * @param width - Image width.
     * @param height - Image height.
     * @param adjustments - The adjustments to apply.
     * @returns A promise resolving to the raw pixel data.
     */
    finalRender(
        imageData: ImageBitmap | ArrayBuffer | null,
        width: number,
        height: number,
        adjustments: QuickFixAdjustments
    ): Promise<{ data: Uint8Array, width: number, height: number }> {
        const requestId = this.nextRequestId++;

        return new Promise((resolve, reject) => {
            this.pendingRequests.set(requestId, { resolve, reject });

            const transfer: Transferable[] = [];
            if (imageData instanceof ArrayBuffer) {
                transfer.push(imageData);
            }
            if (imageData instanceof ImageBitmap) {
                transfer.push(imageData);
            }

            this.worker.postMessage({
                type: 'FINAL_RENDER',
                payload: { requestId, imageData: imageData || undefined, width, height, adjustments }
            } as WorkerMessage, transfer);
        });
    }

    /**
     * Terminates the worker and cleans up resources.
     */
    dispose() {
        this.worker.postMessage({ type: 'DISPOSE' } as WorkerMessage);
        this.worker.terminate();
        this.pendingRequests.forEach(p => p.reject(new Error("Client disposed")));
        this.pendingRequests.clear();
    }
}
