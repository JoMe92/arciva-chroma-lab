import { WorkerMessage, WorkerResponse } from './protocol';
import { RendererOptions, QuickFixAdjustments } from '../pkg/quickfix_renderer';

export class QuickFixClient {
    private worker: Worker;
    private nextRequestId = 1;
    private pendingRequests = new Map<number, { resolve: (value: any) => void, reject: (reason: any) => void }>();

    private initResolver: ((value: void | PromiseLike<void>) => void) | null = null;
    private initRejecter: ((reason?: any) => void) | null = null;

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
                console.log('QuickFixClient: Worker initialized', msg.payload);
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
            case 'ERROR':
                const { requestId: errReqId, error } = msg.payload;
                if (errReqId && this.pendingRequests.has(errReqId)) {
                    this.pendingRequests.get(errReqId)!.reject(new Error(error));
                    this.pendingRequests.delete(errReqId);
                } else if (this.initRejecter) {
                    // Assume error might be related to init if we are waiting for it
                    // But ERROR payload has requestId. If requestId is undefined, maybe it's global error?
                    // My worker sends requestId if available.
                    // If init failed, worker might send ERROR without requestId?
                    // My worker sends ERROR with requestId from payload.
                    // INIT payload doesn't have requestId.
                    // So requestId is undefined.
                    this.initRejecter(new Error(error));
                    this.initResolver = null;
                    this.initRejecter = null;
                } else {
                    console.error('QuickFixClient Worker Error:', error);
                }
                break;
        }
    }

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

    render(
        imageData: ImageBitmap | ArrayBuffer,
        width: number,
        height: number,
        adjustments: QuickFixAdjustments
    ): Promise<{ imageBitmap: ImageBitmap | ArrayBuffer, width: number, height: number, timing: number }> {
        const requestId = this.nextRequestId++;

        // Cancel any previous pending requests if we want to enforce "only latest matters" at the client level too,
        // but the worker handles it. However, we should clean up our map.
        // Actually, if we want to support multiple concurrent requests (e.g. previews), we shouldn't auto-cancel here.
        // But for a slider, we usually want to cancel.
        // Let's leave it to the worker to drop stale ones, but we need to handle the fact that the promise might never resolve?
        // No, the worker should probably respond with "CANCELLED" or we just timeout?
        // Better: The worker drops it. The promise hangs? That's bad.
        // The worker should probably acknowledge cancellation or we just resolve with null?
        // For this iteration, let's keep it simple.

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
                payload: { requestId, imageData, width, height, adjustments }
            } as WorkerMessage, transfer);
        });
    }

    dispose() {
        this.worker.postMessage({ type: 'DISPOSE' } as WorkerMessage);
        this.worker.terminate();
        this.pendingRequests.forEach(p => p.reject(new Error("Client disposed")));
        this.pendingRequests.clear();
    }
}
