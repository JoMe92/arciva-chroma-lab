import { RendererOptions, QuickFixAdjustments } from '../pkg/quickfix_renderer';

/**
 * Messages sent from the Main Thread (Client) to the Worker.
 */
export type WorkerMessage =
    | { type: 'INIT'; payload: { rendererOptions: RendererOptions } }
    | { type: 'SET_IMAGE'; payload: { imageData: ImageBitmap | ArrayBuffer; width: number; height: number } }
    | { type: 'RENDER'; payload: { imageData?: ImageBitmap | ArrayBuffer; width: number; height: number; adjustments: QuickFixAdjustments; requestId: number } }
    | { type: 'CANCEL'; payload: { requestId: number } }
    | { type: 'DISPOSE' }
    | { type: 'UPLOAD_LUT'; payload: { content: string; extension: string } };

/**
 * Messages sent from the Worker back to the Main Thread (Client).
 */
export type WorkerResponse =
    | { type: 'INIT_RESULT'; payload: { success: true; backend: string } }
    | { type: 'FRAME_READY'; payload: { requestId: number; imageBitmap: ImageBitmap | ArrayBuffer; width: number; height: number; timing: number } }
    | { type: 'ERROR'; payload: { requestId?: number; error: string } };
