import { RendererOptions, QuickFixAdjustments } from '../pkg/quickfix_renderer';

export type WorkerMessage =
    | { type: 'INIT'; payload: { rendererOptions: RendererOptions } }
    | { type: 'RENDER'; payload: { imageData: ImageBitmap | ArrayBuffer; width: number; height: number; adjustments: QuickFixAdjustments; requestId: number } }
    | { type: 'CANCEL'; payload: { requestId: number } }
    | { type: 'DISPOSE' };

export type WorkerResponse =
    | { type: 'INIT_RESULT'; payload: { success: true; backend: string } }
    | { type: 'FRAME_READY'; payload: { requestId: number; imageBitmap: ImageBitmap | ArrayBuffer; width: number; height: number; timing: number } }
    | { type: 'ERROR'; payload: { requestId?: number; error: string } };
