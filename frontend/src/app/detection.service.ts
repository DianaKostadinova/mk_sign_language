import { Injectable, signal } from '@angular/core';

const WS_URL    = 'ws://localhost:8000/ws/predictions';
const VIDEO_URL = 'http://localhost:8000/video_feed';

@Injectable({ providedIn: 'root' })
export class DetectionService {
  readonly letter     = signal<string | null>(null);
  readonly confidence = signal(0);
  readonly connected  = signal(false);
  readonly videoUrl   = VIDEO_URL;

  private ws:      WebSocket | null = null;
  private retryId: ReturnType<typeof setTimeout> | null = null;

  start(): void {
    this._connect();
  }

  stop(): void {
    if (this.retryId) { clearTimeout(this.retryId); this.retryId = null; }
    this.ws?.close();
    this.ws = null;
    this.letter.set(null);
    this.confidence.set(0);
    this.connected.set(false);
  }

  private _connect(): void {
    this.ws = new WebSocket(WS_URL);

    this.ws.onopen = () => {
      console.log('[DetectionService] WS connected');
      this.connected.set(true);
    };

    this.ws.onmessage = ({ data }) => {
      const { letter, confidence } = JSON.parse(data) as {
        letter: string | null;
        confidence: number;
      };
      this.letter.set(letter ?? null);
      this.confidence.set(confidence ?? 0);
    };

    this.ws.onclose = () => {
      console.warn('[DetectionService] WS closed — retrying in 2s');
      this.connected.set(false);
      this.ws = null;
      // retry while the user hasn't explicitly stopped
      this.retryId = setTimeout(() => this._connect(), 2000);
    };

    this.ws.onerror = (e) => {
      console.error('[DetectionService] WS error', e);
      this.ws?.close();
    };
  }
}
