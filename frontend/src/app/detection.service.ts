import { Injectable, signal } from '@angular/core';

const WS_URL    = 'ws://localhost:8000/ws/predictions';
const VIDEO_URL = 'http://localhost:8000/video_feed';

@Injectable({ providedIn: 'root' })
export class DetectionService {
  readonly letter      = signal<string | null>(null);
  readonly confidence  = signal(0);
  readonly connected   = signal(false);
  readonly videoUrl    = VIDEO_URL;

  private ws: WebSocket | null = null;

  start(): void {
    if (this.ws) return;

    this.ws = new WebSocket(WS_URL);

    this.ws.onopen = () => this.connected.set(true);

    this.ws.onmessage = ({ data }) => {
      const { letter, confidence } = JSON.parse(data);
      this.letter.set(letter ?? null);
      this.confidence.set(confidence ?? 0);
    };

    this.ws.onclose = () => {
      this.connected.set(false);
      this.ws = null;
    };

    this.ws.onerror = () => this.ws?.close();
  }

  stop(): void {
    this.ws?.close();
    this.ws = null;
    this.letter.set(null);
    this.confidence.set(0);
    this.connected.set(false);
  }
}
