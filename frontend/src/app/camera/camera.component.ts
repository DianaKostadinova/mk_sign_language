import  { Component, computed, inject, OnDestroy } from '@angular/core';
import { DetectionService } from '../detection.service';

@Component({
  selector: 'app-camera',
  standalone: true,
  templateUrl: './camera.component.html',
  styleUrls: ['./camera.component.scss']
})
export class CameraComponent implements OnDestroy {
  readonly detection = inject(DetectionService);

  readonly confidenceLabel = computed(() => {
    if (!this.detection.connected()) return 'no signal';
    if (this.detection.confidence() === 0) return 'waiting…';
    return `${Math.round(this.detection.confidence() * 100)}% confidence`;
  });

  toggle(): void {
    this.detection.connected() ? this.detection.stop() : this.detection.start();
  }

  ngOnDestroy(): void {
    this.detection.stop();
  }
}
