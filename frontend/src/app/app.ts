import { Component } from '@angular/core';
import { CameraComponent } from './camera/camera.component';

@Component({
  selector: 'app-root',
  imports: [CameraComponent],
  template: '<app-camera />',
  styles: [':host { display: block; height: 100vh; }']
})
export class App {}
