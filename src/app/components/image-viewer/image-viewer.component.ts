import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-image-viewer',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="image-viewer" (click)="close.emit()">
      <img [src]="imageUrl" alt="Full screen image" (click)="$event.stopPropagation()">
    </div>
  `,
  styles: [`
    .image-viewer {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.9);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      cursor: pointer;
    }

    img {
      max-width: 90vw;
      max-height: 90vh;
      object-fit: contain;
      cursor: default;
    }
  `]
})
export class ImageViewerComponent {
  @Input() imageUrl: string = '';
  @Output() close = new EventEmitter<void>();
} 