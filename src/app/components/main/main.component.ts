import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';

@Component({
  selector: 'app-main',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.scss']
})
export class MainComponent {
  /** Topics from “What you'll find here” → `/category/:slug` */
  readonly topicCategoryLinks: ReadonlyArray<{ label: string; slug: string }> = [
    { label: 'machine learning fundamentals', slug: 'stage-0-machine-learning-fundamentals' },
    { label: 'PyTorch engineering', slug: 'stage-1-pytorch-engineering' },
    { label: 'distributed training', slug: 'stage-3-distributed-training' },
    { label: 'GPU and performance optimization', slug: 'stage-2-gpu-and-performance-engineering' },
    { label: 'ML infrastructure', slug: 'stage-4-ml-infrastructure' },
    { label: 'large-scale model training', slug: 'stage-5-large-model-training' },
    { label: 'LLM systems', slug: 'stage-6-advanced-ml-systems' }
  ];

  getCurrentYear(): number {
    return new Date().getFullYear();
  }
} 