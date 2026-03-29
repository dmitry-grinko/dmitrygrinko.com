import { Component, OnInit, AfterViewInit, OnDestroy, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, ActivatedRoute } from '@angular/router';
import { BlogService } from '../../services/blog.service';
import { Post } from '../../models/post.interface';
import { ImageViewerComponent } from '../image-viewer/image-viewer.component';

@Component({
  selector: 'app-post',
  standalone: true,
  imports: [CommonModule, RouterModule, ImageViewerComponent],
  templateUrl: './post.component.html',
  styleUrls: ['./post.component.scss']
})
export class PostComponent implements OnInit, AfterViewInit, OnDestroy {
  post: Post | null = null;
  loading: boolean = true;
  showImageViewer: boolean = false;
  selectedImageUrl: string = '';

  constructor(
    private blogService: BlogService,
    private route: ActivatedRoute,
    private elementRef: ElementRef<HTMLElement>
  ) {}

  ngOnInit() {
    this.route.params.subscribe(params => {
      const slug = params['slug'];
      this.loadPost(slug);
    });
  }

  ngAfterViewInit(): void {
    this.elementRef.nativeElement.addEventListener('click', this.onPostImageClick);
  }

  ngOnDestroy(): void {
    this.elementRef.nativeElement.removeEventListener('click', this.onPostImageClick);
  }

  /** Delegated handler: works after innerHTML updates (avoids race with setTimeout vs change detection). */
  private onPostImageClick = (event: MouseEvent): void => {
    const target = event.target;
    if (!(target instanceof HTMLImageElement)) {
      return;
    }
    const contentEl = this.elementRef.nativeElement.querySelector('.content');
    if (!contentEl || !contentEl.contains(target)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    this.selectedImageUrl = target.currentSrc || target.src;
    this.showImageViewer = true;
  };

  private loadPost(slug: string) {
    this.loading = true;
    this.blogService.getPost(slug).subscribe({
      next: (post) => {
        this.post = post;
        this.loading = false;
      },
      error: () => {
        this.post = null;
        this.loading = false;
      }
    });
  }

  getCategorySlug(): string {
    if (this.post?.categorySlug) {
      return this.post.categorySlug;
    }
    if (!this.post?.category) return '';
    
    const categoryName = this.post.category.toLowerCase();
    switch (categoryName) {
      case 'mathematics': return 'mathematics';
      case 'python': return 'python';
      case 'machine learning algorithms': return 'machine-learning-algorithms';
      case 'stage 0 — machine learning fundamentals': return 'stage-0-machine-learning-fundamentals';
      default: return categoryName.replace(/\s+/g, '-');
    }
  }

  getSubcategorySlug(): string {
    if (this.post?.subcategorySlug) {
      return this.post.subcategorySlug;
    }
    if (!this.post?.subcategory) return '';
    
    return this.post.subcategory.toLowerCase().replace(/\s+/g, '-');
  }

  closeImageViewer() {
    this.showImageViewer = false;
    this.selectedImageUrl = '';
  }
} 