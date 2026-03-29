import { Component, OnInit, AfterViewInit, OnDestroy, ElementRef, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule, ActivatedRoute } from '@angular/router';
import { BlogService } from '../../services/blog.service';
import { Post, PostMetadata } from '../../models/post.interface';
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
  prevPost: PostMetadata | null = null;
  nextPost: PostMetadata | null = null;
  loading: boolean = true;
  showImageViewer: boolean = false;
  selectedImageUrl: string = '';

  constructor(
    private blogService: BlogService,
    private route: ActivatedRoute,
    private router: Router,
    private elementRef: ElementRef<HTMLElement>
  ) {}

  @HostListener('document:keydown', ['$event'])
  onDocumentKeydown(event: KeyboardEvent): void {
    if (!this.post?.content || this.loading) {
      return;
    }
    if (!event.ctrlKey || !event.shiftKey) {
      return;
    }
    if (this.isTypingTarget(event.target)) {
      return;
    }

    if (event.key === 'ArrowLeft') {
      if (!this.prevPost) {
        return;
      }
      event.preventDefault();
      void this.router.navigate(['/post', this.prevPost.slug]);
    } else if (event.key === 'ArrowRight') {
      if (!this.nextPost) {
        return;
      }
      event.preventDefault();
      void this.router.navigate(['/post', this.nextPost.slug]);
    }
  }

  private isTypingTarget(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) {
      return false;
    }
    return !!target.closest('input, textarea, select, [contenteditable="true"]');
  }

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
    this.prevPost = null;
    this.nextPost = null;
    this.blogService.getPost(slug).subscribe({
      next: (post) => {
        this.post = post;
        this.loading = false;
        if (post) {
          const adj = this.blogService.getAdjacentPosts(slug);
          this.prevPost = adj.prev;
          this.nextPost = adj.next;
        }
      },
      error: () => {
        this.post = null;
        this.prevPost = null;
        this.nextPost = null;
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