import {
  Component,
  OnInit,
  AfterViewInit,
  OnDestroy,
  ElementRef,
  HostListener,
  ViewChild
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule, ActivatedRoute } from '@angular/router';
import { BlogService } from '../../services/blog.service';
import { PostTagsService } from '../../services/post-tags.service';
import { Post, PostMetadata } from '../../models/post.interface';
import { ImageViewerComponent } from '../image-viewer/image-viewer.component';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-post',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule, ImageViewerComponent],
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
  postTags: string[] = [];
  newTagInput = '';
  showTagInput = false;

  @ViewChild('tagInput') private tagInputRef?: ElementRef<HTMLInputElement>;

  private tagsSub?: Subscription;

  constructor(
    private blogService: BlogService,
    private postTagsService: PostTagsService,
    private route: ActivatedRoute,
    private router: Router,
    private elementRef: ElementRef<HTMLElement>
  ) {}

  @HostListener('document:keydown', ['$event'])
  onDocumentKeydown(event: KeyboardEvent): void {
    if (!this.post?.content || this.loading) {
      return;
    }
    if (event.ctrlKey || event.shiftKey || event.altKey || event.metaKey) {
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
    this.tagsSub = this.postTagsService.tagsChanged$.subscribe(() => this.syncTagsFromStore());
    this.route.params.subscribe(params => {
      const slug = params['slug'];
      this.loadPost(slug);
    });
  }

  ngAfterViewInit(): void {
    this.elementRef.nativeElement.addEventListener('click', this.onPostImageClick);
  }

  ngOnDestroy(): void {
    this.tagsSub?.unsubscribe();
    this.elementRef.nativeElement.removeEventListener('click', this.onPostImageClick);
  }

  /** Keeps tags in sync when storage changes (e.g. Reset in header). */
  private syncTagsFromStore(): void {
    if (this.loading || !this.post?.slug) {
      return;
    }
    this.postTags = this.postTagsService.getTagsForPost(this.post.slug);
    this.newTagInput = '';
    this.showTagInput = false;
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
          this.postTags = this.postTagsService.getTagsForPost(slug);
          this.newTagInput = '';
          this.showTagInput = false;
        } else {
          this.postTags = [];
          this.showTagInput = false;
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
      case '0 — machine learning fundamentals': return 'stage-0-machine-learning-fundamentals';
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

  get tagsForDatalist(): string[] {
    return this.postTagsService.getAllTags().filter(
      t => !this.postTags.some(p => p.toLowerCase() === t.toLowerCase())
    );
  }

  addTag(): void {
    if (!this.post?.slug) {
      return;
    }
    const raw = this.newTagInput.trim().replace(/\s+/g, ' ');
    if (!raw) {
      return;
    }
    if (this.postTags.some(t => t.toLowerCase() === raw.toLowerCase())) {
      this.newTagInput = '';
      return;
    }
    const next = [...this.postTags, raw];
    this.postTagsService.setTagsForPost(this.post.slug, next);
    this.postTags = this.postTagsService.getTagsForPost(this.post.slug);
    this.newTagInput = '';
    this.tagInputRef?.nativeElement?.blur();
    this.showTagInput = false;
  }

  removeTag(tag: string): void {
    if (!this.post?.slug) {
      return;
    }
    const next = this.postTags.filter(t => t !== tag);
    this.postTagsService.setTagsForPost(this.post.slug, next);
    this.postTags = this.postTagsService.getTagsForPost(this.post.slug);
  }

  toggleTagInput(): void {
    if (this.showTagInput) {
      this.tagInputRef?.nativeElement?.blur();
      this.newTagInput = '';
    }
    this.showTagInput = !this.showTagInput;
    if (this.showTagInput) {
      setTimeout(() => this.tagInputRef?.nativeElement?.focus(), 0);
    }
  }
} 