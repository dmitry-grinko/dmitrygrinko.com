import { Component, OnInit, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, ActivatedRoute, Router } from '@angular/router';
import { BlogService } from '../../services/blog.service';
import { PostTagsService } from '../../services/post-tags.service';
import { PostMetadata } from '../../models/post.interface';

@Component({
  selector: 'app-tag',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './tag.component.html',
  styleUrls: ['./tag.component.scss']
})
export class TagComponent implements OnInit {
  tagLabel = '';
  posts: PostMetadata[] = [];
  focusedPostIndex = -1;

  constructor(
    private blogService: BlogService,
    private postTags: PostTagsService,
    private route: ActivatedRoute,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      const raw = params.get('tag') ?? '';
      try {
        this.tagLabel = decodeURIComponent(raw);
      } catch {
        this.tagLabel = raw;
      }
      this.loadPosts();
    });
  }

  private loadPosts(): void {
    const slugs = new Set(this.postTags.getSlugsForTag(this.tagLabel));
    this.blogService.getAllPosts().subscribe(all => {
      this.posts = all.filter(p => slugs.has(p.slug));
      this.autoFocusFirstPost();
    });
  }

  getCategorySlug(categoryName?: string, categorySlug?: string): string {
    if (categorySlug) {
      return categorySlug;
    }
    if (!categoryName) {
      return '';
    }
    const name = categoryName.toLowerCase();
    switch (name) {
      case 'mathematics':
        return 'mathematics';
      case 'python':
        return 'python';
      case 'machine learning algorithms':
        return 'machine-learning-algorithms';
      case '0 — machine learning fundamentals':
        return 'stage-0-machine-learning-fundamentals';
      default:
        return name.replace(/\s+/g, '-');
    }
  }

  getSubcategorySlug(subcategoryName?: string, subcategorySlug?: string): string {
    if (subcategorySlug) {
      return subcategorySlug;
    }
    if (!subcategoryName) {
      return '';
    }
    return subcategoryName.toLowerCase().replace(/\s+/g, '-');
  }

  private autoFocusFirstPost(): void {
    if (this.posts.length > 0) {
      this.focusedPostIndex = 0;
      setTimeout(() => this.scrollToFocusedPost(), 100);
    } else {
      this.focusedPostIndex = -1;
    }
  }

  @HostListener('window:keydown', ['$event'])
  onKeyDown(event: KeyboardEvent): void {
    if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Enter'].includes(event.key) || event.ctrlKey) {
      return;
    }
    const target = event.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
      return;
    }
    if (this.posts.length === 0) {
      return;
    }
    event.preventDefault();
    switch (event.key) {
      case 'ArrowDown':
      case 'ArrowRight':
        this.navigateToNext();
        break;
      case 'ArrowUp':
      case 'ArrowLeft':
        this.navigateToPrevious();
        break;
      case 'Enter':
        this.openFocusedPost();
        break;
    }
  }

  private navigateToNext(): void {
    if (this.posts.length === 0) {
      return;
    }
    this.focusedPostIndex = (this.focusedPostIndex + 1) % this.posts.length;
    this.scrollToFocusedPost();
  }

  private navigateToPrevious(): void {
    if (this.posts.length === 0) {
      return;
    }
    if (this.focusedPostIndex <= 0) {
      this.focusedPostIndex = this.posts.length - 1;
    } else {
      this.focusedPostIndex--;
    }
    this.scrollToFocusedPost();
  }

  private scrollToFocusedPost(): void {
    setTimeout(() => {
      const el = document.querySelector(`[data-post-index="${this.focusedPostIndex}"]`);
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 0);
  }

  private openFocusedPost(): void {
    if (this.focusedPostIndex >= 0 && this.focusedPostIndex < this.posts.length) {
      void this.router.navigate(['/post', this.posts[this.focusedPostIndex].slug]);
    }
  }

  isPostFocused(index: number): boolean {
    return this.focusedPostIndex === index;
  }
}
