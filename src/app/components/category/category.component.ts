import { Component, OnInit, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, ActivatedRoute, Router } from '@angular/router';
import { BlogService } from '../../services/blog.service';
import { PostMetadata } from '../../models/post.interface';

@Component({
  selector: 'app-category',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './category.component.html',
  styleUrls: ['./category.component.scss']
})
export class CategoryComponent implements OnInit {
  posts: PostMetadata[] = [];
  categorySlug: string = '';
  categoryDisplayName: string = '';
  focusedPostIndex: number = -1;

  constructor(
    private blogService: BlogService,
    private route: ActivatedRoute,
    private router: Router
  ) {}

  ngOnInit() {
    this.route.params.subscribe(params => {
      this.categorySlug = params['category'];
      this.loadCategoryData();
      this.loadPosts();
    });
  }

  private loadCategoryData() {
    this.blogService.getCategoryTree().subscribe(categories => {
      const category = categories.find(cat => cat.slug === this.categorySlug);
      this.categoryDisplayName = category ? category.name : this.categorySlug;
    });
  }

  private loadPosts() {
    this.blogService.getPostsByCategory(this.categorySlug).subscribe(posts => {
      this.posts = posts;
      
      // Auto-focus first post if posts exist
      this.autoFocusFirstPost();
    });
  }

  private autoFocusFirstPost() {
    if (this.posts.length > 0) {
      this.focusedPostIndex = 0;
      // Small delay to ensure DOM is updated
      setTimeout(() => {
        this.scrollToFocusedPost();
      }, 100);
    } else {
      this.focusedPostIndex = -1;
    }
  }

  getCategoryDisplayName(): string {
    return this.categoryDisplayName;
  }

  getSubcategorySlug(subcategoryName?: string): string {
    if (!subcategoryName) return '';
    
    // Convert subcategory name back to slug
    return subcategoryName.toLowerCase().replace(/\s+/g, '-');
  }

  @HostListener('window:keydown', ['$event'])
  onKeyDown(event: KeyboardEvent) {
    // Only handle arrow keys and Enter, and skip if Ctrl is pressed (to avoid conflicts with global shortcuts)
    if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Enter'].includes(event.key) || event.ctrlKey) {
      return;
    }

    // Don't interfere with input fields
    const target = event.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
      return;
    }

    // Only work if we have posts to navigate
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

  private navigateToNext() {
    if (this.posts.length === 0) return;
    
    this.focusedPostIndex = (this.focusedPostIndex + 1) % this.posts.length;
    this.scrollToFocusedPost();
  }

  private navigateToPrevious() {
    if (this.posts.length === 0) return;
    
    if (this.focusedPostIndex <= 0) {
      this.focusedPostIndex = this.posts.length - 1;
    } else {
      this.focusedPostIndex--;
    }
    this.scrollToFocusedPost();
  }

  private scrollToFocusedPost() {
    setTimeout(() => {
      const focusedElement = document.querySelector(`[data-post-index="${this.focusedPostIndex}"]`);
      if (focusedElement) {
        focusedElement.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center' 
        });
      }
    }, 0);
  }

  private openFocusedPost() {
    if (this.focusedPostIndex >= 0 && this.focusedPostIndex < this.posts.length) {
      const post = this.posts[this.focusedPostIndex];
      this.router.navigate(['/post', post.slug]);
    }
  }

  isPostFocused(index: number): boolean {
    return this.focusedPostIndex === index;
  }
} 