import { Component, OnInit, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, ActivatedRoute, Router } from '@angular/router';
import { BlogService } from '../../services/blog.service';
import { PostMetadata } from '../../models/post.interface';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {
  posts: PostMetadata[] = [];
  searchQuery: string = '';
  focusedPostIndex: number = -1;

  constructor(
    private blogService: BlogService,
    private route: ActivatedRoute,
    private router: Router
  ) {}

  ngOnInit() {
    this.route.queryParams.subscribe(params => {
      this.searchQuery = params['search'] || '';
      this.loadPosts();
    });
  }

  private loadPosts() {
    this.blogService.getAllPosts().subscribe(posts => {
      if (this.searchQuery) {
        const query = this.searchQuery.toLowerCase().trim();
        this.posts = posts.filter(post => 
          post.title.toLowerCase().includes(query) ||
          post.excerpt.toLowerCase().includes(query)
        );
      } else {
        this.posts = posts;
      }
      
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

  getCategorySlug(categoryName?: string): string {
    if (!categoryName) return '';
    
    // Convert category name back to slug
    const name = categoryName.toLowerCase();
    switch (name) {
      case 'mathematics': return 'mathematics';
      case 'python': return 'python';
      case 'machine learning algorithms': return 'machine-learning-algorithms';
      default: return name.replace(/\s+/g, '-');
    }
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