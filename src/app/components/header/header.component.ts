import { Component, OnInit, OnDestroy, HostListener, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { BlogService } from '../../services/blog.service';
import { ThemeService, Theme } from '../../services/theme.service';
import { PostMetadata, CategoryTree } from '../../models/post.interface';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.scss']
})
export class HeaderComponent implements OnInit, OnDestroy {
  @ViewChild('searchInput') searchInput!: ElementRef<HTMLInputElement>;
  
  categories: string[] = [];
  categoryTree: CategoryTree[] = [];
  searchQuery: string = '';
  searchResults: PostMetadata[] = [];
  allPosts: PostMetadata[] = [];
  isCategoriesOpen: boolean = false;
  expandedCategories: Set<string> = new Set();
  isSearchExpanded: boolean = false;
  currentCategoryIndex: number = -1;
  focusedSearchIndex: number = -1;
  currentTheme: Theme = 'light';

  constructor(
    private blogService: BlogService,
    private router: Router,
    private themeService: ThemeService
  ) {}

  ngOnInit() {
    this.loadCategories();
    this.loadCategoryTree();
    this.loadAllPosts();
    
    // Subscribe to theme changes
    this.themeService.currentTheme$.subscribe(theme => {
      this.currentTheme = theme;
    });
    
    // Initialize system theme listener
    this.themeService.initSystemThemeListener();
  }

  ngOnDestroy() {
    // Clean up scroll lock when component is destroyed
    document.body.classList.remove('scroll-lock');
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: Event) {
    const target = event.target as HTMLElement;
    
    // Close search if clicking outside search area
    if (!target.closest('.search-container') && !target.closest('.search-results')) {
      if (this.isSearchExpanded && !this.searchQuery && this.searchResults.length === 0) {
        this.collapseSearch();
      } else {
        this.clearSearch();
      }
    }
    
    // Close categories dropdown if clicking outside categories area
    if (!target.closest('.categories-dropdown')) {
      this.closeCategoriesDropdown();
    }
  }

  @HostListener('document:keydown', ['$event'])
  onKeyDown(event: KeyboardEvent) {
    const target = event.target as HTMLElement;

    // Handle Escape key globally (works from anywhere)
    if (event.key === 'Escape') {
      if (this.isSearchExpanded) {
        this.collapseSearch();
        event.preventDefault();
        return;
      }
    }

    // Handle Ctrl+S globally (works from anywhere)
    if (event.ctrlKey && event.key === 's') {
      if (this.isSearchExpanded) {
        this.collapseSearch();
      } else {
        this.expandSearch();
      }
      event.preventDefault();
      return;
    }

    // Handle Ctrl+C globally (works from anywhere)
    if (event.ctrlKey && event.key === 'c') {
      this.cycleToNextCategory();
      event.preventDefault();
      return;
    }

    // Handle search navigation when search input is focused
    if (target === this.searchInput?.nativeElement && this.searchResults.length > 0) {
      if (['ArrowUp', 'ArrowDown', 'Enter'].includes(event.key)) {
        this.handleSearchNavigation(event);
        return;
      }
    }

    // Don't interfere with other input fields
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
      return;
    }

    // Handle search navigation when search is visible (from outside input)
    if (this.searchResults.length > 0 && ['ArrowUp', 'ArrowDown', 'Enter'].includes(event.key)) {
      this.handleSearchNavigation(event);
      return;
    }
  }

  private handleSearchNavigation(event: KeyboardEvent) {
    if (this.searchResults.length === 0) return;

    event.preventDefault();

    switch (event.key) {
      case 'ArrowDown':
        this.navigateToNextSearchResult();
        break;
      case 'ArrowUp':
        this.navigateToPreviousSearchResult();
        break;
      case 'Enter':
        this.openFocusedSearchResult();
        break;
    }
  }

  private navigateToNextSearchResult() {
    if (this.searchResults.length === 0) return;
    
    this.focusedSearchIndex = (this.focusedSearchIndex + 1) % this.searchResults.length;
  }

  private navigateToPreviousSearchResult() {
    if (this.searchResults.length === 0) return;
    
    if (this.focusedSearchIndex <= 0) {
      this.focusedSearchIndex = this.searchResults.length - 1;
    } else {
      this.focusedSearchIndex--;
    }
  }

  private openFocusedSearchResult() {
    if (this.focusedSearchIndex >= 0 && this.focusedSearchIndex < this.searchResults.length) {
      const post = this.searchResults[this.focusedSearchIndex];
      this.router.navigate(['/post', post.slug]);
      this.clearSearch();
      this.collapseSearch();
    }
  }

  private loadCategories() {
    this.blogService.getCategories().subscribe(categories => {
      this.categories = categories;
    });
  }

  private loadCategoryTree() {
    this.blogService.getCategoryTree().subscribe(tree => {
      this.categoryTree = tree;
    });
  }

  private loadAllPosts() {
    this.blogService.getAllPosts().subscribe(posts => {
      this.allPosts = posts;
    });
  }

  toggleCategoriesDropdown() {
    this.isCategoriesOpen = !this.isCategoriesOpen;
    this.updateBodyScrollLock();
  }

  closeCategoriesDropdown() {
    this.isCategoriesOpen = false;
    this.updateBodyScrollLock();
  }

  private updateBodyScrollLock() {
    if (this.isCategoriesOpen) {
      document.body.classList.add('scroll-lock');
    } else {
      document.body.classList.remove('scroll-lock');
    }
  }

  toggleCategory(categoryName: string) {
    if (this.expandedCategories.has(categoryName)) {
      this.expandedCategories.delete(categoryName);
    } else {
      this.expandedCategories.add(categoryName);
    }
  }

  isCategoryExpanded(categoryName: string): boolean {
    return this.expandedCategories.has(categoryName);
  }

  getCategoriesWithSubcategories(): CategoryTree[] {
    return this.categoryTree.filter(category => 
      category.subcategories && category.subcategories.length > 0
    );
  }



  onSearch() {
    if (this.searchQuery.trim().length === 0) {
      this.searchResults = [];
      this.focusedSearchIndex = -1;
      return;
    }

    const query = this.searchQuery.toLowerCase().trim();
    this.searchResults = this.allPosts.filter(post => 
      post.title.toLowerCase().includes(query) ||
      post.excerpt.toLowerCase().includes(query)
    ).slice(0, 5); // Limit to 5 results
    
    // Auto-focus first search result if results exist
    this.autoFocusFirstSearchResult();
  }

  private autoFocusFirstSearchResult() {
    if (this.searchResults.length > 0) {
      this.focusedSearchIndex = 0;
    } else {
      this.focusedSearchIndex = -1;
    }
  }

  performSearch() {
    if (this.searchQuery.trim().length === 0) return;
    
    // Navigate to blog with search results
    this.router.navigate(['/blog'], { 
      queryParams: { search: this.searchQuery.trim() } 
    });
    this.clearSearch();
  }

  clearSearch() {
    this.searchQuery = '';
    this.searchResults = [];
    this.focusedSearchIndex = -1;
  }

  expandSearch() {
    this.isSearchExpanded = true;
    // Focus the input after the animation starts
    setTimeout(() => {
      if (this.searchInput) {
        this.searchInput.nativeElement.focus();
      }
    }, 100);
  }

  collapseSearch() {
    this.isSearchExpanded = false;
    this.clearSearch();
  }

  onSearchBlur() {
    // Only collapse if there's no search query and no results
    if (!this.searchQuery && this.searchResults.length === 0) {
      setTimeout(() => {
        this.isSearchExpanded = false;
      }, 200); // Small delay to allow click events on results
    }
  }

  cycleToNextCategory() {
    const categoriesWithPosts = this.getCategoriesWithSubcategories();
    
    if (categoriesWithPosts.length === 0) {
      return;
    }

    // Move to next category (cycle back to 0 if at the end)
    this.currentCategoryIndex = (this.currentCategoryIndex + 1) % categoriesWithPosts.length;
    const targetCategory = categoriesWithPosts[this.currentCategoryIndex];
    
    this.router.navigate(['/category', targetCategory.slug]);
  }

  isSearchResultFocused(index: number): boolean {
    return this.focusedSearchIndex === index;
  }

  toggleTheme(): void {
    this.themeService.toggleTheme();
  }

  getThemeIcon(): string {
    return this.currentTheme === 'light' ? '🌙' : '☀️';
  }

  getThemeLabel(): string {
    return this.currentTheme === 'light' ? 'Switch to dark mode' : 'Switch to light mode';
  }
} 