import { Component, OnInit, AfterViewInit, ElementRef } from '@angular/core';
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
export class PostComponent implements OnInit, AfterViewInit {
  post: Post | null = null;
  loading: boolean = true;
  showImageViewer: boolean = false;
  selectedImageUrl: string = '';

  constructor(
    private blogService: BlogService,
    private route: ActivatedRoute,
    private elementRef: ElementRef
  ) {}

  ngOnInit() {
    this.route.params.subscribe(params => {
      const slug = params['slug'];
      this.loadPost(slug);
    });
  }

  ngAfterViewInit() {
    this.setupImageClickHandlers();
  }

  private setupImageClickHandlers() {
    // Add click event listeners to all clickable images
    const images = this.elementRef.nativeElement.querySelectorAll('.clickable-image');
    images.forEach((img: HTMLImageElement) => {
      img.addEventListener('click', () => {
        this.selectedImageUrl = img.src;
        this.showImageViewer = true;
      });
    });
  }

  private loadPost(slug: string) {
    this.loading = true;
    this.blogService.getPost(slug).subscribe(post => {
      this.post = post;
      this.loading = false;
      // Setup image click handlers after content is loaded
      setTimeout(() => this.setupImageClickHandlers(), 0);
    });
  }

  getCategorySlug(): string {
    if (!this.post?.category) return '';
    
    // Convert category name back to slug
    const categoryName = this.post.category.toLowerCase();
    switch (categoryName) {
      case 'mathematics': return 'mathematics';
      case 'python': return 'python';
      case 'machine learning algorithms': return 'machine-learning-algorithms';
      default: return categoryName.replace(/\s+/g, '-');
    }
  }

  getSubcategorySlug(): string {
    if (!this.post?.subcategory) return '';
    
    // Convert subcategory name back to slug
    return this.post.subcategory.toLowerCase().replace(/\s+/g, '-');
  }

  closeImageViewer() {
    this.showImageViewer = false;
    this.selectedImageUrl = '';
  }
} 