import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, ActivatedRoute } from '@angular/router';
import { BlogService } from '../../services/blog.service';
import { Post } from '../../models/post.interface';

@Component({
  selector: 'app-post',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './post.component.html',
  styleUrls: ['./post.component.scss']
})
export class PostComponent implements OnInit {
  post: Post | null = null;
  loading: boolean = true;

  constructor(
    private blogService: BlogService,
    private route: ActivatedRoute
  ) {}

  ngOnInit() {
    this.route.params.subscribe(params => {
      const slug = params['slug'];
      this.loadPost(slug);
    });
  }

  private loadPost(slug: string) {
    this.loading = true;
    this.blogService.getPost(slug).subscribe(post => {
      this.post = post;
      this.loading = false;
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
} 