import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { BlogService } from '../../services/blog.service';
import { CategoryTree } from '../../models/post.interface';

@Component({
  selector: 'app-categories',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './categories.component.html',
  styleUrls: ['./categories.component.scss']
})
export class CategoriesComponent implements OnInit {
  categories: CategoryTree[] = [];

  constructor(private blogService: BlogService) {}

  ngOnInit(): void {
    this.blogService.getCategoryTree().subscribe(tree => {
      this.categories = tree;
    });
  }
}
