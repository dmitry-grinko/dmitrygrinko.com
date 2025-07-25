import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./components/main/main.component').then(m => m.MainComponent)
  },
  {
    path: 'blog',
    loadComponent: () => import('./components/home/home.component').then(m => m.HomeComponent)
  },
  {
    path: 'category/:category',
    loadComponent: () => import('./components/category/category.component').then(m => m.CategoryComponent)
  },
  {
    path: 'subcategory/:category/:subcategory',
    loadComponent: () => import('./components/subcategory/subcategory.component').then(m => m.SubcategoryComponent)
  },
  {
    path: 'post/:slug',
    loadComponent: () => import('./components/post/post.component').then(m => m.PostComponent)
  },
  {
    path: 'help',
    loadComponent: () => import('./components/help/help.component').then(m => m.HelpComponent)
  },
  {
    path: '**',
    redirectTo: ''
  }
];
