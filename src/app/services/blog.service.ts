import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of, forkJoin } from 'rxjs';
import { map, catchError, switchMap } from 'rxjs/operators';
import { marked } from 'marked';
import hljs from 'highlight.js';
import { ThemeService } from './theme.service';
import { Post, PostMetadata, CategoryTree, SubCategory, PredefinedCategories, PredefinedCategory, PredefinedSubcategory } from '../models/post.interface';

@Injectable({
  providedIn: 'root'
})
export class BlogService {
  private allPostsCache: PostMetadata[] | null = null;

  // Define all categories, subcategories, and posts with slugs matching folder structure
  private predefinedCategories: PredefinedCategories = {
    'mathematics': {
      name: 'Mathematics',
      slug: 'mathematics',
      subcategories: [
        {
          name: 'Linear Algebra',
          slug: 'linear-algebra',
          posts: [
            {
              title: 'Vectors',
              slug: 'vectors',
              excerpt: 'A comprehensive guide to vectors in linear algebra, including their definitions, operations, geometrical interpretations and applications in machine learning.'
            }
          ]
        },
        {
          name: 'Calculus',
          slug: 'calculus',
          posts: [
            {
              title: 'Introduction to Derivatives',
              slug: 'introduction-to-derivatives',
              excerpt: 'Learn the fundamental concepts of derivatives in calculus, including limits, rates of change, and basic differentiation rules.'
            }
          ]
        },
        {
          name: 'Probability Theory',
          slug: 'probability-theory',
          posts: []
        },
        {
          name: 'Statistics',
          slug: 'statistics',
          posts: []
        }
      ]
    },
    'python': {
      name: 'Python',
      slug: 'python',
      subcategories: [
        {
          name: 'Syntax',
          slug: 'syntax',
          posts: [
            {
              title: 'Python Variables and Data Types',
              slug: 'python-variables-data-types',
              excerpt: 'Learn about Python variables, data types, and basic syntax fundamentals.'
            }
          ]
        },
        {
          name: 'Libraries',
          slug: 'libraries',
          posts: [
            {
              title: 'NumPy',
              slug: 'numpy',
              excerpt: 'Introduction to NumPy for numerical computing in Python.'
            }
          ]
        }
      ]
    },
    'machine-learning-algorithms': {
      name: 'Machine Learning Algorithms',
      slug: 'machine-learning-algorithms',
      subcategories: [
        {
          name: 'Supervised Learning',
          slug: 'supervised-learning',
          posts: []
        },
        {
          name: 'Unsupervised Learning',
          slug: 'unsupervised-learning',
          posts: []
        },
        {
          name: 'Reinforcement Learning',
          slug: 'reinforcement-learning',
          posts: []
        }
      ]
    }
  };

  constructor(
    private http: HttpClient,
    private themeService: ThemeService
  ) {
    // Configure marked for better security and formatting
    marked.setOptions({
      gfm: true,
      breaks: true
    });

    // Configure custom renderer for syntax highlighting and image handling
    const renderer = new marked.Renderer();
    
    renderer.code = ({ text, lang }: { text: string; lang?: string }): string => {
      if (lang && hljs.getLanguage(lang)) {
        try {
          const highlighted = hljs.highlight(text, { language: lang }).value;
          const themeClass = this.themeService.currentTheme === 'dark' ? 'hljs-dark' : 'hljs-light';
          return `<pre><code class="hljs language-${lang} ${themeClass}">${highlighted}</code></pre>`;
        } catch (err) {
          console.warn('Highlight.js error:', err);
        }
      }
      
      // Fallback to auto-detection
      const autoDetected = hljs.highlightAuto(text);
      const themeClass = this.themeService.currentTheme === 'dark' ? 'hljs-dark' : 'hljs-light';
      return `<pre><code class="hljs ${themeClass}">${autoDetected.value}</code></pre>`;
    };

    // Configure image renderer for proper path resolution
    renderer.image = function({ href, title, text }: { href: string; title: string | null; text: string }): string {
      // If the image path doesn't start with http/https, treat it as a local image
      if (!href.startsWith('http://') && !href.startsWith('https://')) {
        // Resolve relative image paths to the data/images directory
        href = `/data/images/${href}`;
      }
      
      const titleAttr = title ? ` title="${title}"` : '';
      const altAttr = text ? ` alt="${text}"` : '';
      
      return `<img src="${href}"${altAttr}${titleAttr} class="post-image">`;
    };

    marked.setOptions({ renderer });
  }

  private loadAllPostsMetadata(): Observable<PostMetadata[]> {
    if (this.allPostsCache) {
      return of(this.allPostsCache);
    }

    // Extract all posts from predefined categories with category/subcategory info
    const allPosts: PostMetadata[] = [];
    
    Object.values(this.predefinedCategories).forEach(category => {
      category.subcategories.forEach(subcategory => {
        subcategory.posts.forEach(post => {
          allPosts.push({
            ...post,
            category: category.name,
            subcategory: subcategory.name
          });
        });
      });
    });

    this.allPostsCache = allPosts;
    return of(this.allPostsCache);
  }

  getAllPosts(): Observable<PostMetadata[]> {
    return this.loadAllPostsMetadata();
  }

  getPostsByCategory(categorySlug: string): Observable<PostMetadata[]> {
    const categoryData = this.predefinedCategories[categorySlug];
    if (!categoryData) {
      return of([]);
    }

    const posts: PostMetadata[] = [];
    categoryData.subcategories.forEach(subcategory => {
      subcategory.posts.forEach(post => {
        posts.push({
          ...post,
          category: categoryData.name,
          subcategory: subcategory.name
        });
      });
    });

    return of(posts);
  }

  getPost(slug: string): Observable<Post | null> {
    // Find the post metadata from predefined categories
    let foundPost: PostMetadata | null = null;
    let foundCategory: PredefinedCategory | null = null;
    let foundSubcategory: PredefinedSubcategory | null = null;

    Object.values(this.predefinedCategories).forEach(category => {
      category.subcategories.forEach(subcategory => {
        const post = subcategory.posts.find(p => p.slug === slug);
        if (post) {
          foundPost = post;
          foundCategory = category;
          foundSubcategory = subcategory;
        }
      });
    });

    if (!foundPost || !foundCategory || !foundSubcategory) {
      return of(null);
    }

    // Build file path: /data/posts/{category-slug}/{subcategory-slug}/{post-slug}.md
    const categorySlug = (foundCategory as PredefinedCategory).slug;
    const subcategorySlug = (foundSubcategory as PredefinedSubcategory).slug;
    const filePath = `/data/posts/${categorySlug}/${subcategorySlug}/${slug}.md`;

    return this.http.get(filePath, { responseType: 'text' }).pipe(
      map(markdown => {
        const content = this.parseMarkdown(markdown);
        return { 
          ...foundPost!, 
          category: foundCategory!.name,
          subcategory: foundSubcategory!.name,
          content 
        } as Post;
      }),
      catchError(() => of(null))
    );
  }

  getCategories(): Observable<string[]> {
    return of(Object.values(this.predefinedCategories).map(cat => cat.name));
  }

  getCategoryTree(): Observable<CategoryTree[]> {
    const result: CategoryTree[] = [];

    Object.values(this.predefinedCategories).forEach(category => {
      const categoryTree: CategoryTree = {
        name: category.name,
        slug: category.slug,
        count: 0,
        subcategories: []
      };

      category.subcategories.forEach(subcategory => {
        const subCategoryData: SubCategory = {
          name: subcategory.name,
          slug: subcategory.slug,
          count: subcategory.posts.length,
          posts: subcategory.posts
        };
        
        categoryTree.subcategories.push(subCategoryData);
        categoryTree.count += subcategory.posts.length;
      });

      result.push(categoryTree);
    });

    // Sort categories by count (descending), then alphabetically
    return of(result.sort((a, b) => {
      if (b.count !== a.count) {
        return b.count - a.count;
      }
      return a.name.localeCompare(b.name);
    }));
  }

  getPostsBySubcategory(categorySlug: string, subcategorySlug: string): Observable<PostMetadata[]> {
    const categoryData = this.predefinedCategories[categorySlug];
    if (!categoryData) {
      return of([]);
    }

    const subcategoryData = categoryData.subcategories.find(sub => sub.slug === subcategorySlug);
    if (!subcategoryData) {
      return of([]);
    }

    const posts: PostMetadata[] = subcategoryData.posts.map(post => ({
      ...post,
      category: categoryData.name,
      subcategory: subcategoryData.name
    }));

    return of(posts);
  }

  private parseMarkdown(markdown: string): string {
    // Add standard footer section to all posts
    const footerSection = `
<br/><br/>
<br/><br/>
## Let's Connect!

Found this helpful? Have questions or spotted something I missed? 

I'd love to hear from you on [LinkedIn](https://www.linkedin.com/in/dmitrygrinko/)! Your feedback helps me write better posts and keeps the conversation going.`;

    const contentWithFooter = markdown + footerSection;
    return marked(contentWithFooter) as string;
  }
} 