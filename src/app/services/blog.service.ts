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

  /** Display / iteration order: foundations → ML → sequences → biology layers → applied bioinformatics → DNA ML */
  private readonly studyOrderCategorySlugs: string[] = [
    'mathematics',
    'machine-learning',
    'sequence-modeling',
    'molecular-biology',
    'genomics',
    'bioinformatics',
    'dna-machine-learning'
  ];

  private readonly studyOrderSubcategorySlugs: Record<string, string[]> = {
    mathematics: ['linear-algebra', 'probability-and-statistics'],
    'machine-learning': [
      'machine-learning-fundamentals',
      'deep-learning',
      'neural-network-architectures',
      'optimization-and-training',
      'ml-evaluation-metrics'
    ],
    'sequence-modeling': ['sequence-modeling', 'transformers-and-attention', 'embeddings'],
    'molecular-biology': [
      'molecular-biology',
      'genetics',
      'dna-structure-and-function',
      'mutations-and-variants'
    ],
    genomics: ['genomics', 'gene-regulation', 'gene-expression'],
    bioinformatics: [
      'dna-sequencing',
      'genomics-data-formats',
      'sequence-alignment',
      'variant-calling',
      'bioinformatics-tools',
      'genomics-pipelines'
    ],
    'dna-machine-learning': [
      'dna-sequence-analysis',
      'variant-effect-prediction',
      'regulatory-dna-modeling'
    ]
  };

  private compareBySlugOrder(order: string[], aSlug: string, bSlug: string): number {
    const ia = order.indexOf(aSlug);
    const ib = order.indexOf(bSlug);
    return (ia === -1 ? 1000 : ia) - (ib === -1 ? 1000 : ib);
  }

  // Define all categories, subcategories, and posts with slugs matching folder structure
  // Paths: /data/posts/{category-slug}/{subcategory-slug}/{post-slug}.md
  private predefinedCategories: PredefinedCategories = {
    mathematics: {
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
            },
            {
              title: 'Matrices and Matrix Operations',
              slug: 'matrices-operations',
              excerpt: 'Master matrices and matrix operations - the powerhouse data structures that drive machine learning computations, from neural networks to data transformations.'
            },
            {
              title: 'Eigenvalues and Eigenvectors',
              slug: 'eigenvalues-eigenvectors',
              excerpt: 'Discover eigenvalues and eigenvectors, the mathematical foundation behind PCA, PageRank, neural network stability analysis, and many other ML applications.'
            },
            {
              title: 'Linear Transformations',
              slug: 'linear-transformations',
              excerpt: 'Understand how matrices transform space through linear transformations - the geometric foundation of neural networks, computer vision, and feature engineering.'
            }
          ]
        },
        {
          name: 'Probability and Statistics',
          slug: 'probability-and-statistics',
          posts: [
            {
              title: 'Basic Probability Concepts',
              slug: 'basic-probability-concepts',
              excerpt: 'Master the fundamental concepts of probability theory - the mathematical language of uncertainty that underlies all machine learning algorithms.'
            },
            {
              title: 'Bayes\' Theorem and Applications',
              slug: 'bayes-theorem-applications',
              excerpt: 'Discover Bayes\' theorem, the mathematical heart of machine learning, and its powerful applications in spam detection, medical diagnosis, and AI systems.'
            },
            {
              title: 'Probability Distributions',
              slug: 'probability-distributions',
              excerpt: 'Explore discrete and continuous probability distributions that model real-world phenomena and power statistical methods in machine learning.'
            },
            {
              title: 'Central Limit Theorem',
              slug: 'central-limit-theorem',
              excerpt: 'Understand the Central Limit Theorem - the mathematical principle that explains why normal distributions appear everywhere and enables statistical inference.'
            },
            {
              title: 'Descriptive Statistics',
              slug: 'descriptive-statistics',
              excerpt: 'Master the fundamental tools for summarizing and describing data - the essential first step in any machine learning project and data analysis.'
            },
            {
              title: 'Hypothesis Testing',
              slug: 'hypothesis-testing',
              excerpt: 'Learn the statistical framework for making data-driven decisions, validating model improvements, and conducting rigorous A/B testing in machine learning.'
            },
            {
              title: 'Correlation vs. Causation',
              slug: 'correlation-causation',
              excerpt: 'Understand the crucial distinction between correlation and causation - a fundamental concept for avoiding misleading conclusions in data science and ML.'
            },
            {
              title: 'Statistical Inference',
              slug: 'statistical-inference',
              excerpt: 'Explore confidence intervals, hypothesis testing, and statistical power - the mathematical tools that enable us to draw reliable conclusions from data.'
            }
          ]
        }
      ]
    },
    'machine-learning': {
      name: 'Machine Learning',
      slug: 'machine-learning',
      subcategories: [
        {
          name: 'Machine Learning Fundamentals',
          slug: 'machine-learning-fundamentals',
          posts: [
            {
              title: 'Scikit-learn',
              slug: 'scikit-learn',
              excerpt: 'A comprehensive guide to scikit-learn, the most popular machine learning library in Python for building and evaluating ML models.'
            },
            {
              title: 'XGBoost',
              slug: 'xgboost',
              excerpt: 'Learn about XGBoost, the powerful gradient boosting framework that dominates machine learning competitions and real-world applications.'
            },
            {
              title: 'LightGBM',
              slug: 'lightgbm',
              excerpt: 'Explore LightGBM, Microsoft\'s fast and efficient gradient boosting framework designed for high performance and low memory usage.'
            },
            {
              title: 'CatBoost',
              slug: 'catboost',
              excerpt: 'Discover CatBoost, Yandex\'s gradient boosting library that excels at handling categorical features without preprocessing.'
            },
            {
              title: 'MLxtend',
              slug: 'mlxtend',
              excerpt: 'Introduction to MLxtend, a library of useful tools and extensions for everyday machine learning tasks and model evaluation.'
            }
          ]
        },
        {
          name: 'Deep Learning',
          slug: 'deep-learning',
          posts: [
            {
              title: 'TensorFlow',
              slug: 'tensorflow',
              excerpt: 'A complete guide to TensorFlow, Google\'s open-source platform for building and deploying machine learning models at scale.'
            },
            {
              title: 'Keras',
              slug: 'keras',
              excerpt: 'Learn Keras, the high-level neural networks API that makes deep learning accessible and easy to implement.'
            },
            {
              title: 'PyTorch',
              slug: 'torch',
              excerpt: 'Explore PyTorch, Facebook\'s dynamic deep learning framework favored by researchers for its flexibility and ease of use.'
            },
            {
              title: 'ONNX',
              slug: 'onnx',
              excerpt: 'Understanding ONNX, the open standard for representing machine learning models and enabling interoperability between frameworks.'
            }
          ]
        },
        { name: 'Neural Network Architectures', slug: 'neural-network-architectures', posts: [] },
        {
          name: 'Optimization and Training',
          slug: 'optimization-and-training',
          posts: [
            {
              title: 'Introduction to Derivatives',
              slug: 'introduction-to-derivatives',
              excerpt: 'Learn the fundamental concepts of derivatives in calculus, including limits, rates of change, and basic differentiation rules.'
            },
            {
              title: 'Partial Derivatives and Gradients',
              slug: 'partial-derivatives-gradients',
              excerpt: 'Understand partial derivatives and gradients - essential concepts for multivariable calculus and the mathematical foundation of machine learning optimization.'
            },
            {
              title: 'Chain Rule in Machine Learning',
              slug: 'chain-rule-ml',
              excerpt: 'Deep dive into the chain rule and its crucial role in backpropagation, the algorithm that makes neural network training possible.'
            },
            {
              title: 'Optimization and Critical Points',
              slug: 'optimization-critical-points',
              excerpt: 'Explore optimization theory, critical points, and gradient-based algorithms that power modern machine learning model training.'
            }
          ]
        },
        { name: 'ML Evaluation Metrics', slug: 'ml-evaluation-metrics', posts: [] }
      ]
    },
    'sequence-modeling': {
      name: 'Sequence Modeling',
      slug: 'sequence-modeling',
      subcategories: [
        { name: 'Sequence Modeling', slug: 'sequence-modeling', posts: [] },
        { name: 'Transformers and Attention', slug: 'transformers-and-attention', posts: [] },
        { name: 'Embeddings', slug: 'embeddings', posts: [] }
      ]
    },
    'molecular-biology': {
      name: 'Molecular Biology',
      slug: 'molecular-biology',
      subcategories: [
        { name: 'Molecular Biology', slug: 'molecular-biology', posts: [] },
        { name: 'Genetics', slug: 'genetics', posts: [] },
        {
          name: 'DNA Structure and Function',
          slug: 'dna-structure-and-function',
          posts: [
            {
              title: 'DNA and the Double Helix',
              slug: 'dna-double-helix',
              excerpt: 'Understand the structure of DNA — the double helix, base pairing, and why representing genomes as strings over {A, T, C, G} is the foundation of computational biology.'
            },
            {
              title: 'The Central Dogma',
              slug: 'central-dogma',
              excerpt: 'Follow the flow of genetic information from DNA to RNA to protein — transcription, splicing, translation, and why gene expression data is a natural input for ML models.'
            },
            {
              title: 'DNA Replication',
              slug: 'dna-replication',
              excerpt: 'Learn how cells copy their entire genome before dividing, why sequencing reads have directionality, and how replication errors become the mutations studied in genomics.'
            }
          ]
        },
        { name: 'Mutations and Variants', slug: 'mutations-and-variants', posts: [] }
      ]
    },
    genomics: {
      name: 'Genomics',
      slug: 'genomics',
      subcategories: [
        { name: 'Genomics', slug: 'genomics', posts: [] },
        { name: 'Gene Regulation', slug: 'gene-regulation', posts: [] },
        { name: 'Gene Expression', slug: 'gene-expression', posts: [] }
      ]
    },
    bioinformatics: {
      name: 'Bioinformatics',
      slug: 'bioinformatics',
      subcategories: [
        { name: 'DNA Sequencing', slug: 'dna-sequencing', posts: [] },
        { name: 'Genomics Data Formats', slug: 'genomics-data-formats', posts: [] },
        { name: 'Sequence Alignment', slug: 'sequence-alignment', posts: [] },
        { name: 'Variant Calling', slug: 'variant-calling', posts: [] },
        { name: 'Bioinformatics Tools', slug: 'bioinformatics-tools', posts: [] },
        { name: 'Genomics Pipelines', slug: 'genomics-pipelines', posts: [] }
      ]
    },
    'dna-machine-learning': {
      name: 'DNA Machine Learning',
      slug: 'dna-machine-learning',
      subcategories: [
        { name: 'DNA Sequence Analysis', slug: 'dna-sequence-analysis', posts: [] },
        { name: 'Variant Effect Prediction', slug: 'variant-effect-prediction', posts: [] },
        { name: 'Regulatory DNA Modeling', slug: 'regulatory-dna-modeling', posts: [] }
      ]
    }
  };

  // --- LEGACY predefinedCategories (replaced; kept for reference) ---
  //   private predefinedCategories: PredefinedCategories = {
  //     'mathematics': {
  //       name: 'Mathematics',
  //       slug: 'mathematics',
  //       subcategories: [
  //         {
  //           name: 'Linear Algebra',
  //           slug: 'linear-algebra',
  //           posts: [
  //             {
  //               title: 'Vectors',
  //               slug: 'vectors',
  //               excerpt: 'A comprehensive guide to vectors in linear algebra, including their definitions, operations, geometrical interpretations and applications in machine learning.'
  //             },
  //             {
  //               title: 'Matrices and Matrix Operations',
  //               slug: 'matrices-operations',
  //               excerpt: 'Master matrices and matrix operations - the powerhouse data structures that drive machine learning computations, from neural networks to data transformations.'
  //             },
  //             {
  //               title: 'Eigenvalues and Eigenvectors',
  //               slug: 'eigenvalues-eigenvectors',
  //               excerpt: 'Discover eigenvalues and eigenvectors, the mathematical foundation behind PCA, PageRank, neural network stability analysis, and many other ML applications.'
  //             },
  //             {
  //               title: 'Linear Transformations',
  //               slug: 'linear-transformations',
  //               excerpt: 'Understand how matrices transform space through linear transformations - the geometric foundation of neural networks, computer vision, and feature engineering.'
  //             }
  //           ]
  //         },
  //         {
  //           name: 'Calculus',
  //           slug: 'calculus',
  //           posts: [
  //             {
  //               title: 'Introduction to Derivatives',
  //               slug: 'introduction-to-derivatives',
  //               excerpt: 'Learn the fundamental concepts of derivatives in calculus, including limits, rates of change, and basic differentiation rules.'
  //             },
  //             {
  //               title: 'Partial Derivatives and Gradients',
  //               slug: 'partial-derivatives-gradients',
  //               excerpt: 'Understand partial derivatives and gradients - essential concepts for multivariable calculus and the mathematical foundation of machine learning optimization.'
  //             },
  //             {
  //               title: 'Chain Rule in Machine Learning',
  //               slug: 'chain-rule-ml',
  //               excerpt: 'Deep dive into the chain rule and its crucial role in backpropagation, the algorithm that makes neural network training possible.'
  //             },
  //             {
  //               title: 'Optimization and Critical Points',
  //               slug: 'optimization-critical-points',
  //               excerpt: 'Explore optimization theory, critical points, and gradient-based algorithms that power modern machine learning model training.'
  //             }
  //           ]
  //         },
  //         {
  //           name: 'Probability Theory',
  //           slug: 'probability-theory',
  //           posts: [
  //             {
  //               title: 'Basic Probability Concepts',
  //               slug: 'basic-probability-concepts',
  //               excerpt: 'Master the fundamental concepts of probability theory - the mathematical language of uncertainty that underlies all machine learning algorithms.'
  //             },
  //             {
  //               title: 'Bayes\' Theorem and Applications',
  //               slug: 'bayes-theorem-applications',
  //               excerpt: 'Discover Bayes\' theorem, the mathematical heart of machine learning, and its powerful applications in spam detection, medical diagnosis, and AI systems.'
  //             },
  //             {
  //               title: 'Probability Distributions',
  //               slug: 'probability-distributions',
  //               excerpt: 'Explore discrete and continuous probability distributions that model real-world phenomena and power statistical methods in machine learning.'
  //             },
  //             {
  //               title: 'Central Limit Theorem',
  //               slug: 'central-limit-theorem',
  //               excerpt: 'Understand the Central Limit Theorem - the mathematical principle that explains why normal distributions appear everywhere and enables statistical inference.'
  //             }
  //           ]
  //         },
  //         {
  //           name: 'Statistics',
  //           slug: 'statistics',
  //           posts: [
  //             {
  //               title: 'Descriptive Statistics',
  //               slug: 'descriptive-statistics',
  //               excerpt: 'Master the fundamental tools for summarizing and describing data - the essential first step in any machine learning project and data analysis.'
  //             },
  //             {
  //               title: 'Hypothesis Testing',
  //               slug: 'hypothesis-testing',
  //               excerpt: 'Learn the statistical framework for making data-driven decisions, validating model improvements, and conducting rigorous A/B testing in machine learning.'
  //             },
  //             {
  //               title: 'Correlation vs. Causation',
  //               slug: 'correlation-causation',
  //               excerpt: 'Understand the crucial distinction between correlation and causation - a fundamental concept for avoiding misleading conclusions in data science and ML.'
  //             },
  //             {
  //               title: 'Statistical Inference',
  //               slug: 'statistical-inference',
  //               excerpt: 'Explore confidence intervals, hypothesis testing, and statistical power - the mathematical tools that enable us to draw reliable conclusions from data.'
  //             }
  //           ]
  //         }
  //       ]
  //     },
  //     'python': {
  //       name: 'Python',
  //       slug: 'python',
  //       subcategories: [
  //         {
  //           name: 'Machine Learning Libraries',
  //           slug: 'machine-learning-libraries',
  //           posts: [
  //             {
  //               title: 'Scikit-learn',
  //               slug: 'scikit-learn',
  //               excerpt: 'A comprehensive guide to scikit-learn, the most popular machine learning library in Python for building and evaluating ML models.'
  //             },
  //             {
  //               title: 'XGBoost',
  //               slug: 'xgboost',
  //               excerpt: 'Learn about XGBoost, the powerful gradient boosting framework that dominates machine learning competitions and real-world applications.'
  //             },
  //             {
  //               title: 'LightGBM',
  //               slug: 'lightgbm',
  //               excerpt: 'Explore LightGBM, Microsoft\'s fast and efficient gradient boosting framework designed for high performance and low memory usage.'
  //             },
  //             {
  //               title: 'CatBoost',
  //               slug: 'catboost',
  //               excerpt: 'Discover CatBoost, Yandex\'s gradient boosting library that excels at handling categorical features without preprocessing.'
  //             },
  //             {
  //               title: 'MLxtend',
  //               slug: 'mlxtend',
  //               excerpt: 'Introduction to MLxtend, a library of useful tools and extensions for everyday machine learning tasks and model evaluation.'
  //             }
  //           ]
  //         },
  //         {
  //           name: 'Deep Learning Libraries',
  //           slug: 'deep-learning-libraries',
  //           posts: [
  //             {
  //               title: 'TensorFlow',
  //               slug: 'tensorflow',
  //               excerpt: 'A complete guide to TensorFlow, Google\'s open-source platform for building and deploying machine learning models at scale.'
  //             },
  //             {
  //               title: 'Keras',
  //               slug: 'keras',
  //               excerpt: 'Learn Keras, the high-level neural networks API that makes deep learning accessible and easy to implement.'
  //             },
  //             {
  //               title: 'PyTorch',
  //               slug: 'torch',
  //               excerpt: 'Explore PyTorch, Facebook\'s dynamic deep learning framework favored by researchers for its flexibility and ease of use.'
  //             },
  //             {
  //               title: 'ONNX',
  //               slug: 'onnx',
  //               excerpt: 'Understanding ONNX, the open standard for representing machine learning models and enabling interoperability between frameworks.'
  //             }
  //           ]
  //         },
  //         {
  //           name: 'Data Manipulation & Computation Libraries',
  //           slug: 'data-manipulation-computation-libraries',
  //           posts: [
  //             {
  //               title: 'NumPy',
  //               slug: 'numpy',
  //               excerpt: 'Master NumPy, the fundamental package for scientific computing in Python, providing powerful N-dimensional array operations.'
  //             },
  //             {
  //               title: 'Pandas',
  //               slug: 'pandas',
  //               excerpt: 'Learn Pandas, the essential library for data manipulation and analysis, offering data structures and operations for numerical tables.'
  //             },
  //             {
  //               title: 'SciPy',
  //               slug: 'scipy',
  //               excerpt: 'Discover SciPy, the library that builds on NumPy to provide algorithms for optimization, linear algebra, and scientific computing.'
  //             },
  //             {
  //               title: 'Polars',
  //               slug: 'polars',
  //               excerpt: 'Introduction to Polars, the lightning-fast DataFrame library that offers better performance than Pandas for large datasets.'
  //             }
  //           ]
  //         }
  //       ]
  //     },
  //     'machine-learning-algorithms': {
  //       name: 'Machine Learning Algorithms',
  //       slug: 'machine-learning-algorithms',
  //       subcategories: [
  //         {
  //           name: 'Supervised Learning',
  //           slug: 'supervised-learning',
  //           posts: []
  //         },
  //         {
  //           name: 'Unsupervised Learning',
  //           slug: 'unsupervised-learning',
  //           posts: []
  //         },
  //         {
  //           name: 'Reinforcement Learning',
  //           slug: 'reinforcement-learning',
  //           posts: []
  //         }
  //       ]
  //     }
  //   };

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

    // Configure image renderer for proper path resolution and click handling
    renderer.image = function({ href, title, text }: { href: string; title: string | null; text: string }): string {
      // If the image path doesn't start with http/https, treat it as a local image
      if (!href.startsWith('http://') && !href.startsWith('https://')) {
        // Resolve relative image paths to the data/images directory
        href = `/data/images/${href}`;
      }
      
      const titleAttr = title ? ` title="${title}"` : '';
      const altAttr = text ? ` alt="${text}"` : '';
      
      return `<img src="${href}"${altAttr}${titleAttr} class="post-image clickable-image">`;
    };

    marked.setOptions({ renderer });
  }

  private loadAllPostsMetadata(): Observable<PostMetadata[]> {
    if (this.allPostsCache) {
      return of(this.allPostsCache);
    }

    const allPosts: PostMetadata[] = [];
    const orderedSlugs = new Set(this.studyOrderCategorySlugs);
    const categorySlugs = [
      ...this.studyOrderCategorySlugs,
      ...Object.keys(this.predefinedCategories).filter(s => !orderedSlugs.has(s))
    ];

    for (const categorySlug of categorySlugs) {
      const category = this.predefinedCategories[categorySlug];
      if (!category) {
        continue;
      }
      const subOrder = this.studyOrderSubcategorySlugs[categorySlug] ?? [];
      const subcategories = [...category.subcategories].sort((a, b) =>
        this.compareBySlugOrder(subOrder, a.slug, b.slug)
      );
      for (const subcategory of subcategories) {
        for (const post of subcategory.posts) {
          allPosts.push({
            ...post,
            category: category.name,
            subcategory: subcategory.name
          });
        }
      }
    }

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

    const subOrder = this.studyOrderSubcategorySlugs[categorySlug] ?? [];
    const subcategories = [...categoryData.subcategories].sort((a, b) =>
      this.compareBySlugOrder(subOrder, a.slug, b.slug)
    );

    const posts: PostMetadata[] = [];
    for (const subcategory of subcategories) {
      for (const post of subcategory.posts) {
        posts.push({
          ...post,
          category: categoryData.name,
          subcategory: subcategory.name
        });
      }
    }

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
    const orderedSlugs = new Set(this.studyOrderCategorySlugs);
    const names: string[] = [];
    for (const slug of this.studyOrderCategorySlugs) {
      const cat = this.predefinedCategories[slug];
      if (cat) {
        names.push(cat.name);
      }
    }
    for (const slug of Object.keys(this.predefinedCategories)) {
      if (!orderedSlugs.has(slug)) {
        names.push(this.predefinedCategories[slug].name);
      }
    }
    return of(names);
  }

  getCategoryTree(): Observable<CategoryTree[]> {
    const result: CategoryTree[] = [];
    const orderedSlugs = new Set(this.studyOrderCategorySlugs);

    const pushCategoryTree = (categorySlug: string): void => {
      const category = this.predefinedCategories[categorySlug];
      if (!category) {
        return;
      }

      const subOrder = this.studyOrderSubcategorySlugs[categorySlug] ?? [];
      const sortedSubcategories = [...category.subcategories].sort((a, b) =>
        this.compareBySlugOrder(subOrder, a.slug, b.slug)
      );

      const categoryTree: CategoryTree = {
        name: category.name,
        slug: category.slug,
        count: 0,
        subcategories: []
      };

      for (const subcategory of sortedSubcategories) {
        const subCategoryData: SubCategory = {
          name: subcategory.name,
          slug: subcategory.slug,
          count: subcategory.posts.length,
          posts: subcategory.posts
        };

        categoryTree.subcategories.push(subCategoryData);
        categoryTree.count += subcategory.posts.length;
      }

      result.push(categoryTree);
    };

    for (const slug of this.studyOrderCategorySlugs) {
      pushCategoryTree(slug);
    }
    for (const slug of Object.keys(this.predefinedCategories)) {
      if (!orderedSlugs.has(slug)) {
        pushCategoryTree(slug);
      }
    }

    return of(result);
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