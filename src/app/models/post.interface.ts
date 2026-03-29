export interface Post {
  title: string;
  category: string;
  subcategory?: string;
  /** Canonical URL segment; set when category display name is not trivially slugified */
  categorySlug?: string;
  subcategorySlug?: string;
  date?: string;
  slug: string;
  excerpt: string;
  content?: string;
}

export interface PostMetadata {
  title: string;
  slug: string;
  excerpt: string;
  category?: string;
  subcategory?: string;
  categorySlug?: string;
  subcategorySlug?: string;
}

export interface CategoryTree {
  name: string;
  slug: string;
  count: number;
  subcategories: SubCategory[];
}

export interface SubCategory {
  name: string;
  slug: string;
  count: number;
  posts?: PostMetadata[];
}

export interface PredefinedSubcategory {
  name: string;
  slug: string;
  posts: PostMetadata[];
}

export interface PredefinedCategory {
  name: string;
  slug: string;
  subcategories: PredefinedSubcategory[];
}

export interface PredefinedCategories {
  [categorySlug: string]: PredefinedCategory;
} 