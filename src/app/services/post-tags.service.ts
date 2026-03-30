import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

const STORAGE_KEY = 'dmitrygrinko-post-tags-v1';

interface TagsStore {
  version: 1;
  byPostSlug: Record<string, string[]>;
}

@Injectable({
  providedIn: 'root'
})
export class PostTagsService {
  private readonly store$ = new BehaviorSubject<TagsStore>(this.load());

  readonly tagsChanged$ = this.store$.asObservable();

  getTagsForPost(slug: string): string[] {
    const tags = this.store$.value.byPostSlug[slug];
    return tags ? [...tags] : [];
  }

  setTagsForPost(slug: string, tags: string[]): void {
    const normalized = this.normalizeTagList(tags);
    const next: TagsStore = {
      version: 1,
      byPostSlug: { ...this.store$.value.byPostSlug }
    };
    if (normalized.length === 0) {
      delete next.byPostSlug[slug];
    } else {
      next.byPostSlug[slug] = normalized;
    }
    this.persist(next);
  }

  /** Sorted unique tag labels (as stored). */
  getAllTags(): string[] {
    const set = new Set<string>();
    for (const tags of Object.values(this.store$.value.byPostSlug)) {
      for (const t of tags) {
        set.add(t);
      }
    }
    return [...set].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
  }

  hasAnyTag(): boolean {
    return this.getAllTags().length > 0;
  }

  /** Remove all tags for all posts (clears localStorage entry). */
  clearAllTags(): void {
    this.persist({ version: 1, byPostSlug: {} });
  }

  getSlugsForTag(tag: string): string[] {
    const needle = tag.trim().toLowerCase();
    if (!needle) {
      return [];
    }
    const slugs: string[] = [];
    for (const [slug, tags] of Object.entries(this.store$.value.byPostSlug)) {
      if (tags.some(t => t.trim().toLowerCase() === needle)) {
        slugs.push(slug);
      }
    }
    return slugs;
  }

  private load(): TagsStore {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return { version: 1, byPostSlug: {} };
      }
      const parsed = JSON.parse(raw) as TagsStore;
      if (parsed?.version !== 1 || typeof parsed.byPostSlug !== 'object' || !parsed.byPostSlug) {
        return { version: 1, byPostSlug: {} };
      }
      const cleaned: Record<string, string[]> = {};
      for (const [slug, tags] of Object.entries(parsed.byPostSlug)) {
        if (typeof slug === 'string' && Array.isArray(tags)) {
          const list = this.normalizeTagList(tags.filter((x): x is string => typeof x === 'string'));
          if (list.length > 0) {
            cleaned[slug] = list;
          }
        }
      }
      return { version: 1, byPostSlug: cleaned };
    } catch {
      return { version: 1, byPostSlug: {} };
    }
  }

  private persist(store: TagsStore): void {
    this.store$.next(store);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
    } catch {
      // quota or private mode — in-memory state still updates for session
    }
  }

  private normalizeTagList(tags: string[]): string[] {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const t of tags) {
      const n = t.trim().replace(/\s+/g, ' ');
      if (!n) {
        continue;
      }
      const key = n.toLowerCase();
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      out.push(n);
    }
    return out;
  }
}
