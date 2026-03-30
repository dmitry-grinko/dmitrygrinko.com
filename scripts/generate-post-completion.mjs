#!/usr/bin/env node
/**
 * Scans src/data/posts (recursive) for .md files and writes public/post-completion.json.
 * A post is "written" when the markdown file has more than MIN_LINES lines.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const POSTS_DIR = path.join(ROOT, 'src', 'data', 'posts');
const OUT_FILE = path.join(ROOT, 'public', 'post-completion.json');
const MIN_LINES = 5;

function walkMarkdownFiles(dir, acc = []) {
  if (!fs.existsSync(dir)) {
    return acc;
  }
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      walkMarkdownFiles(p, acc);
    } else if (ent.name.endsWith('.md')) {
      acc.push(p);
    }
  }
  return acc;
}

function main() {
  const files = walkMarkdownFiles(POSTS_DIR);
  const posts = {};

  for (const filePath of files) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lineCount = content.split(/\r?\n/).length;
    const slug = path.basename(filePath, '.md');
    posts[slug] = {
      lineCount,
      written: lineCount > MIN_LINES
    };
  }

  const payload = {
    minLines: MIN_LINES,
    generatedAt: new Date().toISOString(),
    posts
  };

  fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
  fs.writeFileSync(OUT_FILE, JSON.stringify(payload, null, 2) + '\n', 'utf8');
  console.log(`Wrote ${Object.keys(posts).length} entries to ${path.relative(ROOT, OUT_FILE)}`);
}

main();
