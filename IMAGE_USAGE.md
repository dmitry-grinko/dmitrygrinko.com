# Image Support in Markdown Posts

## ✅ Image Support is Now Implemented!

Your blog now supports images in markdown posts with automatic path resolution and responsive styling.

## How to Use Images

### 1. **Local Images** (Recommended)
Place your images in `src/data/images/` directory and reference them in markdown:

```markdown
![Alt text](image-name.jpg)
![Alt text](image-name.jpg "Optional title")
```

**Example:**
```markdown
![Angular Architecture](angular-architecture.png)
![Angular Component Lifecycle](component-lifecycle.jpg "Component Lifecycle Diagram")
```

### 2. **External Images** (URLs)
You can also use external image URLs:

```markdown
![External Image](https://example.com/image.jpg)
```

## Image Features

### ✅ **Automatic Path Resolution**
- Local images: `image.jpg` → `/data/images/image.jpg`
- External URLs: Work as-is

### ✅ **Responsive Design**
- **Desktop**: Max-width 100%, centered with shadow
- **Mobile**: Adjusted margins and border radius
- **Auto-scaling**: Images never exceed container width

### ✅ **Beautiful Styling**
- Rounded corners (8px on desktop, 6px on mobile)
- Subtle shadow for depth
- Centered alignment
- Proper spacing (1.5rem on desktop, 1rem on mobile)

## File Structure

```
src/
├── data/
│   ├── images/          ← Put your images here
│   │   ├── example.jpg
│   │   ├── diagram.png
│   │   └── screenshot.webp
│   └── posts/
│       ├── post1.md     ← Reference images from here
│       └── post2.md
```

## Supported Image Formats

- **JPG/JPEG** - Photos and complex images
- **PNG** - Images with transparency
- **WebP** - Modern format for better compression
- **SVG** - Vector graphics and icons
- **GIF** - Animated images

## Example Markdown Post

```markdown
---
title: "My Angular Tutorial"
category: "development"
subcategory: "angular"
date: "2024-01-25"
slug: "angular-tutorial"
excerpt: "A comprehensive Angular tutorial with diagrams."
---

# Angular Component Architecture

Here's how Angular components work:

![Angular Component Architecture](angular-architecture.png)

The component lifecycle includes several hooks:

![Component Lifecycle](lifecycle-diagram.jpg "Angular Component Lifecycle")

You can also use external images:

![Angular Logo](https://angular.io/assets/images/logos/angular/angular.png)
```

## CSS Classes Applied

All images in posts automatically get the `post-image` class with:

- `max-width: 100%` - Responsive sizing
- `height: auto` - Maintain aspect ratio  
- `border-radius: 8px` - Rounded corners
- `margin: 1.5rem 0` - Vertical spacing
- `box-shadow` - Subtle depth effect
- `margin: auto` - Center alignment

## Ready to Use!

Your image support is now fully implemented and ready to use. Just add images to `src/data/images/` and reference them in your markdown posts! 