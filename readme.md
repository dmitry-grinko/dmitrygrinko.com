# Dmitry Grinko - Personal Landing Page

A modern, responsive landing page built with Angular and deployed via GitHub Pages.

## Project Structure

```
dmitrygrinko_com/
├── src/                 # Angular application source
│   ├── src/
│   │   ├── app/        # Angular components
│   │   ├── index.html  # Main HTML file
│   │   └── styles.css  # Global styles
│   ├── angular.json    # Angular configuration
│   └── package.json    # Dependencies
├── docs/               # GitHub Pages deployment folder (auto-generated)
├── .github/
│   └── workflows/
│       └── deploy.yml  # GitHub Actions deployment
└── README.md
```

## Features

- ✨ Modern, responsive design with smooth animations
- 🎨 Beautiful gradient background with glassmorphism effects
- 📱 Mobile-first responsive layout
- 🚀 Optimized for performance
- 🔗 Social media links (GitHub, LinkedIn, Email)
- 📦 Automated deployment via GitHub Actions

## Development

To run the development server:

```bash
cd src
npm install
npm start
```

The app will be available at `http://localhost:4200/`.

## Deployment

The app is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the main branch. The built files are placed in the `docs/` folder which is configured as the GitHub Pages source.

### Manual Build

To build manually:

```bash
cd src
npm run build
```

This will build the app and place the output in the `docs/` folder.

## GitHub Pages Setup

### 1. Configure GitHub Secrets

Go to your repository settings → Secrets and variables → Actions, and add these secrets:

- `GIT_USER_EMAIL`: Your GitHub email address
- `GIT_USER_NAME`: Your GitHub username or display name

### 2. Enable GitHub Pages

1. Go to your repository settings
2. Navigate to "Pages" section
3. Set source to "Deploy from a branch"
4. Select "main" branch and "docs" folder
5. Your site will be available at `https://yourusername.github.io/repositoryname/`

## Customization

To customize the landing page:

1. Edit `src/src/app/app.html` for content
2. Modify `src/src/app/app.css` for styling
3. Update social media links in the HTML
4. Change the title and meta tags in `src/src/index.html`

## Technologies Used

- Angular 19
- TypeScript
- CSS3 with modern features (Grid, Flexbox, CSS Variables)
- GitHub Actions for CI/CD
- GitHub Pages for hosting 