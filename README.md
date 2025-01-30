# GitHub Stars Organization

This repository automatically organizes and updates my GitHub starred repositories into categories for easy reference.

## Features

- 📁 Automatically categorizes starred repositories 
- 🏷️ Generates relevant tags based on repository content
- 📊 Provides useful metadata (stars, last updated, etc.)
- 🔄 Updates daily via GitHub Actions

## Categories

1. AI & Machine Learning
2. Full-Stack Applications
3. Frontend Frameworks & Libraries
4. Backend & API Development
5. Developer Tools & Utilities
6. DevOps & Infrastructure
7. Security & Authentication
8. Performance & Optimization
9. Data Management & Analytics
10. Learning Resources & Boilerplates

## Setup

1. Star this repository to enable automatic updates
2. Wait for the GitHub Action to run (or trigger it manually)
3. Your stars will be automatically categorized and updated daily

## Manual Update

To manually trigger an update:

1. Go to the Actions tab
2. Select the "Update Stars" workflow
3. Click "Run workflow"

## Local Development

```bash
# Install dependencies
npm install

# Build the TypeScript code
npm run build

# Run the update script
npm start
```

Make sure to set your `GITHUB_TOKEN` and `GITHUB_USERNAME` environment variables.

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT