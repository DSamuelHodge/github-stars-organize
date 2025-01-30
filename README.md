# GitHub Stars Organizer

An automated tool to categorize and organize your GitHub starred repositories into a searchable, well-structured format.

## Features

- 📁 Automatically categorizes starred repositories 
- 🏷️ Generates relevant tags based on repository content
- 📊 Provides useful metadata (stars, last updated, etc.)
- 🔄 Updates daily via GitHub Actions
- 🔍 Smart keyword-based categorization
- 🎯 Customizable categories and tags

## How It Works

The tool automatically:
1. Fetches your starred repositories
2. Analyzes repository names, descriptions, and topics
3. Categorizes them based on predefined keywords
4. Generates relevant tags
5. Creates an organized markdown file (STARRED.md)

## Setup Guide

### Prerequisites
- GitHub account
- Basic knowledge of GitHub Actions
- Node.js installed (for local development)

### Quick Start
1. Fork this repository
2. Enable GitHub Actions in your forked repository:
   - Go to Settings > Actions > General
   - Select "Allow all actions and reusable workflows"
   - Scroll down to "Workflow permissions"
   - Select "Read and write permissions"
   - Click "Save"

3. Update the configuration (optional):
   - Edit `categories` in `update-stars.ts` to customize categories
   - Modify keywords and tags to match your interests
   - Adjust the formatting in the `formatRepository` method

4. Star some repositories and wait for the daily update, or:
   - Go to the Actions tab
   - Select "Update Stars"
   - Click "Run workflow"

### Local Development

```bash
# Clone your forked repository
git clone https://github.com/YOUR_USERNAME/github-stars-organize.git

# Install dependencies
npm install

# Build the TypeScript code
npm run build

# Run the update script (requires GitHub token)
npm start
```

### Environment Variables
When running locally, you need to set:
- `GITHUB_TOKEN`: Your GitHub Personal Access Token
- `GITHUB_USERNAME`: Your GitHub username

## Customization

### Adding New Categories
Edit the `categories` array in `update-stars.ts`:

```typescript
{
  name: 'Your Category',
  keywords: ['keyword1', 'keyword2'],
  tags: ['#tag1', '#tag2']
}
```

### Modifying Tag Generation
Customize the `getRelevantTags` method in `update-stars.ts` to add your own tag generation logic.

## File Structure

- `STARRED.md` - Generated list of organized stars
- `update-stars.ts` - Main script for fetching and organizing stars
- `.github/workflows/update-stars.yml` - GitHub Actions workflow
- `README.md` - Project documentation (this file)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT

---

✨ **Want to see it in action?** Check out [STARRED.md](STARRED.md) for my organized GitHub stars!