# GitHub Stars Organizer

An automated tool to categorize, analyze, and organize your GitHub starred repositories into a searchable, well-structured format with detailed statistics and trending analysis.

## Features

- 📁 Automatically categorizes starred repositories 
- 📊 Generates comprehensive statistics and insights
- 📈 Identifies trending repositories in your stars
- 🏷️ Smart tag generation based on repository content
- 📑 Language and topic analysis
- 🔄 Daily updates via GitHub Actions

## Statistics & Analysis

The tool provides detailed insights about your starred repositories:

### Overall Statistics
- Total repositories and stars count
- Active repositories percentage
- Top languages and topics
- Most starred repositories
- Recent trending repositories

### Per-Category Analysis
- Repository count and total stars
- Average stars per repository
- Language distribution
- Recently updated repositories
- Most popular repositories
- Trending repositories within category

### Trending Analysis
- Star growth rate calculation
- Activity monitoring
- Popular repositories identification
- Recently updated project tracking

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
   - Modify keywords and tags in `src/types.ts`
   - Adjust statistics settings in `src/stats.ts`

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
  tags: ['#tag1', '#tag2'],
  description: 'Category description'
}
```

### Modifying Statistics
Edit `src/stats.ts` to:
- Adjust trending calculation algorithm
- Modify statistics formatting
- Add new metrics
- Customize analysis parameters

### Custom Tag Generation
Customize the `getRelevantTags` method in `update-stars.ts` to add your own tag generation logic.

## File Structure

- `STARRED.md` - Generated list of organized stars with statistics
- `src/stats.ts` - Statistics and trending analysis module
- `src/types.ts` - TypeScript type definitions
- `update-stars.ts` - Main script for fetching and organizing stars
- `.github/workflows/update-stars.yml` - GitHub Actions workflow
- `README.md` - Project documentation (this file)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Some areas for contribution:

1. Enhanced statistics and trending algorithms
2. Additional metrics and insights
3. Improved categorization logic
4. Better visualization of statistics
5. New features and enhancements

## License

MIT

---

✨ **Want to see it in action?** Check out [STARRED.md](STARRED.md) for my organized GitHub stars with complete statistics and trending analysis!