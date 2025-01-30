import { Octokit } from '@octokit/rest';
import { Repository, CategoryConfig } from './src/types';
import { StatsAnalyzer } from './src/stats';

// Configuration
const categories: CategoryConfig[] = [
  {
    name: 'AI & Machine Learning',
    keywords: ['ai', 'ml', 'machine-learning', 'deep-learning', 'neural', 'nlp', 'tensorflow', 'pytorch', 'artificial-intelligence'],
    tags: ['#ai', '#machine-learning', '#nlp', '#neural-networks', '#automation', '#data-science'],
    description: 'Artificial Intelligence, Machine Learning, and intelligent automation tools'
  },
  // ... (previous categories remain the same)
];

class StarOrganizer {
  private octokit: Octokit;
  private username: string;
  private statsAnalyzer: StatsAnalyzer | null = null;

  constructor(token: string, username: string) {
    this.octokit = new Octokit({ auth: token });
    this.username = username;
  }

  private categorizeRepository(repo: Repository): string[] {
    const description = (repo.description || '').toLowerCase();
    const name = repo.name.toLowerCase();
    const topics = repo.topics.map(t => t.toLowerCase());
    const matchedCategories: string[] = [];

    for (const category of categories) {
      const hasKeyword = category.keywords.some(keyword =>
        description.includes(keyword) ||
        name.includes(keyword) ||
        topics.includes(keyword)
      );
      
      if (hasKeyword) {
        matchedCategories.push(category.name);
      }
    }

    return matchedCategories.length > 0 ? matchedCategories : ['Uncategorized'];
  }

  private getRelevantTags(repo: Repository): string[] {
    const allTags = new Set<string>();
    const description = (repo.description || '').toLowerCase();
    const name = repo.name.toLowerCase();
    const topics = repo.topics.map(t => t.toLowerCase());

    // Add language tag if available
    if (repo.language) {
      allTags.add(`#${repo.language.toLowerCase()}`);
    }

    // Add repository topics as tags
    topics.forEach(topic => allTags.add(`#${topic}`));

    // Add category-specific tags
    categories.forEach(category => {
      category.keywords.forEach(keyword => {
        if (
          description.includes(keyword) ||
          name.includes(keyword) ||
          topics.includes(keyword)
        ) {
          category.tags.forEach(tag => allTags.add(tag));
        }
      });
    });

    return Array.from(allTags).sort();
  }

  async generateStarredContent(): Promise<string> {
    const stars = await this.fetchStarredRepos();
    this.statsAnalyzer = new StatsAnalyzer(stars);
    
    const categorizedStars = new Map<string, Repository[]>();

    // Categorize repositories
    stars.forEach(repo => {
      const categories = this.categorizeRepository(repo);
      categories.forEach(category => {
        if (!categorizedStars.has(category)) {
          categorizedStars.set(category, []);
        }
        categorizedStars.get(category)?.push(repo);
      });
    });

    // Generate overall statistics
    const overallStats = this.statsAnalyzer.generateOverallStats();
    let content = this.statsAnalyzer.formatOverallStats(overallStats);

    content += `\n## Repositories by Category\n\n`;

    // Process each category
    for (const category of categories) {
      const repos = categorizedStars.get(category.name) || [];
      if (repos.length > 0) {
        const categoryStats = this.statsAnalyzer.generateCategoryStats(category.name, repos);
        content += this.statsAnalyzer.formatMarkdown(categoryStats);
      }
    }

    // Process uncategorized repositories
    const uncategorized = categorizedStars.get('Uncategorized') || [];
    if (uncategorized.length > 0) {
      const uncategorizedStats = this.statsAnalyzer.generateCategoryStats('Uncategorized', uncategorized);
      content += this.statsAnalyzer.formatMarkdown(uncategorizedStats);
    }

    return content;
  }

  private async fetchStarredRepos(): Promise<Repository[]> {
    const stars: Repository[] = [];
    let page = 1;
    const per_page = 100;

    while (true) {
      const response = await this.octokit.activity.listReposStarredByUser({
        username: this.username,
        per_page,
        page,
      });

      if (response.data.length === 0) break;
      stars.push(...response.data as Repository[]);
      if (response.data.length < per_page) break;
      page++;
    }

    return stars;
  }

  async updateStarred() {
    const content = await this.generateStarredContent();
    
    // Update STARRED.md
    await this.octokit.repos.createOrUpdateFileContents({
      owner: this.username,
      repo: 'github-stars-organize',
      path: 'STARRED.md',
      message: 'Update starred repositories with statistics and analysis [skip ci]',
      content: Buffer.from(content).toString('base64'),
      sha: await this.getCurrentStarredSha(),
    });
  }

  private async getCurrentStarredSha(): Promise<string> {
    try {
      const { data } = await this.octokit.repos.getContent({
        owner: this.username,
        repo: 'github-stars-organize',
        path: 'STARRED.md',
      });

      return (data as any).sha;
    } catch (error) {
      return '';
    }
  }
}

// Usage example
async function main() {
  const token = process.env.GITHUB_TOKEN;
  const username = process.env.GITHUB_USERNAME;

  if (!token || !username) {
    console.error('Please set GITHUB_TOKEN and GITHUB_USERNAME environment variables');
    process.exit(1);
  }

  const organizer = new StarOrganizer(token, username);
  await organizer.updateStarred();
  console.log('STARRED.md updated successfully with statistics!');
}

if (require.main === module) {
  main().catch(console.error);
}

export { StarOrganizer, categories };