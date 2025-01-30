import { Octokit } from '@octokit/rest';
import { marked } from 'marked';

// Types
interface Repository {
  name: string;
  full_name: string;
  html_url: string;
  description: string;
  stargazers_count: number;
  language: string;
  topics: string[];
  pushed_at: string;
  updated_at: string;
}

interface CategoryConfig {
  name: string;
  keywords: string[];
  tags: string[];
}

// Configuration
const categories: CategoryConfig[] = [
  {
    name: 'AI & Machine Learning',
    keywords: ['ai', 'ml', 'machine-learning', 'deep-learning', 'neural', 'nlp', 'tensorflow', 'pytorch'],
    tags: ['#ai', '#machine-learning', '#nlp', '#neural-networks', '#automation', '#data-science']
  },
  {
    name: 'Full-Stack Applications',
    keywords: ['fullstack', 'full-stack', 'webapp', 'application', 'saas'],
    tags: ['#fullstack', '#web-app', '#production', '#enterprise', '#saas', '#microservices']
  },
  {
    name: 'Frontend Frameworks & Libraries',
    keywords: ['react', 'vue', 'angular', 'svelte', 'ui', 'frontend', 'component'],
    tags: ['#react', '#vue', '#angular', '#svelte', '#ui-components', '#design-systems', '#tailwind']
  },
  {
    name: 'Backend & API Development',
    keywords: ['api', 'backend', 'server', 'graphql', 'rest', 'serverless'],
    tags: ['#nodejs', '#express', '#graphql', '#rest-api', '#serverless', '#microservices', '#database']
  },
  {
    name: 'Developer Tools & Utilities',
    keywords: ['cli', 'tool', 'utility', 'debug', 'test', 'productivity'],
    tags: ['#cli', '#developer-tools', '#productivity', '#debugging', '#testing', '#deployment']
  },
  {
    name: 'DevOps & Infrastructure',
    keywords: ['devops', 'docker', 'kubernetes', 'aws', 'azure', 'ci', 'cd'],
    tags: ['#devops', '#docker', '#kubernetes', '#aws', '#azure', '#cicd', '#monitoring']
  },
  {
    name: 'Security & Authentication',
    keywords: ['security', 'auth', 'authentication', 'oauth', 'jwt', 'crypto'],
    tags: ['#security', '#auth', '#encryption', '#oauth', '#jwt', '#cybersecurity']
  },
  {
    name: 'Performance & Optimization',
    keywords: ['performance', 'optimize', 'optimization', 'cache', 'bundle'],
    tags: ['#performance', '#optimization', '#caching', '#bundling', '#lazy-loading']
  },
  {
    name: 'Data Management & Analytics',
    keywords: ['data', 'database', 'analytics', 'visualization', 'sql', 'nosql'],
    tags: ['#database', '#analytics', '#visualization', '#big-data', '#sql', '#nosql']
  },
  {
    name: 'Learning Resources & Boilerplates',
    keywords: ['tutorial', 'learn', 'example', 'boilerplate', 'starter', 'template'],
    tags: ['#tutorial', '#boilerplate', '#starter-kit', '#example', '#learning', '#documentation']
  }
];

class StarOrganizer {
  private octokit: Octokit;
  private username: string;

  constructor(token: string, username: string) {
    this.octokit = new Octokit({ auth: token });
    this.username = username;
  }

  private categorizeRepository(repo: Repository): string {
    const description = (repo.description || '').toLowerCase();
    const name = repo.name.toLowerCase();
    const topics = repo.topics.map(t => t.toLowerCase());

    for (const category of categories) {
      const hasKeyword = category.keywords.some(keyword =>
        description.includes(keyword) ||
        name.includes(keyword) ||
        topics.includes(keyword)
      );
      
      if (hasKeyword) {
        return category.name;
      }
    }

    return 'Uncategorized';
  }

  private getRelevantTags(repo: Repository): string[] {
    const allTags = new Set<string>();
    const description = (repo.description || '').toLowerCase();
    const name = repo.name.toLowerCase();

    categories.forEach(category => {
      category.keywords.forEach(keyword => {
        if (
          description.includes(keyword) ||
          name.includes(keyword) ||
          repo.topics.includes(keyword)
        ) {
          category.tags.forEach(tag => allTags.add(tag));
        }
      });
    });

    // Add language tag if available
    if (repo.language) {
      allTags.add(`#${repo.language.toLowerCase()}`);
    }

    return Array.from(allTags);
  }

  private formatRepository(repo: Repository): string {
    const tags = this.getRelevantTags(repo);
    const date = new Date(repo.pushed_at).toISOString().split('T')[0];
    
    return `- [${repo.name}](${repo.html_url}) - ${repo.description || 'No description available'}
  - **Tags**: ${tags.join(' ')}
  - **Stars**: ${repo.stargazers_count}
  - **Language**: ${repo.language || 'Not specified'}
  - **Last Updated**: ${date}`;
  }

  async generateReadme(): Promise<string> {
    const stars = await this.fetchStarredRepos();
    const categorizedStars = new Map<string, Repository[]>();

    // Categorize repositories
    stars.forEach(repo => {
      const category = this.categorizeRepository(repo);
      if (!categorizedStars.has(category)) {
        categorizedStars.set(category, []);
      }
      categorizedStars.get(category)?.push(repo);
    });

    // Generate README content
    let content = `# My GitHub Stars\n\n`;
    content += `*Last updated: ${new Date().toISOString().split('T')[0]}*\n\n`;
    content += `## Table of Contents\n`;

    categories.forEach(category => {
      const slug = category.name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
      content += `- [${category.name}](#${slug})\n`;
    });

    content += `\n`;

    // Add each category and its repositories
    categories.forEach(category => {
      const repos = categorizedStars.get(category.name) || [];
      content += `## ${category.name}\n\n`;
      
      if (repos.length === 0) {
        content += `*No repositories in this category yet*\n\n`;
      } else {
        repos.forEach(repo => {
          content += `${this.formatRepository(repo)}\n\n`;
        });
      }
    });

    // Add uncategorized repositories if any exist
    const uncategorized = categorizedStars.get('Uncategorized');
    if (uncategorized && uncategorized.length > 0) {
      content += `## Uncategorized\n\n`;
      uncategorized.forEach(repo => {
        content += `${this.formatRepository(repo)}\n\n`;
      });
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

  async updateReadme() {
    const content = await this.generateReadme();
    
    // Update the README.md in the repository
    await this.octokit.repos.createOrUpdateFileContents({
      owner: this.username,
      repo: 'github-stars-organize',
      path: 'README.md',
      message: 'Update README with latest starred repositories',
      content: Buffer.from(content).toString('base64'),
      sha: await this.getCurrentReadmeSha(),
    });
  }

  private async getCurrentReadmeSha(): Promise<string> {
    try {
      const { data } = await this.octokit.repos.getContent({
        owner: this.username,
        repo: 'github-stars-organize',
        path: 'README.md',
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
  await organizer.updateReadme();
  console.log('README updated successfully!');
}

if (require.main === module) {
  main().catch(console.error);
}

export { StarOrganizer, categories };