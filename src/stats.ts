import { Repository } from './types';

export interface CategoryStats {
  name: string;
  count: number;
  totalStars: number;
  avgStars: number;
  languages: Map<string, number>;
  recentUpdates: number; // repos updated in last 30 days
  mostPopular: Repository[];
  trending: Repository[]; // repos with most recent star growth
}

export interface TrendingMetrics {
  starGrowthRate: number;
  updatedAt: string;
  isActive: boolean;
}

export class StatsAnalyzer {
  private readonly RECENT_DAYS = 30;
  private readonly TRENDING_SAMPLE_SIZE = 5;

  constructor(private repos: Repository[]) {}

  public generateCategoryStats(category: string, repos: Repository[]): CategoryStats {
    const languages = new Map<string, number>();
    let totalStars = 0;
    const now = new Date();

    // Calculate recent updates
    const recentUpdates = repos.filter(repo => {
      const updatedAt = new Date(repo.updated_at);
      const daysDiff = (now.getTime() - updatedAt.getTime()) / (1000 * 60 * 60 * 24);
      return daysDiff <= this.RECENT_DAYS;
    }).length;

    // Aggregate language stats
    repos.forEach(repo => {
      if (repo.language) {
        languages.set(
          repo.language,
          (languages.get(repo.language) || 0) + 1
        );
      }
      totalStars += repo.stargazers_count;
    });

    // Sort repos by stars for most popular
    const mostPopular = [...repos]
      .sort((a, b) => b.stargazers_count - a.stargazers_count)
      .slice(0, this.TRENDING_SAMPLE_SIZE);

    // Calculate trending repos
    const trending = this.getTrendingRepos(repos);

    return {
      name: category,
      count: repos.length,
      totalStars,
      avgStars: totalStars / repos.length,
      languages,
      recentUpdates,
      mostPopular,
      trending
    };
  }

  private getTrendingRepos(repos: Repository[]): Repository[] {
    // In a real implementation, we'd track star history over time
    // For now, we'll use update frequency and recent stars as a proxy
    return [...repos]
      .sort((a, b) => {
        const scoreA = this.calculateTrendingScore(a);
        const scoreB = this.calculateTrendingScore(b);
        return scoreB - scoreA;
      })
      .slice(0, this.TRENDING_SAMPLE_SIZE);
  }

  private calculateTrendingScore(repo: Repository): number {
    const updatedAt = new Date(repo.updated_at).getTime();
    const now = new Date().getTime();
    const daysSinceUpdate = (now - updatedAt) / (1000 * 60 * 60 * 24);
    
    // Score formula: stars / (days since update)^2
    // This favors recently updated repos with more stars
    return repo.stargazers_count / Math.pow(daysSinceUpdate + 1, 2);
  }

  public generateOverallStats(): Record<string, any> {
    const totalRepos = this.repos.length;
    const totalStars = this.repos.reduce((sum, repo) => sum + repo.stargazers_count, 0);
    const languages = new Map<string, number>();
    const topics = new Map<string, number>();

    // Aggregate language and topic stats
    this.repos.forEach(repo => {
      if (repo.language) {
        languages.set(repo.language, (languages.get(repo.language) || 0) + 1);
      }
      repo.topics.forEach(topic => {
        topics.set(topic, (topics.get(topic) || 0) + 1);
      });
    });

    // Sort languages and topics by frequency
    const topLanguages = Array.from(languages.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10);

    const topTopics = Array.from(topics.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10);

    // Calculate active repos (updated in last 30 days)
    const activeRepos = this.repos.filter(repo => {
      const updatedAt = new Date(repo.updated_at);
      const now = new Date();
      const daysDiff = (now.getTime() - updatedAt.getTime()) / (1000 * 60 * 60 * 24);
      return daysDiff <= 30;
    });

    return {
      totalRepos,
      totalStars,
      avgStarsPerRepo: totalStars / totalRepos,
      activeReposCount: activeRepos.length,
      activeReposPercentage: (activeRepos.length / totalRepos) * 100,
      topLanguages,
      topTopics,
      mostStarred: this.repos
        .sort((a, b) => b.stargazers_count - a.stargazers_count)
        .slice(0, 5),
      trending: this.getTrendingRepos(this.repos),
    };
  }

  public formatMarkdown(stats: CategoryStats): string {
    let markdown = `### Category: ${stats.name}\n\n`;
    
    markdown += `#### Quick Stats\n`;
    markdown += `- Total Repositories: ${stats.count}\n`;
    markdown += `- Total Stars: ${stats.totalStars.toLocaleString()}\n`;
    markdown += `- Average Stars: ${stats.avgStars.toFixed(1)}\n`;
    markdown += `- Recently Updated: ${stats.recentUpdates} repos in last ${this.RECENT_DAYS} days\n\n`;

    markdown += `#### Top Languages\n`;
    const sortedLanguages = Array.from(stats.languages.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5);
    
    sortedLanguages.forEach(([lang, count]) => {
      markdown += `- ${lang}: ${count} repos\n`;
    });
    markdown += '\n';

    markdown += `#### Most Popular\n`;
    stats.mostPopular.forEach(repo => {
      markdown += `- [${repo.name}](${repo.html_url}) - ⭐ ${repo.stargazers_count.toLocaleString()}\n`;
    });
    markdown += '\n';

    markdown += `#### Trending\n`;
    stats.trending.forEach(repo => {
      markdown += `- [${repo.name}](${repo.html_url}) - Last updated ${new Date(repo.updated_at).toLocaleDateString()}\n`;
    });
    markdown += '\n';

    return markdown;
  }

  public formatOverallStats(stats: Record<string, any>): string {
    let markdown = `## Repository Statistics\n\n`;
    
    markdown += `### Overall Stats\n`;
    markdown += `- Total Repositories: ${stats.totalRepos}\n`;
    markdown += `- Total Stars: ${stats.totalStars.toLocaleString()}\n`;
    markdown += `- Average Stars per Repo: ${stats.avgStarsPerRepo.toFixed(1)}\n`;
    markdown += `- Active Repos: ${stats.activeReposCount} (${stats.activeReposPercentage.toFixed(1)}% updated in last 30 days)\n\n`;

    markdown += `### Top Languages\n`;
    stats.topLanguages.forEach(([lang, count]) => {
      markdown += `- ${lang}: ${count} repos\n`;
    });
    markdown += '\n';

    markdown += `### Top Topics\n`;
    stats.topTopics.forEach(([topic, count]) => {
      markdown += `- #${topic}: ${count} repos\n`;
    });
    markdown += '\n';

    markdown += `### Most Starred Repositories\n`;
    stats.mostStarred.forEach(repo => {
      markdown += `- [${repo.name}](${repo.html_url}) - ⭐ ${repo.stargazers_count.toLocaleString()}\n`;
    });
    markdown += '\n';

    markdown += `### Trending Repositories\n`;
    stats.trending.forEach(repo => {
      markdown += `- [${repo.name}](${repo.html_url}) - Last updated ${new Date(repo.updated_at).toLocaleDateString()}\n`;
    });
    markdown += '\n';

    return markdown;
  }
}
