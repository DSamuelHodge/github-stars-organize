import { Repository, CategoryConfig, Stats } from './types';
import { ChartGenerator, ChartData } from './charts';

export interface CategoryStats {
  name: string;
  count: number;
  totalStars: number;
  avgStars: number;
  languages: Map<string, number>;
  recentUpdates: number;
  mostPopular: Repository[];
  trending: Repository[];
  activityHistory: Map<string, number>;
}

export class StatsAnalyzer {
  private readonly RECENT_DAYS = 30;
  private readonly TRENDING_SAMPLE_SIZE = 5;
  private readonly chartGenerator: ChartGenerator;

  constructor(private repos: Repository[]) {
    this.chartGenerator = new ChartGenerator();
  }

  private generateActivityHistory(repos: Repository[]): Map<string, number> {
    const history = new Map<string, number>();
    const now = new Date();
    const monthAgo = new Date(now.getTime() - (30 * 24 * 60 * 60 * 1000));

    // Initialize all dates
    for (let d = new Date(monthAgo); d <= now; d.setDate(d.getDate() + 1)) {
      history.set(d.toISOString().split('T')[0], 0);
    }

    // Count updates per day
    repos.forEach(repo => {
      const updateDate = new Date(repo.updated_at);
      if (updateDate >= monthAgo && updateDate <= now) {
        const dateKey = updateDate.toISOString().split('T')[0];
        history.set(dateKey, (history.get(dateKey) || 0) + 1);
      }
    });

    return history;
  }

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

    // Generate activity history
    const activityHistory = this.generateActivityHistory(repos);

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
      trending,
      activityHistory
    };
  }

  public formatCategoryStats(stats: CategoryStats): string {
    let markdown = `### ${stats.name}\n\n`;

    // Add summary stats
    markdown += `#### Quick Stats\n`;
    markdown += `- Total Repositories: ${stats.count}\n`;
    markdown += `- Total Stars: ${stats.totalStars.toLocaleString()}\n`;
    markdown += `- Average Stars: ${stats.avgStars.toFixed(1)}\n`;
    markdown += `- Recently Updated: ${stats.recentUpdates} repositories in last ${this.RECENT_DAYS} days\n\n`;

    // Add language distribution chart if there are multiple languages
    if (stats.languages.size > 1) {
      const languageData: ChartData = {
        labels: Array.from(stats.languages.keys()),
        values: Array.from(stats.languages.values())
      };
      markdown += `#### Language Distribution\n\n`;
      markdown += this.chartGenerator.generatePieChart(languageData, `Languages in ${stats.name}`);
      markdown += '\n\n';
    }

    // Add activity trend chart
    const activityData: ChartData = {
      labels: Array.from(stats.activityHistory.keys()),
      values: Array.from(stats.activityHistory.values())
    };
    markdown += `#### Activity Trend\n\n`;
    markdown += this.chartGenerator.generateTrendLine(activityData, `Repository Activity - Last 30 Days`);
    markdown += '\n\n';

    // Add popular repositories
    markdown += `#### Most Popular Repositories\n\n`;
    stats.mostPopular.forEach(repo => {
      markdown += `- [${repo.name}](${repo.html_url})\n`;
      markdown += `  - ⭐ ${repo.stargazers_count.toLocaleString()}\n`;
      markdown += `  - 💻 ${repo.language || 'Not specified'}\n`;
      markdown += `  - 📅 Last updated: ${new Date(repo.updated_at).toLocaleDateString()}\n\n`;
    });

    // Add trending repositories
    markdown += `#### Trending Repositories\n\n`;
    stats.trending.forEach(repo => {
      markdown += `- [${repo.name}](${repo.html_url})\n`;
      markdown += `  - 📈 Activity Score: ${this.calculateTrendingScore(repo).toFixed(2)}\n`;
      markdown += `  - ⭐ ${repo.stargazers_count.toLocaleString()} stars\n`;
      markdown += `  - 📅 Updated: ${new Date(repo.updated_at).toLocaleDateString()}\n\n`;
    });

    return markdown;
  }

  private calculateTrendingScore(repo: Repository): number {
    const updatedAt = new Date(repo.updated_at).getTime();
    const now = new Date().getTime();
    const daysSinceUpdate = (now - updatedAt) / (1000 * 60 * 60 * 24);
    return repo.stargazers_count / Math.pow(daysSinceUpdate + 1, 2);
  }

  private getTrendingRepos(repos: Repository[]): Repository[] {
    return [...repos]
      .sort((a, b) => {
        const scoreA = this.calculateTrendingScore(a);
        const scoreB = this.calculateTrendingScore(b);
        return scoreB - scoreA;
      })
      .slice(0, this.TRENDING_SAMPLE_SIZE);
  }

  public generateOverallStats(): Stats {
    const totalRepos = this.repos.length;
    const totalStars = this.repos.reduce((sum, repo) => sum + repo.stargazers_count, 0);
    const languages = new Map<string, number>();
    const topics = new Map<string, number>();
    const activityHistory = this.generateActivityHistory(this.repos);

    // Aggregate stats
    this.repos.forEach(repo => {
      if (repo.language) {
        languages.set(repo.language, (languages.get(repo.language) || 0) + 1);
      }
      repo.topics.forEach(topic => {
        topics.set(topic, (topics.get(topic) || 0) + 1);
      });
    });

    // Calculate active repos
    const activeRepos = this.repos.filter(repo => {
      const updatedAt = new Date(repo.updated_at);
      const now = new Date();
      const daysDiff = (now.getTime() - updatedAt.getTime()) / (1000 * 60 * 60 * 24);
      return daysDiff <= this.RECENT_DAYS;
    });

    // Sort stats
    const topLanguages = Array.from(languages.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10);

    const topTopics = Array.from(topics.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10);

    return {
      totalRepos,
      totalStars,
      avgStarsPerRepo: totalStars / totalRepos,
      activeReposCount: activeRepos.length,
      activeReposPercentage: (activeRepos.length / totalRepos) * 100,
      activityHistory,
      topLanguages,
      topTopics,
      mostStarred: this.repos
        .sort((a, b) => b.stargazers_count - a.stargazers_count)
        .slice(0, 5),
      trending: this.getTrendingRepos(this.repos)
    };
  }

  public formatOverallStats(stats: Stats): string {
    let markdown = `## Overall Statistics\n\n`;

    // Summary stats
    markdown += `### Quick Stats\n\n`;
    markdown += `- Total Repositories: ${stats.totalRepos}\n`;
    markdown += `- Total Stars: ${stats.totalStars.toLocaleString()}\n`;
    markdown += `- Average Stars per Repo: ${stats.avgStarsPerRepo.toFixed(1)}\n`;
    markdown += `- Active Repos: ${stats.activeReposCount} (${stats.activeReposPercentage.toFixed(1)}% updated in last ${this.RECENT_DAYS} days)\n\n`;

    // Activity trend
    const activityData: ChartData = {
      labels: Array.from(stats.activityHistory.keys()),
      values: Array.from(stats.activityHistory.values())
    };
    markdown += `### Repository Activity\n\n`;
    markdown += this.chartGenerator.generateTrendLine(activityData, 'Activity Trend - Last 30 Days');
    markdown += '\n\n';

    // Language distribution
    const languageData: ChartData = {
      labels: stats.topLanguages.map(([lang]) => lang),
      values: stats.topLanguages.map(([, count]) => count)
    };
    markdown += `### Language Distribution\n\n`;
    markdown += this.chartGenerator.generateBarChart(languageData, 'Top Languages');
    markdown += '\n\n';

    // Topics distribution
    const topicsData: ChartData = {
      labels: stats.topTopics.map(([topic]) => topic),
      values: stats.topTopics.map(([, count]) => count)
    };
    markdown += `### Popular Topics\n\n`;
    markdown += this.chartGenerator.generatePieChart(topicsData, 'Most Used Topics');
    markdown += '\n\n';

    // Most starred repositories
    markdown += `### Most Starred Repositories\n\n`;
    stats.mostStarred.forEach((repo, index) => {
      markdown += `${index + 1}. [${repo.name}](${repo.html_url})\n`;
      markdown += `   - ⭐ ${repo.stargazers_count.toLocaleString()} stars\n`;
      markdown += `   - 💻 ${repo.language || 'Not specified'}\n`;
      markdown += `   - 📅 Last updated: ${new Date(repo.updated_at).toLocaleDateString()}\n\n`;
    });

    // Trending repositories
    markdown += `### Trending Repositories\n\n`;
    stats.trending.forEach((repo, index) => {
      markdown += `${index + 1}. [${repo.name}](${repo.html_url})\n`;
      markdown += `   - 📈 Activity Score: ${this.calculateTrendingScore(repo).toFixed(2)}\n`;
      markdown += `   - ⭐ ${repo.stargazers_count.toLocaleString()} stars\n`;
      markdown += `   - 📅 Updated: ${new Date(repo.updated_at).toLocaleDateString()}\n\n`;
    });

    return markdown;
  }
}