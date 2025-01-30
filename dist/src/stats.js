"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.StatsAnalyzer = void 0;
class StatsAnalyzer {
    constructor(repositories) {
        this.repositories = repositories;
    }
    generateOverallStats() {
        const languages = new Map();
        const categoryDistribution = new Map();
        let totalStars = 0;
        this.repositories.forEach(repo => {
            // Count stars
            totalStars += repo.stargazers_count;
            // Count languages
            if (repo.language) {
                languages.set(repo.language, (languages.get(repo.language) || 0) + 1);
            }
        });
        // Sort repositories by stars for top repos
        const topRepositories = [...this.repositories]
            .sort((a, b) => b.stargazers_count - a.stargazers_count)
            .slice(0, 5);
        // Sort repositories by update date for recent updates
        const mostRecentUpdates = [...this.repositories]
            .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
            .slice(0, 5);
        return {
            totalRepositories: this.repositories.length,
            totalStars,
            popularLanguages: languages,
            topRepositories,
            mostRecentUpdates,
            categoryDistribution
        };
    }
    generateCategoryStats(categoryName, repositories) {
        const languages = new Map();
        let totalStars = 0;
        repositories.forEach(repo => {
            totalStars += repo.stargazers_count;
            if (repo.language) {
                languages.set(repo.language, (languages.get(repo.language) || 0) + 1);
            }
        });
        const topRepositories = [...repositories]
            .sort((a, b) => b.stargazers_count - a.stargazers_count)
            .slice(0, 3);
        const mostRecentUpdates = [...repositories]
            .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
            .slice(0, 3);
        return {
            name: categoryName,
            repositories,
            totalStars,
            languages,
            topRepositories,
            mostRecentUpdates
        };
    }
    formatOverallStats(stats) {
        let content = '# Starred Repositories Statistics\n\n';
        content += `## Overall Statistics\n\n`;
        content += `- Total Repositories: ${stats.totalRepositories}\n`;
        content += `- Total Stars: ${stats.totalStars}\n\n`;
        content += `### Popular Languages\n\n`;
        const sortedLanguages = [...stats.popularLanguages.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);
        sortedLanguages.forEach(([language, count]) => {
            content += `- ${language}: ${count} repositories\n`;
        });
        content += `\n### Top Repositories\n\n`;
        stats.topRepositories.forEach(repo => {
            content += `- [${repo.name}](${repo.html_url}) - ⭐ ${repo.stargazers_count}\n  ${repo.description || ''}\n`;
        });
        content += `\n### Recent Updates\n\n`;
        stats.mostRecentUpdates.forEach(repo => {
            const date = new Date(repo.updated_at).toLocaleDateString();
            content += `- [${repo.name}](${repo.html_url}) - ${date}\n  ${repo.description || ''}\n`;
        });
        return content;
    }
    formatMarkdown(stats) {
        let content = `\n### ${stats.name}\n\n`;
        content += `Total repositories: ${stats.repositories.length} | Total stars: ${stats.totalStars}\n\n`;
        if (stats.languages.size > 0) {
            content += `**Languages:**\n`;
            const sortedLanguages = [...stats.languages.entries()]
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5);
            sortedLanguages.forEach(([language, count]) => {
                content += `- ${language}: ${count} repositories\n`;
            });
            content += '\n';
        }
        if (stats.topRepositories.length > 0) {
            content += `**Top Repositories:**\n`;
            stats.topRepositories.forEach(repo => {
                content += `- [${repo.name}](${repo.html_url}) - ⭐ ${repo.stargazers_count}\n  ${repo.description || ''}\n`;
            });
            content += '\n';
        }
        if (stats.mostRecentUpdates.length > 0) {
            content += `**Recent Updates:**\n`;
            stats.mostRecentUpdates.forEach(repo => {
                const date = new Date(repo.updated_at).toLocaleDateString();
                content += `- [${repo.name}](${repo.html_url}) - ${date}\n  ${repo.description || ''}\n`;
            });
            content += '\n';
        }
        content += `**All Repositories:**\n`;
        stats.repositories
            .sort((a, b) => b.stargazers_count - a.stargazers_count)
            .forEach(repo => {
            content += `- [${repo.name}](${repo.html_url}) - ⭐ ${repo.stargazers_count}${repo.language ? ` - ${repo.language}` : ''}\n  ${repo.description || ''}\n`;
        });
        return content + '\n';
    }
}
exports.StatsAnalyzer = StatsAnalyzer;
