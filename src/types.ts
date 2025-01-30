export interface Repository {
  name: string;
  full_name: string;
  html_url: string;
  description: string;
  stargazers_count: number;
  language: string;
  topics: string[];
  pushed_at: string;
  updated_at: string;
  created_at: string;
  fork: boolean;
  homepage: string | null;
  size: number;
  default_branch: string;
  open_issues_count: number;
  has_issues: boolean;
  has_wiki: boolean;
  archived: boolean;
  disabled: boolean;
  visibility: string;
  forks_count: number;
  watchers_count: number;
}

export interface CategoryConfig {
  name: string;
  keywords: string[];
  tags: string[];
  description: string;
}

export interface Stats {
  totalRepos: number;
  totalStars: number;
  avgStarsPerRepo: number;
  activeReposCount: number;
  activeReposPercentage: number;
  topLanguages: [string, number][];
  topTopics: [string, number][];
  mostStarred: Repository[];
  trending: Repository[];
}

export interface CategoryStats {
  name: string;
  count: number;
  totalStars: number;
  avgStars: number;
  languages: Map<string, number>;
  recentUpdates: number;
  mostPopular: Repository[];
  trending: Repository[];
}