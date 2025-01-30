import { Octokit } from '@octokit/rest';
import { Repository, CategoryConfig } from './src/types';
import { StatsAnalyzer } from './src/stats';
import * as dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

// Configuration
const categories: CategoryConfig[] = [
  // ... (rest of the file remains the same)
];
