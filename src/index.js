/**
 * AI Model Validation PoC - Main Entry Point
 * London School TDD - Dependency injection and testable architecture
 */

import { container } from './infrastructure/dependency-injection/container.js';

// This file will be driven by the tests
// London School: Implementation follows test requirements

console.log('🚀 AI Model Validation PoC Starting...');
console.log('🎭 London School TDD Architecture Initialized');
console.log('📋 Mock-first development approach enabled');

// Dependency injection setup will be driven by test requirements
// Tests will define what collaborators are needed and how they interact

export { container } from './infrastructure/dependency-injection/container.js';
export * from './domain/contracts/index.js';