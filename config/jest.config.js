/**
 * Jest Configuration for TDD London School Methodology
 * 
 * Optimized for mock-first development, behavior verification,
 * and outside-in testing approach with SPARC methodology support.
 */

export default {
  // Test environment
  testEnvironment: 'node',
  
  // Module settings
  moduleFileExtensions: ['js', 'json'],
  transform: {
    '^.+\\.js$': 'babel-jest'
  },
  
  // Test discovery
  testMatch: [
    '**/tests/**/*.test.js',
    '**/tests/**/*.spec.js'
  ],
  
  // Setup files
  setupFilesAfterEnv: [
    '<rootDir>/tests/setup.js'
  ],
  
  // Coverage settings optimized for London School TDD
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/index.js',
    '!src/**/*.config.js',
    '!src/**/*.mock.js'
  ],
  coverageDirectory: 'coverage',
  coverageReporters: [
    'text',
    'lcov', 
    'html',
    'json-summary'
  ],
  
  // London School TDD focuses on behavior, so we adjust thresholds
  coverageThreshold: {
    global: {
      branches: 75,      // Lower than classical TDD due to mock-first approach
      functions: 85,     // Focus on interaction coverage
      lines: 80,         // Reasonable line coverage
      statements: 80     // Statement coverage
    },
    // Service layer should have higher coverage (core business logic)
    'src/services/': {
      branches: 85,
      functions: 95,
      lines: 90,
      statements: 90
    },
    // Controllers focus on coordination (behavior over implementation)
    'src/controllers/': {
      branches: 70,
      functions: 90,
      lines: 75,
      statements: 75
    }
  },
  
  // Mock settings for London School
  clearMocks: true,
  restoreMocks: true,
  resetMocks: true,
  
  // Test execution settings
  verbose: true,
  testTimeout: 10000,
  maxWorkers: '50%',
  
  // Error handling
  errorOnDeprecated: true,
  
  // Reporting
  reporters: [
    'default',
    [
      'jest-junit',
      {
        outputDirectory: 'coverage',
        outputName: 'junit.xml',
        suiteNameTemplate: '{filepath}',
        classNameTemplate: '{classname}',
        titleTemplate: '{title}'
      }
    ]
  ],
  
  // Global test settings for London School methodology
  globals: {
    'london-school': {
      mockFirst: true,
      behaviorVerification: true,
      contractTesting: true,
      outsideInDevelopment: true
    }
  },
  
  // Test categories for SPARC methodology
  projects: [
    // Acceptance tests (SPARC Specification phase)
    {
      displayName: 'acceptance',
      testMatch: ['<rootDir>/tests/acceptance/**/*.test.js'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
      globals: {
        testType: 'acceptance',
        sparcPhase: 'specification'
      }
    },
    
    // Unit tests (SPARC Refinement phase)
    {
      displayName: 'unit', 
      testMatch: ['<rootDir>/tests/unit/**/*.test.js'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
      globals: {
        testType: 'unit',
        sparcPhase: 'refinement'
      }
    },
    
    // Contract tests (SPARC Architecture phase)
    {
      displayName: 'contracts',
      testMatch: ['<rootDir>/tests/contracts/**/*.test.js'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
      globals: {
        testType: 'contracts',
        sparcPhase: 'architecture'
      }
    },
    
    // Integration tests (SPARC Completion phase)
    {
      displayName: 'integration',
      testMatch: ['<rootDir>/tests/integration/**/*.test.js'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
      globals: {
        testType: 'integration',
        sparcPhase: 'completion'
      }
    }
  ],
  
  // Watch mode settings for TDD workflow
  watchPathIgnorePatterns: [
    '<rootDir>/node_modules/',
    '<rootDir>/coverage/',
    '<rootDir>/docs/'
  ],
  
  // London School specific test naming patterns
  testNamePattern: '(should|when|given|then|describe|it)',
  
  // Mock module patterns
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@tests/(.*)$': '<rootDir>/tests/$1',
    '^@mocks/(.*)$': '<rootDir>/tests/mocks/$1'
  },
  
  // Custom matchers and utilities
  setupFiles: [
    '<rootDir>/tests/jest.polyfills.js'
  ],
  
  // Snapshot settings
  snapshotSerializers: [
    'jest-serializer-path'
  ],
  
  // Test result processor for SPARC methodology tracking
  testResultsProcessor: '<rootDir>/tests/utils/sparcResultsProcessor.js'
};