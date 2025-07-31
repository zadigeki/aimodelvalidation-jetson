# TDD London School Setup for AI Model Validation PoC

## Overview

This project implements the **London School TDD approach** (mockist style) for an AI model validation PoC. The London School emphasizes:

- **Mock-first development**: Define collaborator contracts through test doubles
- **Outside-in development**: Start with acceptance tests and work inward
- **Behavior verification**: Focus on HOW objects collaborate, not WHAT they contain
- **Dependency injection**: Enable easy mock substitution for testing

## Project Structure

```
/workspaces/aimodelvalidation/
├── src/
│   ├── domain/
│   │   └── contracts/           # Interface definitions driven by mocks
│   ├── infrastructure/
│   │   └── dependency-injection/ # DI container for testability
│   ├── application/             # Application services (TBD)
│   └── index.js                 # Main entry point
├── tests/
│   ├── setup.js                 # Jest configuration with London School patterns
│   ├── mocks/                   # Mock definitions and contracts
│   │   └── index.js            # Comprehensive mock library
│   ├── acceptance/             # Outside-in acceptance tests
│   │   └── ai-model-validation-workflow.test.js
│   ├── unit/                   # Mock-driven unit tests
│   │   └── webcam-capture-service.test.js
│   └── integration/            # Service integration tests
│       └── cvat-deepchecks-integration.test.js
└── package.json                # Dependencies and scripts
```

## London School TDD Principles Applied

### 1. Mock-First Approach

All external dependencies are mocked to:
- Define clear contracts and interfaces
- Enable isolated unit testing
- Drive design through collaboration patterns
- Facilitate dependency injection

```javascript
// Example: Mock defines the contract
const webcamCapture = createWebcamCaptureMock();
const dataStorage = createDataStorageMock();
const logger = createLoggerMock();

// Service constructor driven by mock contracts
const service = new WebcamCaptureService({
  webcamCapture,
  dataStorage,
  logger
});
```

### 2. Outside-In Development

Development starts with acceptance tests that define user behavior:

```javascript
describe('AI Model Validation Workflow - Acceptance Tests', () => {
  it('should successfully execute complete AI model validation workflow', async () => {
    // Arrange - Define expected workflow behavior
    // Act - Execute workflow
    // Assert - Verify collaboration between ALL components
  });
});
```

### 3. Behavior Verification

Tests focus on interactions between objects rather than internal state:

```javascript
// London School: Test collaborations
expect(webcamCapture.initialize).toHaveBeenCalledBefore(webcamCapture.captureFrame);
expect(annotationService.createProject).toHaveBeenCalledBefore(annotationService.uploadData);
expect(validationService.validateDataset).toHaveBeenCalledBefore(modelTrainer.train);
```

### 4. Dependency Injection

The DI container enables mock injection for testing:

```javascript
// Test setup
const testContainer = createTestContainer({
  webcamCapture: createWebcamCaptureMock(),
  dataStorage: createDataStorageMock(),
  logger: createLoggerMock()
});

// Production setup
container.register('webcamCapture', WebcamCaptureImpl);
container.register('dataStorage', FileSystemStorage);
container.register('logger', WinstonLogger);
```

## PoC Domain Coverage

The TDD setup covers the complete AI model validation workflow:

### Data Capture Layer
- **WebcamCapture**: Manual data capture with webcam
- **DataStorage**: Local file system storage
- Mock contracts define camera initialization, frame capture, and storage

### Annotation Layer (CVAT Integration)
- **AnnotationService**: CVAT project management
- **CvatClient**: Direct CVAT API integration
- Mock contracts define project creation, data upload, annotation export

### Validation Layer (Deepchecks Integration)
- **ValidationService**: Dataset and model validation
- **DeepchecksClient**: Deepchecks suite execution
- Mock contracts define validation checks, report generation

### Model Training Layer (Ultralytics Integration)
- **ModelTrainer**: YOLO model training orchestration
- **UltralyticsClient**: Direct Ultralytics API integration
- Mock contracts define model training, validation, export

## Test Categories

### Acceptance Tests
- Complete workflow scenarios
- User behavior verification
- End-to-end collaboration patterns

### Unit Tests
- Individual service behavior
- Mock-driven contract verification
- Isolated component testing

### Integration Tests
- Service-to-service collaboration
- External API integration patterns
- Error handling and resilience

## Running Tests

```bash
# All tests
npm test

# Specific test categories
npm run test:acceptance
npm run test:unit
npm run test:integration

# Watch mode for TDD development
npm run test:watch

# Coverage reporting
npm run test:coverage
```

## London School TDD Development Flow

1. **Write acceptance test** - Define user behavior and system boundaries
2. **Create mocks** - Define collaborator contracts and interfaces  
3. **Write unit tests** - Specify object interactions and responsibilities
4. **Implement services** - Create minimal implementation to satisfy mocks
5. **Refactor** - Improve design while maintaining mock contracts
6. **Repeat** - Continue outside-in for next feature

## Mock Library Features

The comprehensive mock library includes:
- **Behavior tracking**: All interactions recorded and verifiable
- **Contract definitions**: Clear interface specifications
- **Swarm coordination**: Integration with agent swarm workflows
- **Custom matchers**: London School-specific Jest extensions

## Key Benefits

1. **Testability First**: Every component designed for easy testing
2. **Clear Contracts**: Interfaces defined through mock expectations
3. **Rapid Feedback**: Fast test execution with comprehensive mocking
4. **Design Guidance**: Tests drive architecture and design decisions
5. **Swarm Integration**: Coordinated development with other agents

## Next Steps

The TDD London School foundation is now ready for:
1. Implementation of actual services driven by test requirements
2. Integration with real CVAT, Deepchecks, and Ultralytics APIs
3. Extension of test coverage based on PoC requirements
4. Coordination with other swarm agents for comprehensive development

This setup ensures that all development follows London School TDD principles with mock-first, outside-in development focusing on behavior verification and object collaboration.