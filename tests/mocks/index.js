/**
 * Mock Definitions for AI Model Validation Domain
 * Following London School TDD - Define collaborator contracts through mocks
 */

import { jest } from '@jest/globals';

// Data Capture Layer Mocks
export const createWebcamCaptureMock = () => createMockObject('WebcamCapture', {
  initialize: jest.fn().mockResolvedValue(true),
  captureFrame: jest.fn().mockResolvedValue({
    id: 'frame-123',
    timestamp: Date.now(),
    data: Buffer.from('mock-image-data'),
    metadata: { width: 640, height: 480, format: 'jpeg' }
  }),
  startStream: jest.fn().mockResolvedValue(true),
  stopStream: jest.fn().mockResolvedValue(true),
  isActive: jest.fn().mockReturnValue(true)
});

export const createDataStorageMock = () => createMockObject('DataStorage', {
  save: jest.fn().mockResolvedValue({ id: 'data-456', path: '/mock/path' }),
  load: jest.fn().mockResolvedValue({ data: 'mock-data' }),
  exists: jest.fn().mockReturnValue(true),
  delete: jest.fn().mockResolvedValue(true),
  list: jest.fn().mockResolvedValue(['file1.jpg', 'file2.jpg'])
});

// Annotation Layer Mocks (CVAT integration)
export const createAnnotationServiceMock = () => createMockObject('AnnotationService', {
  createProject: jest.fn().mockResolvedValue({ projectId: 'cvat-project-789' }),
  uploadData: jest.fn().mockResolvedValue({ taskId: 'cvat-task-101' }),
  getAnnotations: jest.fn().mockResolvedValue({
    annotations: [
      { id: 1, label: 'object', bbox: [10, 10, 50, 50] }
    ]
  }),
  exportAnnotations: jest.fn().mockResolvedValue({ format: 'YOLO', data: 'mock-export' }),
  validateAnnotations: jest.fn().mockResolvedValue({ valid: true, errors: [] })
});

export const createCvatClientMock = () => createMockObject('CvatClient', {
  connect: jest.fn().mockResolvedValue(true),
  authenticate: jest.fn().mockResolvedValue({ token: 'mock-token' }),
  createTask: jest.fn().mockResolvedValue({ id: 'task-202' }),
  uploadImages: jest.fn().mockResolvedValue({ uploaded: 5 }),
  getTaskStatus: jest.fn().mockResolvedValue({ status: 'completed' })
});

// Validation Layer Mocks (Deepchecks integration)
export const createValidationServiceMock = () => createMockObject('ValidationService', {
  validateDataset: jest.fn().mockResolvedValue({
    passed: true,
    score: 0.95,
    checks: [
      { name: 'data_integrity', passed: true, score: 1.0 },
      { name: 'label_distribution', passed: true, score: 0.9 }
    ]
  }),
  validateModel: jest.fn().mockResolvedValue({
    passed: true,
    performance: { accuracy: 0.92, precision: 0.89, recall: 0.94 },
    checks: ['model_performance', 'data_drift', 'feature_importance']
  }),
  generateReport: jest.fn().mockResolvedValue({
    reportId: 'validation-report-303',
    htmlPath: '/mock/report.html',
    summary: 'All validation checks passed'
  })
});

export const createDeepchecksClientMock = () => createMockObject('DeepchecksClient', {
  runSuite: jest.fn().mockResolvedValue({ suite_result: 'passed' }),
  runCheck: jest.fn().mockResolvedValue({ check_result: 'passed' }),
  generateHtmlReport: jest.fn().mockResolvedValue('/mock/deepchecks-report.html'),
  getCheckResults: jest.fn().mockResolvedValue([
    { check_name: 'data_integrity', result: 'PASS' }
  ])
});

// Model Training Layer Mocks (Ultralytics integration)
export const createModelTrainerMock = () => createMockObject('ModelTrainer', {
  initialize: jest.fn().mockResolvedValue(true),
  train: jest.fn().mockResolvedValue({
    modelId: 'yolo-model-404',
    epochs: 100,
    finalLoss: 0.05,
    metrics: { mAP: 0.87, precision: 0.89, recall: 0.85 }
  }),
  validateModel: jest.fn().mockResolvedValue({
    validation_loss: 0.07,
    validation_metrics: { mAP: 0.84 }
  }),
  saveModel: jest.fn().mockResolvedValue({ path: '/mock/model.pt' }),
  loadModel: jest.fn().mockResolvedValue({ loaded: true })
});

export const createUltralyticsClientMock = () => createMockObject('UltralyticsClient', {
  createModel: jest.fn().mockResolvedValue({ model_instance: 'mock-yolo' }),
  trainModel: jest.fn().mockResolvedValue({ training_complete: true }),
  predict: jest.fn().mockResolvedValue({
    predictions: [{ class: 'object', confidence: 0.92, bbox: [10, 10, 50, 50] }]
  }),
  exportModel: jest.fn().mockResolvedValue({ export_path: '/mock/exported-model.onnx' })
});

// Infrastructure Mocks
export const createFileSystemMock = () => createMockObject('FileSystem', {
  readFile: jest.fn().mockResolvedValue('mock-file-content'),
  writeFile: jest.fn().mockResolvedValue(true),
  createDirectory: jest.fn().mockResolvedValue(true),
  exists: jest.fn().mockReturnValue(true),
  getStats: jest.fn().mockResolvedValue({ size: 1024, modified: Date.now() })
});

export const createLoggerMock = () => {
  const mockLogger = createMockObject('Logger', {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
    createChild: jest.fn()
  });
  
  // Avoid circular dependency by creating a simple child mock
  mockLogger.createChild.mockReturnValue({
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
    _mockName: 'Logger.child',
    _mockType: 'london-school-mock'
  });
  
  return mockLogger;
};

export const createConfigManagerMock = () => createMockObject('ConfigManager', {
  get: jest.fn().mockReturnValue('mock-config-value'),
  set: jest.fn().mockReturnValue(true),
  load: jest.fn().mockResolvedValue(true),
  validate: jest.fn().mockReturnValue(true)
});

// Workflow Orchestration Mocks
export const createWorkflowOrchestratorMock = () => createMockObject('WorkflowOrchestrator', {
  executeWorkflow: jest.fn().mockResolvedValue({
    workflowId: 'workflow-505',
    status: 'completed',
    steps: ['capture', 'annotate', 'validate', 'train'],
    results: { success: true }
  }),
  getWorkflowStatus: jest.fn().mockResolvedValue({ status: 'running', progress: 0.75 }),
  cancelWorkflow: jest.fn().mockResolvedValue(true)
});

// Contract Definitions for Swarm Coordination
export const CONTRACTS = {
  WebcamCapture: {
    initialize: { input: [], output: 'boolean' },
    captureFrame: { input: [], output: 'object' },
    startStream: { input: [], output: 'boolean' },
    stopStream: { input: [], output: 'boolean' }
  },
  
  AnnotationService: {
    createProject: { input: ['object'], output: 'object' },
    uploadData: { input: ['array'], output: 'object' },
    getAnnotations: { input: ['string'], output: 'object' }
  },
  
  ValidationService: {
    validateDataset: { input: ['object'], output: 'object' },
    validateModel: { input: ['object', 'object'], output: 'object' },
    generateReport: { input: ['object'], output: 'object' }
  },
  
  ModelTrainer: {
    train: { input: ['object'], output: 'object' },
    validateModel: { input: ['object'], output: 'object' },
    saveModel: { input: ['string'], output: 'object' }
  }
};

// Mock Object Factory - Core implementation
const createMockObject = (name, methods) => {
  const mock = { _mockName: name };
  
  Object.entries(methods).forEach(([methodName, mockFn]) => {
    mock[methodName] = mockFn;
  });
  
  return mock;
};

// Mock Factory for Swarm Coordination
export const createSwarmMock = (mockName, methods) => {
  const mock = createMockObject(mockName, methods);
  
  // Add swarm-specific metadata
  mock._swarmContract = CONTRACTS[mockName] || {};
  mock._swarmId = `swarm-mock-${Date.now()}`;
  
  return mock;
};

// Export the core factory for use in other files
export { createMockObject };