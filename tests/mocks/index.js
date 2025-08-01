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

/**
 * Create Supervision client mock for testing
 * @returns {Object} Mock supervision client with all methods
 */
export function createSupervisionClientMock() {
  return createMockObject('SupervisionClient', {
    // Object detection methods
    detectObjects: jest.fn().mockResolvedValue([
      { class: 'person', bbox: [100, 100, 200, 300], confidence: 0.85 },
      { class: 'car', bbox: [300, 200, 450, 350], confidence: 0.92 }
    ]),

    // Annotation generation methods
    generateAnnotations: jest.fn().mockResolvedValue({
      coco: { 
        annotations: [{ id: 1, category_id: 1, bbox: [100, 100, 100, 200] }],
        categories: [{ id: 1, name: 'person' }]
      },
      yolo: {
        labels: ['0 0.25 0.25 0.15 0.40']
      },
      pascal_voc: {
        annotation: { object: [{ name: 'person', bndbox: { xmin: 100, ymin: 100, xmax: 200, ymax: 300 } }] }
      }
    }),

    // Quality assessment methods
    calculateMetrics: jest.fn().mockResolvedValue({
      overall_score: 0.88,
      precision: 0.92,
      recall: 0.85,
      f1_score: 0.88,
      map50: 0.76,
      map95: 0.45
    }),

    runQualityChecks: jest.fn().mockResolvedValue({
      overall_score: 0.85,
      metrics: {
        precision: 0.90,
        recall: 0.82,
        f1_score: 0.86,
        map50: 0.78
      },
      checks: ['bbox_accuracy', 'class_consistency', 'annotation_completeness']
    }),

    checkAnnotationConsistency: jest.fn().mockResolvedValue({
      score: 0.93,
      issues: [],
      consistency_metrics: {
        bbox_consistency: 0.95,
        class_consistency: 0.91,
        temporal_consistency: 0.89
      }
    }),

    checkAnnotationCompleteness: jest.fn().mockResolvedValue({
      score: 0.87,
      missing_annotations: 3,
      completeness_metrics: {
        object_coverage: 0.92,
        class_coverage: 0.85,
        spatial_coverage: 0.88
      }
    }),

    // Stream processing methods
    createStreamValidator: jest.fn().mockResolvedValue({
      start: jest.fn().mockImplementation((callbacks) => {
        // Simulate stream processing
        if (callbacks && callbacks.onFrameProcessed) {
          setTimeout(() => {
            callbacks.onFrameProcessed({
              frameId: 'stream-frame-001',
              detections: [{ class: 'person', confidence: 0.87 }],
              processingTime: 45
            });
          }, 100);
        }
        return Promise.resolve();
      }),
      stop: jest.fn().mockResolvedValue()
    }),

    // Benchmark methods
    createBenchmarkSuite: jest.fn().mockResolvedValue({
      execute: jest.fn().mockResolvedValue({
        throughput: 35.7,
        latency: 28.1,
        accuracy: 0.84,
        resourceUsage: {
          cpu: 72,
          memory: 1536,
          gpu: 65
        },
        passed: true
      })
    }),

    // Report generation methods
    generateReport: jest.fn().mockResolvedValue({
      visualizations: [
        '/reports/detection_heatmap.png',
        '/reports/confidence_distribution.png',
        '/reports/class_performance.png'
      ],
      charts: [
        '/reports/precision_recall_curve.json',
        '/reports/confusion_matrix.json'
      ],
      statistics: {
        total_detections: 1547,
        total_images: 250,
        average_objects_per_image: 6.2
      }
    }),

    exportReport: jest.fn().mockResolvedValue({
      json: '/reports/supervision-report.json',
      html: '/reports/supervision-report.html',
      pdf: '/reports/supervision-report.pdf'
    }),

    // Utility methods
    validateModel: jest.fn().mockResolvedValue({
      isValid: true,
      modelInfo: {
        type: 'yolov8',
        version: '8.0.0',
        classes: 80,
        inputShape: [640, 640, 3]
      }
    }),

    optimizeModel: jest.fn().mockResolvedValue({
      optimizedPath: '/models/optimized_model.pt',
      optimizations: ['quantization', 'pruning'],
      performanceImprovement: 0.35
    })
  });
}

/**
 * Create metrics collector mock for testing
 * @returns {Object} Mock metrics collector
 */
export function createMetricsCollectorMock() {
  return createMockObject('MetricsCollector', {
    recordValidation: jest.fn().mockResolvedValue({
      recordId: 'metric-record-001',
      timestamp: Date.now(),
      stored: true
    }),

    recordRealTimeFrame: jest.fn().mockResolvedValue({
      frameId: 'frame-record-001',
      metrics: {
        processingTime: 33,
        detectionCount: 2,
        avgConfidence: 0.87
      }
    }),

    getMetrics: jest.fn().mockResolvedValue({
      totalValidations: 150,
      averageQualityScore: 0.82,
      performanceMetrics: {
        avgProcessingTime: 45.2,
        avgThroughput: 28.7
      }
    }),

    generateReport: jest.fn().mockResolvedValue({
      reportPath: '/metrics/performance-report.json',
      summary: 'Metrics collected successfully'
    })
  });
}

// Export the core factory for use in other files
export { createMockObject };