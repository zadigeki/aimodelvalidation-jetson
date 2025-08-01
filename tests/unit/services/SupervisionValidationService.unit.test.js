/**
 * SupervisionValidationService Unit Tests
 * London School TDD - Supervision.py integration with comprehensive validation workflows
 */

import { jest, describe, it, beforeEach, expect } from '@jest/globals';
import { SupervisionValidationService } from '../../../src/services/SupervisionValidationService.js';
import { 
  createSupervisionClientMock, 
  createLoggerMock, 
  createMetricsCollectorMock 
} from '../../mocks/index.js';

describe('SupervisionValidationService', () => {
  let supervisionService;
  let mockSupervisionClient;
  let mockLogger;
  let mockMetricsCollector;

  beforeEach(() => {
    // London School: Mocks define collaboration contracts
    mockSupervisionClient = createSupervisionClientMock();
    mockLogger = createLoggerMock();
    mockMetricsCollector = createMetricsCollectorMock();

    supervisionService = new SupervisionValidationService({
      supervisionClient: mockSupervisionClient,
      logger: mockLogger,
      metricsCollector: mockMetricsCollector,
      validationConfig: {
        confidenceThreshold: 0.7,
        iouThreshold: 0.5,
        maxDetections: 50,
        annotationFormats: ['coco', 'yolo'],
        qualityMetrics: ['precision', 'recall', 'f1_score'],
        validationModes: ['single_image', 'batch_processing', 'real_time']
      }
    });
  });

  describe('Single Image Validation Workflow', () => {
    it('should validate single image and coordinate detection pipeline', async () => {
      // Arrange
      const imageData = {
        id: 'test-image-001',
        path: '/path/to/test-image.jpg',
        metadata: { width: 640, height: 480 },
        groundTruth: [
          { class: 'person', bbox: [100, 100, 200, 300], confidence: 1.0 }
        ]
      };
      
      const modelConfig = {
        type: 'yolov8',
        weights: '/path/to/model.pt',
        confidenceThreshold: 0.5
      };

      const mockDetections = [
        { class: 'person', bbox: [105, 95, 195, 305], confidence: 0.85 }
      ];

      const mockAnnotations = {
        coco: { /* COCO format annotations */ },
        yolo: { /* YOLO format annotations */ }
      };

      const mockQualityMetrics = {
        overall_score: 0.92,
        precision: 0.95,
        recall: 0.89,
        f1_score: 0.92
      };

      // Setup mocks
      mockSupervisionClient.detectObjects.mockResolvedValue(mockDetections);
      mockSupervisionClient.generateAnnotations.mockResolvedValue(mockAnnotations);
      mockSupervisionClient.calculateMetrics.mockResolvedValue(mockQualityMetrics);

      // Act
      const result = await supervisionService.validateSingleImage(imageData, modelConfig);

      // Assert - London School: Verify collaboration pattern
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Validating single image with Supervision',
        { imageId: 'test-image-001', modelType: 'yolov8' }
      );

      expect(mockSupervisionClient.detectObjects).toHaveBeenCalledWith(
        imageData,
        modelConfig
      );

      expect(mockSupervisionClient.generateAnnotations).toHaveBeenCalledWith(
        mockDetections,
        ['coco', 'yolo']
      );

      expect(mockSupervisionClient.calculateMetrics).toHaveBeenCalledWith(
        mockDetections,
        imageData.groundTruth
      );

      expect(mockMetricsCollector.recordValidation).toHaveBeenCalledWith(
        'single_image',
        expect.objectContaining({
          imageId: 'test-image-001',
          passed: true
        })
      );

      // Verify result structure
      expect(result).toMatchObject({
        imageId: 'test-image-001',
        timestamp: expect.any(Number),
        detections: {
          count: 1,
          objects: mockDetections,
          confidence: 0.85
        },
        annotations: mockAnnotations,
        quality: mockQualityMetrics,
        passed: true
      });
    });

    it('should handle single image validation failure gracefully', async () => {
      // Arrange
      const imageData = { id: 'fail-image-001', path: '/bad-path.jpg' };
      const modelConfig = { type: 'yolov8' };

      mockSupervisionClient.detectObjects.mockRejectedValue(
        new Error('Image processing failed')
      );

      // Act & Assert
      await expect(
        supervisionService.validateSingleImage(imageData, modelConfig)
      ).rejects.toThrow('Image processing failed');

      expect(mockLogger.info).toHaveBeenCalledWith(
        'Validating single image with Supervision',
        { imageId: 'fail-image-001', modelType: 'yolov8' }
      );
    });
  });

  describe('Batch Processing Validation Workflow', () => {
    it('should validate image batch and coordinate parallel processing', async () => {
      // Arrange
      const imageBatch = [
        { id: 'batch-img-001', path: '/img1.jpg', groundTruth: [] },
        { id: 'batch-img-002', path: '/img2.jpg', groundTruth: [] },
        { id: 'batch-img-003', path: '/img3.jpg', groundTruth: [] }
      ];
      
      const modelConfig = { type: 'yolov8', batchSize: 3 };

      // Mock successful individual validations
      mockSupervisionClient.detectObjects.mockResolvedValue([]);
      mockSupervisionClient.generateAnnotations.mockResolvedValue({});
      mockSupervisionClient.calculateMetrics.mockResolvedValue({
        overall_score: 0.8,
        precision: 0.85,
        recall: 0.75
      });

      // Act
      const result = await supervisionService.validateBatch(imageBatch, modelConfig);

      // Assert - Verify batch coordination
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Starting batch validation with Supervision',
        { batchSize: 3, modelType: 'yolov8' }
      );

      expect(mockSupervisionClient.detectObjects).toHaveBeenCalledTimes(3);
      expect(mockMetricsCollector.recordValidation).toHaveBeenCalledWith(
        'batch_processing',
        expect.objectContaining({
          batchId: expect.stringMatching(/^batch-\d+$/),
          summary: {
            totalImages: 3,
            processedSuccessfully: 3,
            failed: 0,
            successRate: 1
          }
        })
      );

      // Verify batch result structure
      expect(result).toMatchObject({
        batchId: expect.stringMatching(/^batch-\d+$/),
        summary: {
          totalImages: 3,
          processedSuccessfully: 3,
          failed: 0,
          successRate: 1
        },
        performance: expect.objectContaining({
          totalTime: expect.any(Number),
          averageProcessingTime: expect.any(Number)
        }),
        results: expect.arrayContaining([
          expect.objectContaining({ imageId: 'batch-img-001', passed: true }),
          expect.objectContaining({ imageId: 'batch-img-002', passed: true }),
          expect.objectContaining({ imageId: 'batch-img-003', passed: true })
        ])
      });
    });

    it('should handle partial batch failures and maintain processing', async () => {
      // Arrange
      const imageBatch = [
        { 
          id: 'good-img', 
          path: '/good.jpg',
          groundTruth: [{ class: 'object', bbox: [100, 100, 200, 200], confidence: 1.0 }]
        },
        { 
          id: 'bad-img', 
          path: '/bad.jpg',
          groundTruth: [{ class: 'object', bbox: [100, 100, 200, 200], confidence: 1.0 }]
        }
      ];
      
      const modelConfig = { type: 'yolov8' };

      // Mock one success, one failure
      mockSupervisionClient.detectObjects
        .mockResolvedValueOnce([{ class: 'object', confidence: 0.9 }])
        .mockRejectedValueOnce(new Error('Processing failed'));

      mockSupervisionClient.generateAnnotations
        .mockResolvedValueOnce({})
        .mockResolvedValueOnce({});
      
      mockSupervisionClient.calculateMetrics
        .mockResolvedValueOnce({ overall_score: 0.85 }); // Above 0.7 threshold

      // Act
      const result = await supervisionService.validateBatch(imageBatch, modelConfig);

      // Assert - Verify failure handling
      expect(mockLogger.error).toHaveBeenCalledWith(
        'Image validation failed',
        { imageId: 'bad-img', error: 'Processing failed' }
      );

      expect(result.summary).toMatchObject({
        totalImages: 2,
        processedSuccessfully: 1,
        failed: 1,
        successRate: 0.5
      });

      expect(result.results).toHaveLength(2);
      // First result should succeed since we mocked good detection and quality metrics
      const firstResult = result.results[0];
      expect(firstResult.imageId).toBe('good-img');
      expect(firstResult.passed).toBe(true);
      expect(result.results[1]).toMatchObject({ 
        imageId: 'bad-img', 
        error: 'Processing failed', 
        passed: false 
      });
    });
  });

  describe('Real-Time Stream Validation Workflow', () => {
    it('should setup real-time stream validation and coordinate stream processing', async () => {
      // Arrange
      const streamConfig = {
        source: 'webcam',
        targetFps: 30,
        resolution: { width: 640, height: 480 }
      };
      
      const modelConfig = { type: 'yolov8', realTime: true };

      const mockStreamController = {
        start: jest.fn(),
        stop: jest.fn()
      };

      mockSupervisionClient.createStreamValidator.mockResolvedValue(mockStreamController);

      // Act
      const streamValidator = await supervisionService.validateRealTimeStream(
        streamConfig, 
        modelConfig
      );

      // Assert - Verify stream setup coordination
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Starting real-time stream validation',
        { streamSource: 'webcam', fps: 30 }
      );

      expect(mockSupervisionClient.createStreamValidator).toHaveBeenCalledWith(
        streamConfig,
        modelConfig
      );

      expect(mockStreamController.start).toHaveBeenCalledWith(
        expect.objectContaining({
          onFrameProcessed: expect.any(Function),
          onMetricsUpdate: expect.any(Function),
          onError: expect.any(Function)
        })
      );

      // Verify stream controller structure
      expect(streamValidator).toMatchObject({
        streamId: expect.stringMatching(/^stream-\d+$/),
        controller: mockStreamController,
        metrics: expect.objectContaining({
          framesProcessed: 0,
          isActive: true
        }),
        stop: expect.any(Function)
      });
    });

    it('should handle stream validation callbacks and coordinate metrics collection', async () => {
      // Arrange
      const streamConfig = { source: 'webcam', targetFps: 15 };
      const modelConfig = { type: 'yolov8' };

      const mockStreamController = {
        start: jest.fn((callbacks) => {
          // Simulate frame processing callback
          setTimeout(() => {
            callbacks.onFrameProcessed({
              frameId: 'frame-001',
              detections: [{ class: 'person', confidence: 0.9 }],
              processingTime: 33
            });
          }, 10);
        }),
        stop: jest.fn()
      };

      mockSupervisionClient.createStreamValidator.mockResolvedValue(mockStreamController);

      // Act
      const streamValidator = await supervisionService.validateRealTimeStream(
        streamConfig, 
        modelConfig
      );

      // Wait for callback execution
      await new Promise(resolve => setTimeout(resolve, 50));

      // Assert - Verify callback coordination
      expect(mockMetricsCollector.recordRealTimeFrame).toHaveBeenCalledWith(
        expect.objectContaining({
          frameId: 'frame-001',
          detections: expect.arrayContaining([
            expect.objectContaining({ class: 'person', confidence: 0.9 })
          ])
        })
      );

      expect(mockLogger.debug).toHaveBeenCalledWith(
        'Frame processed',
        expect.objectContaining({
          frameId: 'frame-001',
          detections: 1,
          processingTime: 33
        })
      );
    });

    it('should stop stream validation and record final metrics', async () => {
      // Arrange
      const streamConfig = { source: 'rtsp://stream' };
      const modelConfig = { type: 'yolov8' };

      const mockStreamController = {
        start: jest.fn(),
        stop: jest.fn()
      };

      mockSupervisionClient.createStreamValidator.mockResolvedValue(mockStreamController);

      const streamValidator = await supervisionService.validateRealTimeStream(
        streamConfig, 
        modelConfig
      );

      // Act
      await streamValidator.stop();

      // Assert - Verify stop coordination
      expect(mockStreamController.stop).toHaveBeenCalled();
      expect(mockMetricsCollector.recordValidation).toHaveBeenCalledWith(
        'real_time_stream',
        expect.objectContaining({
          streamId: expect.stringMatching(/^stream-\d+$/),
          finalMetrics: expect.objectContaining({
            isActive: false
          })
        })
      );
    });
  });

  describe('Annotation Quality Assessment Workflow', () => {
    it('should assess annotation quality and coordinate quality checks', async () => {
      // Arrange
      const annotations = {
        images: [
          {
            id: 'img-001',
            annotations: [
              { class: 'person', bbox: [100, 100, 200, 300], confidence: 0.9 }
            ]
          }
        ]
      };

      const groundTruth = {
        images: [
          {
            id: 'img-001',
            annotations: [
              { class: 'person', bbox: [105, 95, 195, 305], confidence: 1.0 }
            ]
          }
        ]
      };

      const mockQualityChecks = {
        overall_score: 0.88,
        metrics: { precision: 0.92, recall: 0.85, f1_score: 0.88 }
      };

      const mockConsistencyCheck = { score: 0.95, issues: [] };
      const mockCompletenessCheck = { score: 0.90, missing_annotations: 2 };

      mockSupervisionClient.runQualityChecks.mockResolvedValue(mockQualityChecks);
      mockSupervisionClient.checkAnnotationConsistency.mockResolvedValue(mockConsistencyCheck);
      mockSupervisionClient.checkAnnotationCompleteness.mockResolvedValue(mockCompletenessCheck);

      // Act
      const result = await supervisionService.assessAnnotationQuality(annotations, groundTruth);

      // Assert - Verify quality assessment coordination
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Assessing annotation quality with Supervision'
      );

      expect(mockSupervisionClient.runQualityChecks).toHaveBeenCalledWith(
        annotations,
        groundTruth,
        expect.objectContaining({
          metrics: ['precision', 'recall', 'f1_score'],
          iouThreshold: 0.5
        })
      );

      expect(mockSupervisionClient.checkAnnotationConsistency).toHaveBeenCalledWith(annotations);
      expect(mockSupervisionClient.checkAnnotationCompleteness).toHaveBeenCalledWith(
        annotations,
        groundTruth
      );

      // Verify result structure
      expect(result).toMatchObject({
        overall_score: 0.88,
        metrics: mockQualityChecks.metrics,
        consistency: mockConsistencyCheck,
        completeness: mockCompletenessCheck,
        recommendations: expect.any(Array),
        passed: true
      });
    });
  });

  describe('Performance Benchmark Workflow', () => {
    it('should run performance benchmarks and coordinate benchmark execution', async () => {
      // Arrange
      const benchmarkConfig = {
        modelType: 'yolov8',
        testDataset: '/path/to/benchmark/data',
        metrics: ['throughput', 'latency', 'accuracy'],
        iterations: 100
      };

      const mockBenchmarkSuite = {
        execute: jest.fn().mockResolvedValue({
          throughput: 45.2,
          latency: 22.1,
          accuracy: 0.91,
          resourceUsage: { cpu: 65, memory: 2048, gpu: 80 },
          passed: true
        })
      };

      mockSupervisionClient.createBenchmarkSuite.mockResolvedValue(mockBenchmarkSuite);

      // Act
      const result = await supervisionService.runPerformanceBenchmark(benchmarkConfig);

      // Assert - Verify benchmark coordination
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Running Supervision performance benchmark',
        benchmarkConfig
      );

      expect(mockSupervisionClient.createBenchmarkSuite).toHaveBeenCalledWith(benchmarkConfig);
      expect(mockBenchmarkSuite.execute).toHaveBeenCalled();

      expect(mockMetricsCollector.recordValidation).toHaveBeenCalledWith(
        'benchmark',
        expect.objectContaining({
          benchmarkId: expect.stringMatching(/^benchmark-\d+$/),
          config: benchmarkConfig,
          results: expect.objectContaining({
            throughput: 45.2,
            latency: 22.1,
            accuracy: 0.91
          }),
          passed: true
        })
      );

      // Verify benchmark result structure
      expect(result).toMatchObject({
        benchmarkId: expect.stringMatching(/^benchmark-\d+$/),
        config: benchmarkConfig,
        results: expect.objectContaining({
          throughput: 45.2,
          latency: 22.1,
          accuracy: 0.91,
          resourceUsage: expect.objectContaining({
            cpu: 65,
            memory: 2048,
            gpu: 80
          })
        }),
        passed: true,
        recommendations: expect.any(Array)
      });
    });
  });

  describe('Validation Report Generation Workflow', () => {
    it('should generate comprehensive validation report and coordinate report creation', async () => {
      // Arrange
      const validationResults = [
        { 
          validationType: 'single_image', 
          passed: true, 
          quality: { overall_score: 0.92 } 
        },
        { 
          validationType: 'batch_processing', 
          passed: true, 
          quality: { overall_score: 0.88 } 
        },
        { 
          validationType: 'real_time_stream', 
          passed: false, 
          quality: { overall_score: 0.65 } 
        }
      ];

      const mockReportData = {
        visualizations: ['detection_heatmap.png', 'accuracy_trend.png'],
        charts: ['performance_metrics.json']
      };

      const mockExportPaths = {
        json: '/reports/supervision-report.json',
        html: '/reports/supervision-report.html',
        pdf: '/reports/supervision-report.pdf'
      };

      mockSupervisionClient.generateReport.mockResolvedValue(mockReportData);
      mockSupervisionClient.exportReport.mockResolvedValue(mockExportPaths);

      // Act
      const report = await supervisionService.generateValidationReport(validationResults);

      // Assert - Verify report generation coordination
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Generating comprehensive Supervision validation report'
      );

      expect(mockSupervisionClient.generateReport).toHaveBeenCalledWith(validationResults);
      expect(mockSupervisionClient.exportReport).toHaveBeenCalledWith(
        mockReportData,
        expect.objectContaining({
          formats: ['json', 'html', 'pdf'],
          includeVisualizations: true
        })
      );

      // Verify report structure
      expect(report).toMatchObject({
        reportId: expect.stringMatching(/^supervision-report-\d+$/),
        timestamp: expect.any(Number),
        summary: {
          totalValidations: 3,
          passedValidations: 2,
          averageQualityScore: expect.any(Number),
          validationTypes: {
            single_image: 1,
            batch_processing: 1,
            real_time_stream: 1
          }
        },
        detailedResults: validationResults,
        visualizations: mockReportData.visualizations,
        recommendations: expect.any(Array),
        exportPaths: mockExportPaths
      });
    });
  });

  describe('Service Contract Compliance', () => {
    it('should satisfy SupervisionValidationService contract', () => {
      const expectedContract = {
        validateSingleImage: expect.any(Function),
        validateBatch: expect.any(Function),
        validateRealTimeStream: expect.any(Function),
        assessAnnotationQuality: expect.any(Function),
        runPerformanceBenchmark: expect.any(Function),
        generateValidationReport: expect.any(Function)
      };

      expect(supervisionService).toSatisfyContract(expectedContract);
    });

    it('should have proper default configuration', () => {
      const defaultService = new SupervisionValidationService({
        supervisionClient: mockSupervisionClient,
        logger: mockLogger,
        metricsCollector: mockMetricsCollector
      });

      expect(defaultService.validationConfig).toMatchObject({
        confidenceThreshold: 0.5,
        iouThreshold: 0.5,
        maxDetections: 100,
        annotationFormats: expect.arrayContaining(['coco', 'yolo', 'pascal_voc']),
        qualityMetrics: expect.arrayContaining(['precision', 'recall', 'f1_score']),
        validationModes: expect.arrayContaining(['single_image', 'batch_processing', 'real_time'])
      });
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle missing ground truth gracefully', async () => {
      // Arrange
      const imageData = { id: 'no-gt-image', path: '/image.jpg' };
      const modelConfig = { type: 'yolov8' };

      mockSupervisionClient.detectObjects.mockResolvedValue([]);
      mockSupervisionClient.generateAnnotations.mockResolvedValue({});

      // Act
      const result = await supervisionService.validateSingleImage(imageData, modelConfig);

      // Assert
      expect(result.quality).toMatchObject({
        overall_score: 0.5,
        note: 'No ground truth available'
      });
    });

    it('should handle empty detection results', async () => {
      // Arrange
      const imageData = { id: 'empty-detections', path: '/image.jpg' };
      const modelConfig = { type: 'yolov8' };

      mockSupervisionClient.detectObjects.mockResolvedValue([]);
      mockSupervisionClient.generateAnnotations.mockResolvedValue({});
      mockSupervisionClient.calculateMetrics.mockResolvedValue({
        overall_score: 0.0,
        note: 'No objects detected'
      });

      // Act
      const result = await supervisionService.validateSingleImage(imageData, modelConfig);

      // Assert
      expect(result.detections.count).toBe(0);
      expect(result.detections.confidence).toBe(0);
      expect(result.passed).toBe(false);
    });
  });
});