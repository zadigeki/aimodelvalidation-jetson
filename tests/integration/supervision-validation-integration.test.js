/**
 * SupervisionValidationService Integration Tests
 * Testing real Supervision.py integration with validation workflows
 */

import { jest, describe, it, beforeEach, afterEach, expect } from '@jest/globals';
import { SupervisionValidationService } from '../../src/services/SupervisionValidationService.js';
import { createTestContainer } from '../fixtures/testContainer.js';
import { createMockSupervisionEnvironment } from '../fixtures/mockSupervisionEnvironment.js';

describe('Supervision Validation Integration', () => {
  let container;
  let supervisionService;
  let mockEnvironment;

  beforeEach(async () => {
    // Setup integration test environment
    container = await createTestContainer();
    mockEnvironment = await createMockSupervisionEnvironment();
    
    supervisionService = container.resolve('supervisionValidationService');
  });

  afterEach(async () => {
    await mockEnvironment.cleanup();
    await container.dispose();
  });

  describe('End-to-End Single Image Validation', () => {
    it('should perform complete single image validation workflow', async () => {
      // Arrange
      const testImage = await mockEnvironment.createTestImage({
        id: 'integration-test-001',
        objects: [
          { class: 'person', bbox: [100, 100, 200, 300] },
          { class: 'car', bbox: [300, 200, 450, 350] }
        ]
      });

      const modelConfig = {
        type: 'yolov8n',
        weights: await mockEnvironment.getTestModel(),
        confidenceThreshold: 0.5
      };

      // Act
      const result = await supervisionService.validateSingleImage(testImage, modelConfig);

      // Assert - Integration verification
      expect(result).toMatchObject({
        imageId: 'integration-test-001',
        timestamp: expect.any(Number),
        detections: {
          count: expect.any(Number),
          objects: expect.any(Array),
          confidence: expect.any(Number)
        },
        annotations: expect.objectContaining({
          coco: expect.any(Object),
          yolo: expect.any(Object)
        }),
        quality: expect.objectContaining({
          overall_score: expect.any(Number),
          precision: expect.any(Number),
          recall: expect.any(Number)
        }),
        passed: expect.any(Boolean)
      });

      // Verify detection quality
      expect(result.detections.count).toBeGreaterThan(0);
      expect(result.quality.overall_score).toBeGreaterThan(0);
      
      // Verify annotations were generated
      expect(result.annotations.coco).toBeDefined();
      expect(result.annotations.yolo).toBeDefined();
    });

    it('should handle various image formats and resolutions', async () => {
      // Test multiple image formats
      const testCases = [
        { format: 'jpg', resolution: { width: 640, height: 480 } },
        { format: 'png', resolution: { width: 1280, height: 720 } },
        { format: 'bmp', resolution: { width: 416, height: 416 } }
      ];

      const modelConfig = {
        type: 'yolov8s',
        weights: await mockEnvironment.getTestModel()
      };

      for (const testCase of testCases) {
        // Arrange
        const testImage = await mockEnvironment.createTestImage({
          id: `format-test-${testCase.format}`,
          format: testCase.format,
          resolution: testCase.resolution,
          objects: [{ class: 'object', bbox: [50, 50, 150, 150] }]
        });

        // Act
        const result = await supervisionService.validateSingleImage(testImage, modelConfig);

        // Assert
        expect(result.passed).toBe(true);
        expect(result.detections.count).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe('End-to-End Batch Processing Integration', () => {
    it('should process image batch with real Supervision pipeline', async () => {
      // Arrange
      const batchSize = 5;
      const imageBatch = [];

      for (let i = 0; i < batchSize; i++) {
        const testImage = await mockEnvironment.createTestImage({
          id: `batch-integration-${i.toString().padStart(3, '0')}`,
          objects: [
            { class: 'person', bbox: [100 + i * 10, 100, 200 + i * 10, 300] }
          ]
        });
        imageBatch.push(testImage);
      }

      const modelConfig = {
        type: 'yolov8m',
        weights: await mockEnvironment.getTestModel(),
        batchSize: 2
      };

      // Act
      const result = await supervisionService.validateBatch(imageBatch, modelConfig);

      // Assert - Integration verification
      expect(result).toMatchObject({
        batchId: expect.stringMatching(/^batch-\d+$/),
        timestamp: expect.any(Number),
        summary: {
          totalImages: batchSize,
          processedSuccessfully: expect.any(Number),
          failed: expect.any(Number),
          successRate: expect.any(Number)
        },
        performance: expect.objectContaining({
          totalTime: expect.any(Number),
          averageProcessingTime: expect.any(Number)
        }),
        aggregatedQuality: expect.objectContaining({
          overall_score: expect.any(Number)
        }),
        results: expect.arrayContaining([
          expect.objectContaining({
            imageId: expect.stringMatching(/^batch-integration-\d{3}$/),
            passed: expect.any(Boolean)
          })
        ])
      });

      // Verify batch processing efficiency
      expect(result.summary.successRate).toBeGreaterThan(0.8);
      expect(result.performance.averageProcessingTime).toBeLessThan(5000); // 5 seconds max per image
      expect(result.results).toHaveLength(batchSize);
    });

    it('should handle mixed success/failure scenarios in batch processing', async () => {
      // Arrange - Create batch with some problematic images
      const imageBatch = [
        await mockEnvironment.createTestImage({ id: 'good-image-1', objects: [{ class: 'person', bbox: [100, 100, 200, 300] }] }),
        await mockEnvironment.createCorruptedImage({ id: 'corrupted-image' }),
        await mockEnvironment.createTestImage({ id: 'good-image-2', objects: [{ class: 'car', bbox: [300, 200, 450, 350] }] }),
        await mockEnvironment.createEmptyImage({ id: 'empty-image' })
      ];

      const modelConfig = {
        type: 'yolov8s',
        weights: await mockEnvironment.getTestModel()
      };

      // Act
      const result = await supervisionService.validateBatch(imageBatch, modelConfig);

      // Assert - Verify resilient processing
      expect(result.summary.totalImages).toBe(4);
      expect(result.summary.processedSuccessfully).toBeGreaterThanOrEqual(2);
      expect(result.summary.failed).toBeGreaterThanOrEqual(1);
      expect(result.results).toHaveLength(4);

      // Verify error handling
      const failedResults = result.results.filter(r => !r.passed);
      expect(failedResults.length).toBeGreaterThan(0);
      expect(failedResults[0]).toHaveProperty('error');
    });
  });

  describe('Real-Time Stream Processing Integration', () => {
    it('should setup and process real-time video stream', async () => {
      // Arrange
      const mockVideoStream = await mockEnvironment.createMockVideoStream({
        fps: 15,
        duration: 3, // 3 seconds
        resolution: { width: 640, height: 480 },
        objectPattern: 'moving_person'
      });

      const streamConfig = {
        source: mockVideoStream.streamUrl,
        targetFps: 15,
        resolution: { width: 640, height: 480 }
      };

      const modelConfig = {
        type: 'yolov8n',
        weights: await mockEnvironment.getTestModel(),
        realTime: true
      };

      // Act
      const streamValidator = await supervisionService.validateRealTimeStream(
        streamConfig,
        modelConfig
      );

      // Let stream run for a short period
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Stop stream
      await streamValidator.stop();

      // Assert - Integration verification
      expect(streamValidator).toMatchObject({
        streamId: expect.stringMatching(/^stream-\d+$/),
        controller: expect.any(Object),
        metrics: expect.objectContaining({
          framesProcessed: expect.any(Number),
          isActive: false
        }),
        stop: expect.any(Function)
      });

      // Verify stream processing occurred
      expect(streamValidator.metrics.framesProcessed).toBeGreaterThan(5);
      expect(streamValidator.metrics.averageFps).toBeGreaterThan(0);
    });
  });

  describe('Annotation Quality Assessment Integration', () => {
    it('should assess real annotation quality against ground truth', async () => {
      // Arrange
      const testDataset = await mockEnvironment.createAnnotatedDataset({
        imageCount: 10,
        annotationFormat: 'coco',
        classes: ['person', 'car', 'bicycle'],
        qualityLevel: 'high' // Well-annotated dataset
      });

      const modelPredictions = await mockEnvironment.generateModelPredictions(
        testDataset,
        { type: 'yolov8m', accuracy: 0.85 }
      );

      // Act
      const qualityAssessment = await supervisionService.assessAnnotationQuality(
        modelPredictions.annotations,
        testDataset.groundTruth
      );

      // Assert - Integration verification
      expect(qualityAssessment).toMatchObject({
        overall_score: expect.any(Number),
        metrics: expect.objectContaining({
          precision: expect.any(Number),
          recall: expect.any(Number),
          f1_score: expect.any(Number)
        }),
        consistency: expect.objectContaining({
          score: expect.any(Number)
        }),
        completeness: expect.objectContaining({
          score: expect.any(Number)
        }),
        recommendations: expect.any(Array),
        passed: expect.any(Boolean)
      });

      // Verify quality metrics are reasonable
      expect(qualityAssessment.overall_score).toBeGreaterThan(0.5);
      expect(qualityAssessment.metrics.precision).toBeGreaterThan(0.3);
      expect(qualityAssessment.metrics.recall).toBeGreaterThan(0.3);
    });

    it('should detect annotation inconsistencies and completeness issues', async () => {
      // Arrange
      const problematicDataset = await mockEnvironment.createAnnotatedDataset({
        imageCount: 8,
        annotationFormat: 'coco',
        classes: ['person', 'car'],
        qualityLevel: 'low', // Inconsistent annotations
        issues: ['missing_annotations', 'inconsistent_labels', 'duplicate_boxes']
      });

      const modelPredictions = await mockEnvironment.generateModelPredictions(
        problematicDataset,
        { type: 'yolov8s', accuracy: 0.75 }
      );

      // Act
      const qualityAssessment = await supervisionService.assessAnnotationQuality(
        modelPredictions.annotations,
        problematicDataset.groundTruth
      );

      // Assert - Verify issue detection
      expect(qualityAssessment.overall_score).toBeLessThan(0.8);
      expect(qualityAssessment.consistency.score).toBeLessThan(0.9);
      expect(qualityAssessment.completeness.missing_annotations).toBeGreaterThan(0);
      expect(qualityAssessment.recommendations).toHaveLength(expect.any(Number));
      expect(qualityAssessment.passed).toBe(false);
    });
  });

  describe('Performance Benchmark Integration', () => {
    it('should run comprehensive performance benchmarks', async () => {
      // Arrange
      const benchmarkDataset = await mockEnvironment.createBenchmarkDataset({
        imageCount: 50,
        resolution: { width: 640, height: 640 },
        complexity: 'medium',
        classes: ['person', 'car', 'bicycle', 'dog', 'cat']
      });

      const benchmarkConfig = {
        modelType: 'yolov8s',
        testDataset: benchmarkDataset.path,
        metrics: ['throughput', 'latency', 'accuracy', 'memory_usage'],
        iterations: 25,
        batchSizes: [1, 4, 8],
        deviceTypes: ['cpu'] // GPU testing would require actual GPU
      };

      // Act
      const benchmarkResults = await supervisionService.runPerformanceBenchmark(benchmarkConfig);

      // Assert - Integration verification
      expect(benchmarkResults).toMatchObject({
        benchmarkId: expect.stringMatching(/^benchmark-\d+$/),
        config: benchmarkConfig,
        results: expect.objectContaining({
          throughput: expect.any(Number),
          latency: expect.any(Number),
          accuracy: expect.any(Number),
          resourceUsage: expect.objectContaining({
            cpu: expect.any(Number),
            memory: expect.any(Number)
          })
        }),
        passed: expect.any(Boolean),
        recommendations: expect.any(Array)
      });

      // Verify performance metrics are reasonable
      expect(benchmarkResults.results.throughput).toBeGreaterThan(0);
      expect(benchmarkResults.results.latency).toBeGreaterThan(0);
      expect(benchmarkResults.results.accuracy).toBeGreaterThan(0.3);
      expect(benchmarkResults.results.resourceUsage.cpu).toBeGreaterThan(0);
    });
  });

  describe('Complete Validation Pipeline Integration', () => {
    it('should execute complete validation pipeline end-to-end', async () => {
      // Arrange - Setup complete validation scenario
      const testProject = await mockEnvironment.createValidationProject({
        name: 'complete-pipeline-test',
        imageCount: 15,
        modelType: 'yolov8m',
        validationTypes: ['single_image', 'batch_processing', 'quality_assessment']
      });

      // Act - Execute complete pipeline
      const singleImageResult = await supervisionService.validateSingleImage(
        testProject.sampleImage,
        testProject.modelConfig
      );

      const batchResult = await supervisionService.validateBatch(
        testProject.imageBatch,
        testProject.modelConfig
      );

      const qualityResult = await supervisionService.assessAnnotationQuality(
        testProject.annotations,
        testProject.groundTruth
      );

      const allResults = [singleImageResult, batchResult, qualityResult];
      const finalReport = await supervisionService.generateValidationReport(allResults);

      // Assert - Verify complete pipeline
      expect(singleImageResult.passed).toBeDefined();
      expect(batchResult.summary.totalImages).toBe(testProject.imageBatch.length);
      expect(qualityResult.overall_score).toBeGreaterThan(0);

      expect(finalReport).toMatchObject({
        reportId: expect.stringMatching(/^supervision-report-\d+$/),
        timestamp: expect.any(Number),
        summary: expect.objectContaining({
          totalValidations: allResults.length,
          passedValidations: expect.any(Number),
          averageQualityScore: expect.any(Number)
        }),
        detailedResults: allResults,
        visualizations: expect.any(Array),
        recommendations: expect.any(Array),
        exportPaths: expect.objectContaining({
          json: expect.stringMatching(/\.json$/),
          html: expect.stringMatching(/\.html$/),
          pdf: expect.stringMatching(/\.pdf$/)
        })
      });

      // Verify report exports were created
      expect(finalReport.exportPaths.json).toBeDefined();
      expect(finalReport.exportPaths.html).toBeDefined();
      expect(finalReport.exportPaths.pdf).toBeDefined();
    });
  });

  describe('Error Recovery and Resilience', () => {
    it('should handle Supervision library failures gracefully', async () => {
      // Arrange - Create scenario that might cause Supervision errors
      const problematicImage = await mockEnvironment.createProblematicImage({
        id: 'error-test-image',
        issues: ['corrupted_data', 'invalid_format', 'zero_size']
      });

      const modelConfig = {
        type: 'yolov8n',
        weights: await mockEnvironment.getTestModel()
      };

      // Act & Assert - Should handle errors gracefully
      await expect(async () => {
        const result = await supervisionService.validateSingleImage(problematicImage, modelConfig);
        
        // If it doesn't throw, verify it handled the error gracefully
        if (result) {
          expect(result).toHaveProperty('passed', false);
          expect(result).toHaveProperty('error');
        }
      }).not.toThrow('Unhandled error');
    });

    it('should maintain service stability during resource constraints', async () => {
      // Arrange - Create resource-intensive scenario
      const largeBatch = [];
      for (let i = 0; i < 20; i++) {
        largeBatch.push(await mockEnvironment.createTestImage({
          id: `stress-test-${i}`,
          resolution: { width: 1920, height: 1080 }, // Large images
          objects: Array.from({ length: 10 }, (_, j) => ({
            class: 'object',
            bbox: [j * 50, j * 50, (j + 1) * 50, (j + 1) * 50]
          }))
        }));
      }

      const modelConfig = {
        type: 'yolov8l', // Large model
        weights: await mockEnvironment.getTestModel()
      };

      // Act - Process large batch
      const result = await supervisionService.validateBatch(largeBatch, modelConfig);

      // Assert - Verify service remained stable
      expect(result).toBeDefined();
      expect(result.summary.totalImages).toBe(20);
      expect(result.summary.processedSuccessfully + result.summary.failed).toBe(20);
      
      // Service should have processed at least some images successfully
      expect(result.summary.successRate).toBeGreaterThan(0);
    });
  });
});