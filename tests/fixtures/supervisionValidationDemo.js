/**
 * Supervision Validation Demo - Interactive demonstration scenarios
 * Showcases different validation workflows and quality assessment capabilities
 */

export class SupervisionValidationDemo {
  constructor(supervisionService, logger) {
    this.supervisionService = supervisionService;
    this.logger = logger;
    this.demoResults = [];
  }

  /**
   * Demo Scenario 1: Single Image Object Detection Validation
   */
  async demonstrateSingleImageValidation() {
    this.logger.info('🎯 Demo: Single Image Object Detection Validation');
    
    try {
      // Create demo image with known objects
      const demoImage = {
        id: 'demo-single-image-001',
        path: '/demo/images/street_scene.jpg',
        metadata: {
          width: 640,
          height: 480,
          format: 'jpeg',
          source: 'webcam_capture'
        },
        groundTruth: [
          { class: 'person', bbox: [150, 100, 250, 400], confidence: 1.0 },
          { class: 'car', bbox: [300, 200, 500, 350], confidence: 1.0 },
          { class: 'bicycle', bbox: [50, 250, 120, 380], confidence: 1.0 }
        ]
      };

      const modelConfig = {
        type: 'yolov8n',
        weights: '/models/yolov8n.pt',
        confidenceThreshold: 0.5,
        iouThreshold: 0.45
      };

      this.logger.info('📸 Processing single image with YOLOv8...');
      const result = await this.supervisionService.validateSingleImage(demoImage, modelConfig);

      this.logger.info('✅ Single Image Validation Results:');
      this.logger.info(`   • Image ID: ${result.imageId}`);
      this.logger.info(`   • Objects Detected: ${result.detections.count}`);
      this.logger.info(`   • Average Confidence: ${result.detections.confidence.toFixed(3)}`);
      this.logger.info(`   • Quality Score: ${result.quality.overall_score.toFixed(3)}`);
      this.logger.info(`   • Validation: ${result.passed ? 'PASSED' : 'FAILED'}`);

      if (result.detections.objects.length > 0) {
        this.logger.info('   • Detected Objects:');
        result.detections.objects.forEach((obj, idx) => {
          this.logger.info(`     ${idx + 1}. ${obj.class} (${obj.confidence.toFixed(3)})`);
        });
      }

      this.demoResults.push({
        scenario: 'single_image_validation',
        result,
        demoNotes: 'Demonstrates basic object detection validation with quality metrics'
      });

      return result;

    } catch (error) {
      this.logger.error('❌ Single image validation demo failed:', error.message);
      throw error;
    }
  }

  /**
   * Demo Scenario 2: Batch Processing with Performance Metrics
   */
  async demonstrateBatchProcessing() {
    this.logger.info('🎯 Demo: Batch Processing with Performance Analysis');

    try {
      // Create batch of demo images
      const imageBatch = [
        {
          id: 'demo-batch-001',
          path: '/demo/batch/traffic_001.jpg',
          groundTruth: [
            { class: 'car', bbox: [100, 150, 300, 250], confidence: 1.0 },
            { class: 'person', bbox: [350, 100, 400, 300], confidence: 1.0 }
          ]
        },
        {
          id: 'demo-batch-002',
          path: '/demo/batch/traffic_002.jpg',
          groundTruth: [
            { class: 'car', bbox: [200, 180, 450, 280], confidence: 1.0 },
            { class: 'traffic_light', bbox: [500, 50, 550, 150], confidence: 1.0 }
          ]
        },
        {
          id: 'demo-batch-003',
          path: '/demo/batch/pedestrian_001.jpg',
          groundTruth: [
            { class: 'person', bbox: [150, 100, 250, 400], confidence: 1.0 },
            { class: 'person', bbox: [300, 120, 380, 380], confidence: 1.0 }
          ]
        },
        {
          id: 'demo-batch-004',
          path: '/demo/batch/parking_001.jpg',
          groundTruth: [
            { class: 'car', bbox: [50, 200, 200, 300], confidence: 1.0 },
            { class: 'car', bbox: [250, 180, 400, 280], confidence: 1.0 },
            { class: 'car', bbox: [450, 190, 600, 290], confidence: 1.0 }
          ]
        }
      ];

      const modelConfig = {
        type: 'yolov8s',
        weights: '/models/yolov8s.pt',
        batchSize: 2,
        confidenceThreshold: 0.6
      };

      this.logger.info(`📦 Processing batch of ${imageBatch.length} images...`);
      const startTime = Date.now();
      
      const result = await this.supervisionService.validateBatch(imageBatch, modelConfig);
      
      const totalTime = Date.now() - startTime;

      this.logger.info('✅ Batch Processing Results:');
      this.logger.info(`   • Total Images: ${result.summary.totalImages}`);
      this.logger.info(`   • Successfully Processed: ${result.summary.processedSuccessfully}`);
      this.logger.info(`   • Failed: ${result.summary.failed}`);
      this.logger.info(`   • Success Rate: ${(result.summary.successRate * 100).toFixed(1)}%`);
      this.logger.info(`   • Total Processing Time: ${totalTime}ms`);
      this.logger.info(`   • Average per Image: ${result.performance.averageProcessingTime.toFixed(0)}ms`);
      this.logger.info(`   • Overall Quality Score: ${result.aggregatedQuality.overall_score.toFixed(3)}`);

      // Show per-image results
      this.logger.info('   • Per-Image Results:');
      result.results.forEach((imgResult, idx) => {
        if (imgResult.passed) {
          this.logger.info(`     ${idx + 1}. ${imgResult.imageId}: ✅ ${imgResult.detections.count} objects (${imgResult.quality.overall_score.toFixed(3)})`);
        } else {
          this.logger.info(`     ${idx + 1}. ${imgResult.imageId}: ❌ ${imgResult.error || 'Validation failed'}`);
        }
      });

      this.demoResults.push({
        scenario: 'batch_processing',
        result,
        demoNotes: 'Demonstrates batch processing efficiency and aggregated quality metrics'
      });

      return result;

    } catch (error) {
      this.logger.error('❌ Batch processing demo failed:', error.message);
      throw error;
    }
  }

  /**
   * Demo Scenario 3: Real-Time Stream Validation
   */
  async demonstrateRealTimeValidation() {
    this.logger.info('🎯 Demo: Real-Time Stream Validation');

    try {
      const streamConfig = {
        source: 'demo_webcam',
        targetFps: 15,
        resolution: { width: 640, height: 480 },
        bufferSize: 30
      };

      const modelConfig = {
        type: 'yolov8n',
        weights: '/models/yolov8n.pt',
        realTime: true,
        confidenceThreshold: 0.4
      };

      this.logger.info('📹 Starting real-time stream validation...');
      this.logger.info(`   • Target FPS: ${streamConfig.targetFps}`);
      this.logger.info(`   • Resolution: ${streamConfig.resolution.width}x${streamConfig.resolution.height}`);

      const streamValidator = await this.supervisionService.validateRealTimeStream(
        streamConfig,
        modelConfig
      );

      // Let it run for demo period
      this.logger.info('⏱️  Running stream for 10 seconds...');
      await new Promise(resolve => setTimeout(resolve, 10000));

      // Stop and get final metrics
      await streamValidator.stop();

      this.logger.info('✅ Real-Time Stream Results:');
      this.logger.info(`   • Stream ID: ${streamValidator.streamId}`);
      this.logger.info(`   • Frames Processed: ${streamValidator.metrics.framesProcessed}`);
      this.logger.info(`   • Average FPS: ${streamValidator.metrics.averageFps.toFixed(1)}`);
      this.logger.info(`   • Average Latency: ${streamValidator.metrics.latencyMs.toFixed(1)}ms`);
      this.logger.info(`   • Detection Accuracy: ${(streamValidator.metrics.detectionAccuracy * 100).toFixed(1)}%`);

      this.demoResults.push({
        scenario: 'real_time_stream',
        result: {
          streamId: streamValidator.streamId,
          finalMetrics: streamValidator.metrics
        },
        demoNotes: 'Demonstrates real-time processing capabilities with FPS and latency metrics'
      });

      return streamValidator;

    } catch (error) {
      this.logger.error('❌ Real-time stream demo failed:', error.message);
      throw error;
    }
  }

  /**
   * Demo Scenario 4: Annotation Quality Assessment
   */
  async demonstrateQualityAssessment() {
    this.logger.info('🎯 Demo: Annotation Quality Assessment');

    try {
      // Demo annotations with deliberate quality variations
      const annotations = {
        info: {
          description: 'Demo annotation quality assessment',
          version: '1.0',
          year: 2024
        },
        images: [
          {
            id: 'quality-demo-001',
            file_name: 'quality_test_001.jpg',
            width: 640,
            height: 480
          }
        ],
        annotations: [
          // High quality annotation
          {
            id: 1,
            image_id: 'quality-demo-001',
            category_id: 1,
            bbox: [100, 100, 150, 200],
            area: 30000,
            segmentation: [],
            iscrowd: 0
          },
          // Lower quality annotation (slightly off)
          {
            id: 2,
            image_id: 'quality-demo-001',
            category_id: 2,
            bbox: [300, 150, 120, 180],
            area: 21600,
            segmentation: [],
            iscrowd: 0
          }
        ],
        categories: [
          { id: 1, name: 'person', supercategory: 'person' },
          { id: 2, name: 'car', supercategory: 'vehicle' }
        ]
      };

      const groundTruth = {
        info: {
          description: 'Ground truth for quality assessment',
          version: '1.0'
        },
        images: [
          {
            id: 'quality-demo-001',
            file_name: 'quality_test_001.jpg',
            width: 640,
            height: 480
          }
        ],
        annotations: [
          // Perfect ground truth
          {
            id: 1,
            image_id: 'quality-demo-001',
            category_id: 1,
            bbox: [98, 102, 152, 198],
            area: 30096,
            segmentation: [],
            iscrowd: 0
          },
          {
            id: 2,
            image_id: 'quality-demo-001',
            category_id: 2,
            bbox: [305, 145, 125, 185],
            area: 23125,
            segmentation: [],
            iscrowd: 0
          }
        ],
        categories: [
          { id: 1, name: 'person', supercategory: 'person' },
          { id: 2, name: 'car', supercategory: 'vehicle' }
        ]
      };

      this.logger.info('🔍 Assessing annotation quality...');
      const qualityResult = await this.supervisionService.assessAnnotationQuality(
        annotations,
        groundTruth
      );

      this.logger.info('✅ Annotation Quality Assessment Results:');
      this.logger.info(`   • Overall Score: ${qualityResult.overall_score.toFixed(3)}`);
      this.logger.info(`   • Precision: ${qualityResult.metrics.precision.toFixed(3)}`);
      this.logger.info(`   • Recall: ${qualityResult.metrics.recall.toFixed(3)}`);
      this.logger.info(`   • F1 Score: ${qualityResult.metrics.f1_score.toFixed(3)}`);
      this.logger.info(`   • Consistency Score: ${qualityResult.consistency.score.toFixed(3)}`);
      this.logger.info(`   • Completeness Score: ${qualityResult.completeness.score.toFixed(3)}`);
      this.logger.info(`   • Assessment: ${qualityResult.passed ? 'PASSED' : 'NEEDS IMPROVEMENT'}`);

      if (qualityResult.recommendations.length > 0) {
        this.logger.info('   • Recommendations:');
        qualityResult.recommendations.forEach((rec, idx) => {
          this.logger.info(`     ${idx + 1}. ${rec}`);
        });
      }

      this.demoResults.push({
        scenario: 'quality_assessment',
        result: qualityResult,
        demoNotes: 'Demonstrates annotation quality evaluation with precision/recall metrics'
      });

      return qualityResult;

    } catch (error) {
      this.logger.error('❌ Quality assessment demo failed:', error.message);
      throw error;
    }
  }

  /**
   * Demo Scenario 5: Performance Benchmarking
   */
  async demonstratePerformanceBenchmark() {
    this.logger.info('🎯 Demo: Performance Benchmarking');

    try {
      const benchmarkConfig = {
        modelType: 'yolov8s',
        testDataset: '/demo/benchmark_dataset',
        metrics: ['throughput', 'latency', 'accuracy', 'memory_usage'],
        iterations: 25,
        batchSizes: [1, 2, 4],
        imageResolutions: [
          { width: 416, height: 416 },
          { width: 640, height: 640 }
        ],
        deviceTypes: ['cpu'],
        warmupRuns: 5
      };

      this.logger.info('🏃 Running performance benchmark...');
      this.logger.info(`   • Model: ${benchmarkConfig.modelType}`);
      this.logger.info(`   • Iterations: ${benchmarkConfig.iterations}`);
      this.logger.info(`   • Batch Sizes: [${benchmarkConfig.batchSizes.join(', ')}]`);

      const benchmarkResult = await this.supervisionService.runPerformanceBenchmark(
        benchmarkConfig
      );

      this.logger.info('✅ Performance Benchmark Results:');
      this.logger.info(`   • Benchmark ID: ${benchmarkResult.benchmarkId}`);
      this.logger.info(`   • Throughput: ${benchmarkResult.results.throughput.toFixed(1)} FPS`);
      this.logger.info(`   • Average Latency: ${benchmarkResult.results.latency.toFixed(1)}ms`);
      this.logger.info(`   • Accuracy: ${(benchmarkResult.results.accuracy * 100).toFixed(1)}%`);
      this.logger.info(`   • CPU Usage: ${benchmarkResult.results.resourceUsage.cpu.toFixed(1)}%`);
      this.logger.info(`   • Memory Usage: ${benchmarkResult.results.resourceUsage.memory}MB`);
      this.logger.info(`   • Benchmark: ${benchmarkResult.passed ? 'PASSED' : 'NEEDS OPTIMIZATION'}`);

      if (benchmarkResult.recommendations.length > 0) {
        this.logger.info('   • Performance Recommendations:');
        benchmarkResult.recommendations.forEach((rec, idx) => {
          this.logger.info(`     ${idx + 1}. ${rec}`);
        });
      }

      this.demoResults.push({
        scenario: 'performance_benchmark',
        result: benchmarkResult,
        demoNotes: 'Demonstrates performance profiling with throughput and resource usage metrics'
      });

      return benchmarkResult;

    } catch (error) {
      this.logger.error('❌ Performance benchmark demo failed:', error.message);
      throw error;
    }
  }

  /**
   * Generate comprehensive demo report
   */
  async generateDemoReport() {
    this.logger.info('📊 Generating comprehensive demo report...');

    try {
      const report = await this.supervisionService.generateValidationReport(
        this.demoResults.map(demo => demo.result)
      );

      // Enhance report with demo-specific information
      const enhancedReport = {
        ...report,
        demoInfo: {
          title: 'Supervision Validation Service Demo Report',
          timestamp: new Date().toISOString(),
          scenarios: this.demoResults.map(demo => ({
            scenario: demo.scenario,
            description: demo.demoNotes,
            status: demo.result.passed !== false ? 'SUCCESS' : 'NEEDS_ATTENTION'
          })),
          summary: {
            totalScenarios: this.demoResults.length,
            successfulScenarios: this.demoResults.filter(demo => 
              demo.result.passed !== false
            ).length,
            demonstratedCapabilities: [
              'Single image object detection',
              'Batch processing with performance metrics',
              'Real-time stream validation',
              'Annotation quality assessment',
              'Performance benchmarking'
            ]
          }
        }
      };

      this.logger.info('✅ Demo Report Generated:');
      this.logger.info(`   • Report ID: ${enhancedReport.reportId}`);
      this.logger.info(`   • Total Scenarios: ${enhancedReport.demoInfo.summary.totalScenarios}`);
      this.logger.info(`   • Successful: ${enhancedReport.demoInfo.summary.successfulScenarios}`);
      this.logger.info(`   • Export Formats: JSON, HTML, PDF`);

      if (enhancedReport.exportPaths) {
        this.logger.info('   • Export Paths:');
        Object.entries(enhancedReport.exportPaths).forEach(([format, path]) => {
          this.logger.info(`     • ${format.toUpperCase()}: ${path}`);
        });
      }

      return enhancedReport;

    } catch (error) {
      this.logger.error('❌ Demo report generation failed:', error.message);
      throw error;
    }
  }

  /**
   * Run complete demo workflow
   */
  async runCompleteDemo() {
    this.logger.info('🚀 Starting Complete Supervision Validation Demo');
    this.logger.info('=' * 60);

    try {
      // Clear previous results
      this.demoResults = [];

      // Run all demo scenarios
      await this.demonstrateSingleImageValidation();
      this.logger.info('');

      await this.demonstrateBatchProcessing();
      this.logger.info('');

      await this.demonstrateRealTimeValidation();
      this.logger.info('');

      await this.demonstrateQualityAssessment();
      this.logger.info('');

      await this.demonstratePerformanceBenchmark();
      this.logger.info('');

      // Generate final report
      const finalReport = await this.generateDemoReport();

      this.logger.info('');
      this.logger.info('🎉 Complete Demo Finished Successfully!');
      this.logger.info('=' * 60);
      this.logger.info('All Supervision validation capabilities demonstrated.');
      this.logger.info(`Check ${finalReport.exportPaths?.html || 'the generated report'} for detailed results.`);

      return {
        success: true,
        demoResults: this.demoResults,
        finalReport
      };

    } catch (error) {
      this.logger.error('❌ Complete demo failed:', error.message);
      return {
        success: false,
        error: error.message,
        partialResults: this.demoResults
      };
    }
  }

  /**
   * Get demo results summary
   */
  getDemoSummary() {
    return {
      totalScenarios: this.demoResults.length,
      scenarios: this.demoResults.map(demo => ({
        name: demo.scenario,
        status: demo.result.passed !== false ? 'SUCCESS' : 'FAILED',
        notes: demo.demoNotes
      })),
      capabilities: [
        'Object detection validation',
        'Batch processing with metrics',
        'Real-time stream analysis',
        'Annotation quality scoring',
        'Performance benchmarking',
        'Comprehensive reporting'
      ]
    };
  }
}