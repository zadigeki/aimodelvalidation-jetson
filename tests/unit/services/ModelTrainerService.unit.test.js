/**
 * ModelTrainerService Unit Tests
 * London School TDD - Ultralytics integration with behavior verification
 */

import { jest, describe, it, beforeEach, expect } from '@jest/globals';
import { ModelTrainerService } from '../../../src/services/ModelTrainerService.js';
import { createUltralyticsClientMock, createLoggerMock } from '../../mocks/index.js';

describeUnit('ModelTrainerService', () => {
  let modelTrainerService;
  let mockUltralyticsClient;
  let mockLogger;

  beforeEach(() => {
    // London School: Mock contracts drive service design
    mockUltralyticsClient = createUltralyticsClientMock();
    mockLogger = createLoggerMock();

    modelTrainerService = new ModelTrainerService({
      ultralyticsClient: mockUltralyticsClient,
      logger: mockLogger
    });
  });

  describeCollaboration('Training Initialization', () => {
    it('should initialize trainer and coordinate with Ultralytics client', async () => {
      // Act
      const result = await modelTrainerService.initialize();

      // Assert - London School: Focus on collaboration patterns
      expect(mockLogger.info).toHaveBeenCalledWith('Initializing model trainer');
      expect(mockUltralyticsClient.createModel).toHaveBeenCalled();
      expect(result).toBe(true);
    });
  });

  describeCollaboration('Model Training Workflow', () => {
    it('should train model and coordinate training execution', async () => {
      // Arrange
      const trainingConfig = {
        epochs: 100,
        batchSize: 32,
        learningRate: 0.001
      };

      // Act
      const result = await modelTrainerService.train(trainingConfig);

      // Assert - Verify training coordination
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Starting model training',
        trainingConfig
      );
      expect(mockUltralyticsClient.trainModel).toHaveBeenCalled();
      
      // London School: Verify result structure follows contract
      expect(result).toMatchObject({
        modelId: expect.stringMatching(/^yolo-model-\d+$/),
        epochs: 100,
        finalLoss: expect.any(Number),
        metrics: expect.objectContaining({
          mAP: expect.any(Number),
          precision: expect.any(Number),
          recall: expect.any(Number)
        })
      });
    });

    it('should coordinate initialization before training', async () => {
      // Arrange
      await modelTrainerService.initialize();

      // Act
      await modelTrainerService.train({ epochs: 50 });

      // Assert - London School: Verify collaboration sequence
      expect(mockUltralyticsClient.createModel)
        .toHaveBeenCalledBefore(mockUltralyticsClient.trainModel);
    });
  });

  describeCollaboration('Model Validation During Training', () => {
    it('should validate model and coordinate prediction testing', async () => {
      // Arrange
      const modelPath = '/path/to/model.pt';
      const validationData = { test_images: 100 };

      // Act
      const result = await modelTrainerService.validateModel(modelPath, validationData);

      // Assert - Verify validation coordination
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Validating trained model',
        { modelPath }
      );
      expect(mockUltralyticsClient.predict).toHaveBeenCalled();
      
      expect(result).toEqual({
        validation_loss: 0.07,
        validation_metrics: { mAP: 0.84 }
      });
    });
  });

  describeCollaboration('Model Persistence Workflow', () => {
    it('should save model and coordinate export', async () => {
      // Arrange
      const outputPath = '/output/trained-model.pt';

      // Act
      const result = await modelTrainerService.saveModel(outputPath);

      // Assert - Verify save coordination
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Saving trained model',
        { outputPath }
      );
      expect(mockUltralyticsClient.exportModel).toHaveBeenCalled();
      
      expect(result.path).toBeDefined();
    });

    it('should load existing model and coordinate loading', async () => {
      // Arrange
      const modelPath = '/path/to/existing-model.pt';

      // Act
      const result = await modelTrainerService.loadModel(modelPath);

      // Assert - London School: Focus on loading collaboration
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Loading model',
        { modelPath }
      );
      expect(result.loaded).toBe(true);
    });
  });

  describeContract('ModelTrainerService Contract', () => {
    it('should satisfy Ultralytics integration contract', () => {
      const expectedContract = {
        initialize: expect.any(Function),
        train: expect.any(Function),
        validateModel: expect.any(Function),
        saveModel: expect.any(Function),
        loadModel: expect.any(Function)
      };

      expect(modelTrainerService).toSatisfyContract(expectedContract);
    });
  });

  describeCollaboration('Complete Training Pipeline', () => {
    it('should coordinate complete training pipeline sequence', async () => {
      // Arrange
      const trainingConfig = { epochs: 50, batchSize: 16 };

      // Act - Execute complete training pipeline
      await modelTrainerService.initialize();
      const trainingResult = await modelTrainerService.train(trainingConfig);
      await modelTrainerService.validateModel('/model', '/validation-data');
      await modelTrainerService.saveModel('/output/final-model.pt');

      // Assert - London School: Verify complete collaboration sequence
      expect(mockUltralyticsClient.createModel)
        .toHaveBeenCalledBefore(mockUltralyticsClient.trainModel);
      expect(mockUltralyticsClient.trainModel)
        .toHaveBeenCalledBefore(mockUltralyticsClient.predict);
      expect(mockUltralyticsClient.predict)
        .toHaveBeenCalledBefore(mockUltralyticsClient.exportModel);

      // Verify all pipeline steps were logged
      expect(mockLogger.info).toHaveBeenCalledTimes(4);
      expect(trainingResult.modelId).toBeDefined();
    });
  });

  describeCollaboration('Training Error Handling', () => {
    it('should handle training failures gracefully', async () => {
      // Arrange - Mock training failure
      mockUltralyticsClient.trainModel.mockResolvedValue({ 
        training_complete: false,
        error: 'insufficient_data' 
      });

      // Act
      const result = await modelTrainerService.train({ epochs: 10 });

      // Assert - Service should still coordinate properly during failures
      expect(mockUltralyticsClient.trainModel).toHaveBeenCalled();
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Starting model training',
        expect.any(Object)
      );
      
      // London School: Focus on behavior, not implementation details
      expect(result).toBeDefined();
    });
  });
});